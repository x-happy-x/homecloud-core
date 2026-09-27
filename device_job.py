"""Run a configurable, fully local HomeCloud scan job on this device."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback

import catalogdb
import envs
import filekeys
import hublink
import pathkeys
import pathrules
import settings as catalog_settings
from catalog_index import EXCLUDED_NAMES, SUPPORTED, connect as index_db, take_inventory


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')  # свой у каждого процесса
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    # На Windows os.replace изредка натыкается на файл, который в этот момент
    # читает веб-сервер (опрос статуса) — это долей секунды, а не зависание,
    # так что несколько попыток решают дело без риска потерять прогресс.
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


ANALYSIS_FEATURES = ('visual', 'ocr', 'caption', 'adult', 'speech', 'diarize', 'authenticity',
                     'curation')
# Подборки собираются по всему каталогу из готовых оценок — описи не требуют.
CATALOG_FEATURES = ('highlights',)


def planned_phases(features, remote=False, shard=False, resume=False):
    """Этапы задания в том порядке, в котором их на самом деле выполняет run().

    resume — продолжение упавшего задания: опись и ключи копий уже сделаны.
    """
    plan = []
    # Доля параллельного задания не обходит источник: опись уже сделана.
    if not shard and not resume and (features.get('inventory') or (
            not features.get('faces') and any(features.get(name) for name in ANALYSIS_FEATURES))):
        plan.append('inventory')
        # У ядра при хабе обход источника всегда обновляет превью сетки.
        if remote:
            plan.append('thumbs')
    # Ключи файлов и копии — один раз, у хозяина задания, не в долях.
    if not shard and not resume and (
            features.get('faces') or any(features.get(name) for name in ANALYSIS_FEATURES)):
        plan.append('copies')
    if features.get('faces'):
        plan.append('faces')
    if any(features.get(name) for name in ('visual', 'ocr', 'caption')):
        plan.append('visual')
    plan.extend(name for name in ('ocr', 'adult', 'caption', 'speech', 'authenticity', 'diarize',
                                  'curation')
                if features.get(name))
    if features.get('faces') or any(features.get(name) for name in ANALYSIS_FEATURES):
        plan.append('propagate')
    if features.get('highlights'):
        plan.append('highlights')
    return plan


# Замеры прошлых запусков: сколько этап грузит модель и сколько тратит на файл.
# Интерфейс собирает из них оценку «сколько осталось» для всего задания.
TIMINGS_KEEP = 0.6


def timings_file(args):
    return args.progress_file.with_name('device-timings.json')


def timing_key(args, phase):
    """Скорость зависит от вида файлов: ролик в разы дороже снимка."""
    return f'{phase}:{args.kinds.get(phase, "all")}'


def learn_timing(args, phase, record):
    """Законченный без ошибок этап обновляет скользящее среднее своей скорости."""
    started, finished = record.get('started_at'), record.get('finished_at')
    loaded = record.get('loaded_at')
    completed = int(record.get('completed') or 0)
    if not started or not finished or completed <= 0:
        return
    timings = args.timings
    item = dict(timings.get(timing_key(args, phase), {}))
    load = max(0.0, (loaded or started) - started)
    after_load = completed - int(record.get('completed_at_load') or 0)
    work = max(0.0, finished - (loaded or started))
    observed = {'load': load}
    if after_load >= 3 and work > 0:
        observed['per_file'] = work / after_load
    elif completed >= 3:
        observed['per_file'] = max(0.0, finished - started - load) / completed
    for name, value in observed.items():
        old = item.get(name)
        item[name] = value if old is None else old * TIMINGS_KEEP + value * (1 - TIMINGS_KEEP)
    item['runs'] = int(item.get('runs') or 0) + 1
    item['last_total'] = int(record.get('total') or 0)
    timings[timing_key(args, phase)] = item
    write_json(timings_file(args), timings)


def publish(args, status, phase, **extra):
    state = {
        'status': status, 'phase': phase, 'roots': [str(root) for root in args.root],
        'paths': [str(path) for path in args.path],
        'features': args.features, 'kinds': args.kinds, 'pid': os.getpid(),
        'job_started_at': getattr(args, 'job_started_at', time.time()),
        'plan': getattr(args, 'plan', []),
        'phase_history': getattr(args, 'history', {}),
        'timings': getattr(args, 'timings', {}),
        'shard': getattr(args, 'shard', ''),
        'updated_at': time.time(), **extra,
    }
    write_json(args.progress_file, state)


def last_error(log, code):
    """Причина падения этапа: последние строки его вывода."""
    try:
        lines = [line.strip() for line
                 in log.read_text(encoding='utf-8', errors='replace').splitlines()
                 if line.strip()]
    except OSError:
        lines = []
    tail = ' · '.join(lines[-3:])
    return f'Этап завершился с кодом {code}' + (f': {tail}' if tail else '')


def run_child(args, command, phase):
    stage = args.progress_file.with_name('device-stage-progress.json')
    stage.unlink(missing_ok=True)
    phase_started_at = time.time()
    # Этап лиц идёт по разу на каждую папку — история копит их в одну запись.
    record = args.history.setdefault(phase, {'started_at': phase_started_at, 'status': 'running'})
    record['status'] = 'running'
    base_completed = int(record.get('completed') or 0)
    base_total = int(record.get('total') or 0)
    # Вывод этапа раньше уходил в никуда: у web_server.py stdout и stderr —
    # DEVNULL, и упавший этап сообщал только «код 1», без причины. Теперь он
    # пишется в файл рядом с прогрессом, а последние строки идут в задание.
    log = args.progress_file.with_name(f'device-stage-{phase}.log')
    with log.open('w', encoding='utf-8', errors='replace') as sink:
        process = subprocess.Popen(command, cwd=Path(__file__).parent,
                                   stdout=sink, stderr=subprocess.STDOUT)
        while process.poll() is None:
            child = read_json(stage)
            track_child(record, child, base_completed, base_total)
            publish(args, 'running', phase, phase_started_at=phase_started_at, **{
                key: value for key, value in child.items()
                if key not in {'status', 'phase', 'pid', 'updated_at'}
            })
            time.sleep(.5)
    child = read_json(stage)
    track_child(record, child, base_completed, base_total)
    record['finished_at'] = time.time()
    if process.returncode:
        record['status'] = 'error'
        publish(args, 'error', phase, return_code=process.returncode,
                error=child.get('error') or last_error(log, process.returncode))
        raise SystemExit(process.returncode)
    record['status'] = 'stopped' if args.stop_file.exists() else 'done'
    if record['status'] == 'done':
        learn_timing(args, phase, record)


def track_child(record, child, base_completed=0, base_total=0):
    """Счётчики этапа и миг, когда модель загрузилась и пошёл первый файл."""
    completed = base_completed + int(child.get('completed') or 0)
    total = base_total + int(child.get('total') or 0)
    record['completed'] = completed
    record['total'] = max(total, completed)
    if child.get('videos_total') is not None:
        record['videos_total'] = int(child.get('videos_total') or 0)
    if completed > base_completed and not record.get('loaded_at'):
        record['loaded_at'] = time.time()
        record['completed_at_load'] = completed


def inventory(args):
    """Опись: обход выбранных папок и сверка с прошлым разом."""
    db = index_db(args.catalog)

    record = args.history.setdefault('inventory', {'started_at': time.time(), 'status': 'running'})

    def report(done, current, total=0):
        record['completed'], record['total'] = done, max(total, done)
        if done and not record.get('loaded_at'):
            record['loaded_at'], record['completed_at_load'] = time.time(), done
        publish(args, 'running', 'inventory', total=total, completed=done, current=current)

    try:
        # Даже остановленный на середине обход уже что-то нашёл и сохранил —
        # эти цифры возвращаем тоже, а не отбрасываем как пустой результат.
        roots = args.root or sorted({pathkeys.parent(path) for path in args.path})
        summary = take_inventory(db, roots, report=report, stop=args.stop_file.exists,
                                 rules=pathrules.load(args.catalog))
        record['finished_at'] = time.time()
        record['status'] = 'stopped' if summary.get('stopped') else 'done'
        if summary.get('total') is not None:
            record['total'] = int(summary['total'])
            record['completed'] = max(int(record.get('completed') or 0), int(summary['total']))
        if record['status'] == 'done':
            learn_timing(args, 'inventory', record)
        return summary
    finally:
        db.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--root', action='append', type=str, default=[])
    parser.add_argument('--path', action='append', type=str, default=[])
    parser.add_argument('--features', required=True)
    parser.add_argument('--kinds', default='{}',
                        help='JSON «фаза → вид файлов»: all, photos или videos')
    parser.add_argument('--progress-file', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--force', action='store_true',
                        help='Переделать даже то, что уже посчитано для этой версии файла')
    parser.add_argument('--shard', default='',
                        help='«номер/всего»: доля параллельного задания; опись делает хаб отдельно')
    parser.add_argument('--resume', action='store_true',
                        help='Продолжить упавшее задание: без описи и ключей копий, этапы доделают своё')
    args = parser.parse_args()
    args.catalog = args.catalog.resolve()
    # Ключи источников (pc-x:D:\\Фото) остаются как есть, свои пути — абсолютными.
    args.root = [pathkeys.normalize_arg(root) for root in args.root]
    args.path = [pathkeys.normalize_arg(path) for path in args.path]
    if not args.root and not args.path:
        parser.error('provide at least one --root or --path')
    args.features = json.loads(args.features)
    args.kinds = json.loads(args.kinds)
    args.progress_file = args.progress_file.resolve()
    args.stop_file = args.stop_file.resolve()
    return args


def thumbnails(args):
    """Превью сетки и сведения о файлах — на хаб, для нового и изменившегося."""
    here = Path(__file__).resolve().parent
    stage = args.progress_file.with_name('device-stage-progress.json')
    command = [sys.executable, str(here / 'thumbs.py'), '--catalog', str(args.catalog),
               '--progress-file', str(stage), '--stop-file', str(args.stop_file)]
    if args.force:
        command.append('--force')
    for root in args.root:
        command.extend(('--root', root))
    for path in args.path:
        command.extend(('--path', path))
    run_child(args, command, 'thumbs')


def run(args):
    args.job_started_at = time.time()
    args.remote = hublink.is_remote(args.catalog)
    args.plan = planned_phases(args.features, args.remote, bool(args.shard), args.resume)
    args.history = {}
    args.timings = read_json(timings_file(args))
    args.stop_file.unlink(missing_ok=True)
    publish(args, 'preparing', 'inventory', total=0, completed=0)

    # По умолчанию каждый этап пропускает файлы, посчитанные для этой же версии.
    force = ['--force'] if args.force else []

    def kinds_of(phase):
        """Вид файлов этапа. Без карты — всё сразу, как в прежних заданиях."""
        return ['--kinds', args.kinds.get(phase, 'all')]

    def indexed(folder):
        """Есть ли готовая опись этой папки: тогда лица не будут обходить диск."""
        if pathkeys.is_key(folder):
            return ['--use-inventory']
        try:
            db = index_db(args.catalog)
            try:
                row = db.execute('SELECT 1 FROM scan_roots WHERE path=?',
                                 (str(Path(folder).resolve()),)).fetchone()
            finally:
                db.close()
            return ['--use-inventory'] if row else []
        except Exception:
            return []
    if args.features.get('inventory'):
        summary = inventory(args)
        # Даже остановленный на середине обход уже сохранил найденное — эти
        # цифры показываем тоже, а не притворяемся, что ничего не нашли.
        publish(args, 'running', 'inventory', **{
            key: summary[key] for key in ('new', 'changed', 'known', 'missing', 'excluded',
                                          'total', 'bytes') if key in summary})
        if summary.get('stopped'):
            publish(args, 'stopped', 'inventory', inventory=summary)
            return
        if args.remote:
            thumbnails(args)
            if args.stop_file.exists():
                publish(args, 'stopped', 'thumbs', inventory=summary)
                return
        if not any(args.features.get(name) for name in
                   ('faces', *ANALYSIS_FEATURES, *CATALOG_FEATURES)):
            publish(args, 'completed', 'complete', completed=summary.get('total', 0),
                    total=summary.get('total', 0),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    inventory=summary)
            return

    here = Path(__file__).resolve().parent
    worker_root = envs.worker_root(here)
    face_python = Path(sys.executable)
    vision_python = worker_root / 'vision-venv' / 'Scripts' / 'python.exe'
    ocr_python = envs.ocr_python()
    hf_home = str(envs.hf_home(here))
    audio_python = worker_root / 'audio-venv' / 'Scripts' / 'python.exe'
    imgutils_python = worker_root / 'imgutils-venv' / 'Scripts' / 'python.exe'
    stage = args.progress_file.with_name('device-stage-progress.json')
    if args.shard:
        # Этапы ниже берут файлы через pathkeys.scope_sql/shard_sql — только свою долю.
        os.environ[pathkeys.SHARD_ENV] = args.shard

    per_file = args.features.get('faces') or any(args.features.get(name) for name in ANALYSIS_FEATURES)
    root_keys = [item for root in args.root for item in ('--root', str(root))] +         [item for path in args.path for item in ('--path', str(path))]
    if per_file:
        # Таблица копий нужна этапам и в долях задания — создаём её, даже если
        # ключи считает не это ядро.
        db = catalogdb.connect(args.catalog, timeout=60)
        try:
            filekeys.ensure_schema(db)
        finally:
            db.close()
    if per_file and not args.shard and not args.resume:
        # Копии файлов этапы не считают: оригинал — один раз, результат переносится.
        run_child(args, [str(face_python), str(here / 'filekeys.py'), 'keys',
                  '--catalog', str(args.catalog), '--progress-file', str(stage),
                  '--stop-file', str(args.stop_file), *force, *root_keys], 'copies')
        if args.stop_file.exists():
            publish(args, 'stopped', 'copies')
            return

    if args.features.get('faces') and args.path:
        parents = {}
        for path in args.path:
            parents.setdefault(pathkeys.parent(path), []).append(path)
        for root, paths in parents.items():
            # Файлы источника берутся из описи: сам источник лица не обходят.
            keyed = ['--use-inventory'] if pathkeys.is_key(root) else []
            command = [str(face_python), str(here / 'prototype.py'), 'scan',
                       '--photos', str(root), '--models', str(here / 'models' / 'buffalo_l'),
                       '--data', str(args.catalog), '--limit', str(len(paths)), '--min-side', '160',
                       '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                       *force, *keyed, *kinds_of('faces')]
            for path in paths:
                command.extend(('--include-path', str(path)))
            run_child(args, command, 'faces')
            if args.stop_file.exists():
                publish(args, 'stopped', 'faces')
                return
    elif args.features.get('faces'):
        for root in args.root:
            command = [str(face_python), str(here / 'prototype.py'), 'scan',
                       '--photos', str(root), '--models', str(here / 'models' / 'buffalo_l'),
                       '--data', str(args.catalog), '--limit', '1000000', '--min-side', '160',
                       '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                       *force, *indexed(root), *kinds_of('faces')]
            for name in sorted(EXCLUDED_NAMES):
                command.extend(('--exclude-dir-name', name))
            run_child(args, command, 'faces')
            if args.stop_file.exists():
                publish(args, 'stopped', 'faces')
                return
    elif (any(args.features.get(name) for name in ANALYSIS_FEATURES)
          and not args.features.get('inventory') and not args.shard and not args.resume):
        # Опись уже прошла в начале задания — второй раз обходить источник незачем.
        if inventory(args).get('stopped'):
            publish(args, 'stopped', 'inventory')
            return
        if args.remote:
            thumbnails(args)
            if args.stop_file.exists():
                publish(args, 'stopped', 'thumbs')
                return

    root_args = [item for root in args.root for item in ('--root', str(root))]
    root_args.extend(item for path in args.path for item in ('--path', str(path)))
    if any(args.features.get(name) for name in ('visual', 'ocr', 'caption')):
        options = catalog_settings.load(args.catalog)
        env = os.environ.copy()
        env.update(HF_HOME=hf_home, HF_HUB_OFFLINE='1', PYTHONUTF8='1')
        command = [str(vision_python), str(here / 'analyze_photos.py'), 'analyze',
                   '--catalog', str(args.catalog), '--limit', '100000000', '--batch-size', '24',
                   '--model', options['visual_model'],
                   '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                   *force, *kinds_of('visual'), *root_args]
        old_env = os.environ.copy()
        os.environ.update(env)
        try:
            run_child(args, command, 'visual')
        finally:
            os.environ.clear()
            os.environ.update(old_env)
    if args.stop_file.exists():
        publish(args, 'stopped', 'visual')
        return

    if args.features.get('ocr'):
        os.environ.update(PADDLE_PDX_CACHE_HOME=str(envs.models_root(here) / 'paddle'),
                          PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True', PYTHONUTF8='1')
        run_child(args, [str(ocr_python), str(here / 'ocr_photos.py'),
                  '--catalog', str(args.catalog), '--limit', '100000000',
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *kinds_of('ocr'), *root_args], 'ocr')
    if args.stop_file.exists():
        publish(args, 'stopped', 'ocr')
        return

    # WD/NudeNet идут до описания: тогда Qwen получает уже готовые локальные
    # рейтинги и только самые уверенные теги как вспомогательный контекст.
    if args.features.get('adult'):
        os.environ.update(HF_HOME=hf_home, HF_HUB_OFFLINE='1', PYTHONUTF8='1')
        run_child(args, [str(vision_python), str(here / 'adult_photos.py'),
                  '--catalog', str(args.catalog), '--limit', '100000000',
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *kinds_of('adult'), *root_args], 'adult')
    if args.stop_file.exists():
        publish(args, 'stopped', 'adult')
        return

    if args.features.get('caption'):
        os.environ.update(HF_HOME=hf_home, HF_HUB_OFFLINE='1', PYTHONUTF8='1')
        options = catalog_settings.load(args.catalog)
        run_child(args, [str(vision_python), str(here / 'caption_photos.py'),
                  '--catalog', str(args.catalog), '--limit', '100000000',
                  '--backend', options['caption_backend'], '--model', options['caption_model'],
                  '--lmstudio-url', options['caption_lmstudio_url'],
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *kinds_of('caption'), *root_args], 'caption')
    if args.stop_file.exists():
        publish(args, 'stopped', 'caption')
        return

    if args.features.get('speech'):
        options = catalog_settings.load(args.catalog)
        # Предыдущие этапы включили офлайн для Hugging Face; здесь он мешает
        # забрать модель распознавания, если её ещё нет.
        os.environ.update(HF_HOME=hf_home, HF_HUB_OFFLINE='0',
                          HF_HUB_DISABLE_SYMLINKS_WARNING='1', PYTHONUTF8='1')
        run_child(args, [str(audio_python), str(here / 'speech_videos.py'),
                  '--catalog', str(args.catalog), '--limit', '100000000',
                  '--model', options['speech_model'],
                  '--language', options['speech_language'],
                  '--fallback-language', options['speech_fallback_language'],
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *root_args], 'speech')
    if args.stop_file.exists():
        publish(args, 'stopped', 'speech')
        return

    if args.features.get('authenticity'):
        run_child(args, [str(imgutils_python), str(here / 'authenticity_photos.py'),
                  '--catalog', str(args.catalog), '--limit', '1000000',
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *kinds_of('authenticity'), *root_args], 'authenticity')
    if args.stop_file.exists():
        publish(args, 'stopped', 'authenticity')
        return

    if args.features.get('diarize'):
        # Тот же офлайн-манёвр, что и у речи: закачка модели должна идти в сеть.
        os.environ.update(HF_HOME=hf_home, HF_HUB_OFFLINE='0',
                          HF_HUB_DISABLE_SYMLINKS_WARNING='1', PYTHONUTF8='1')
        run_child(args, [str(audio_python), str(here / 'speaker_diarization.py'),
                  '--catalog', str(args.catalog), '--limit', '100000000',
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *root_args], 'diarize')
    if args.stop_file.exists():
        publish(args, 'stopped', 'diarize')
        return

    # Оценка снимков для подборок — после всех этапов, чьи результаты она
    # читает: визуального индекса, 18+, лиц и проверки рисованных лиц.
    if args.features.get('curation'):
        options = catalog_settings.load(args.catalog)
        prompts_log = args.progress_file.with_name('device-stage-curation-prompts.log')
        if vision_python.is_file():
            # Векторы текстовых описаний «удачного кадра» считаются один раз на
            # модель в окружении визуального индекса; если они уже есть, скрипт
            # выходит, не загружая модель. Ошибка здесь не роняет этап: без
            # описаний оценка обойдётся одной технической частью.
            publish(args, 'running', 'curation', total=0, completed=0)
            prompts_env = {**os.environ, 'HF_HOME': hf_home,
                           'HF_HUB_OFFLINE': '1', 'PYTHONUTF8': '1'}
            with prompts_log.open('w', encoding='utf-8', errors='replace') as sink:
                subprocess.run([str(vision_python), str(here / 'photo_curation.py'), 'prompts',
                                '--catalog', str(args.catalog), '--model', options['visual_model'],
                                '--indexed'],
                               cwd=here, env=prompts_env, stdout=sink, stderr=subprocess.STDOUT)
        run_child(args, [str(face_python), str(here / 'photo_curation.py'), 'curate',
                  '--catalog', str(args.catalog),
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *root_args], 'curation')
    if args.stop_file.exists():
        publish(args, 'stopped', 'curation')
        return

    if per_file:
        run_child(args, [str(face_python), str(here / 'filekeys.py'), 'propagate',
                  '--catalog', str(args.catalog), '--progress-file', str(stage),
                  '--stop-file', str(args.stop_file)], 'propagate')
    if args.stop_file.exists():
        publish(args, 'stopped', 'propagate')
        return

    # Подборки строятся по всему каталогу из готовых оценок: без файлов и моделей.
    if args.features.get('highlights'):
        run_child(args, [str(face_python), str(here / 'highlight_generator.py'), 'generate',
                  '--catalog', str(args.catalog),
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force], 'highlights')
    if args.stop_file.exists():
        publish(args, 'stopped', 'highlights')
        return

    publish(args, 'completed', 'complete', completed=1, total=1,
            finished_at=datetime.now(timezone.utc).isoformat())


def main():
    args = parse_args()
    try:
        run(args)
    except SystemExit:
        raise
    except Exception:
        # stderr процесса уходит в DEVNULL (его никто не читает), поэтому без
        # этого падение задания выглядело бы так, будто оно тихо ничего не сделало.
        publish(args, 'error', 'device', error=traceback.format_exc(limit=8))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
