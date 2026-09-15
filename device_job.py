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

import pathrules
import settings as catalog_settings
from catalog_index import EXCLUDED_NAMES, SUPPORTED, connect as index_db, take_inventory


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
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


def publish(args, status, phase, **extra):
    state = {
        'status': status, 'phase': phase, 'roots': [str(root) for root in args.root],
        'paths': [str(path) for path in args.path],
        'features': args.features, 'pid': os.getpid(),
        'job_started_at': getattr(args, 'job_started_at', time.time()),
        'updated_at': time.time(), **extra,
    }
    write_json(args.progress_file, state)


def run_child(args, command, phase):
    stage = args.progress_file.with_name('device-stage-progress.json')
    stage.unlink(missing_ok=True)
    phase_started_at = time.time()
    process = subprocess.Popen(command, cwd=Path(__file__).parent)
    while process.poll() is None:
        child = read_json(stage)
        publish(args, 'running', phase, phase_started_at=phase_started_at, **{
            key: value for key, value in child.items()
            if key not in {'status', 'phase', 'pid', 'updated_at'}
        })
        time.sleep(.5)
    child = read_json(stage)
    if process.returncode:
        publish(args, 'error', phase, return_code=process.returncode,
                error=child.get('error', f'Этап завершился с кодом {process.returncode}'))
        raise SystemExit(process.returncode)


def inventory(args):
    """Опись: обход выбранных папок и сверка с прошлым разом."""
    db = index_db(args.catalog)

    def report(done, current, total=0):
        publish(args, 'running', 'inventory', total=total, completed=done, current=current)

    try:
        # Даже остановленный на середине обход уже что-то нашёл и сохранил —
        # эти цифры возвращаем тоже, а не отбрасываем как пустой результат.
        return take_inventory(db, args.root or [path.parent for path in args.path],
                              report=report, stop=args.stop_file.exists,
                              rules=pathrules.load(args.catalog))
    finally:
        db.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--root', action='append', type=Path, default=[])
    parser.add_argument('--path', action='append', type=Path, default=[])
    parser.add_argument('--features', required=True)
    parser.add_argument('--progress-file', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--force', action='store_true',
                        help='Переделать даже то, что уже посчитано для этой версии файла')
    args = parser.parse_args()
    args.catalog = args.catalog.resolve()
    args.root = [root.resolve() for root in args.root]
    args.path = [path.resolve() for path in args.path]
    if not args.root and not args.path:
        parser.error('provide at least one --root or --path')
    args.features = json.loads(args.features)
    args.progress_file = args.progress_file.resolve()
    args.stop_file = args.stop_file.resolve()
    return args


def run(args):
    args.job_started_at = time.time()
    args.stop_file.unlink(missing_ok=True)
    publish(args, 'preparing', 'inventory', total=0, completed=0)

    # По умолчанию каждый этап пропускает файлы, посчитанные для этой же версии.
    force = ['--force'] if args.force else []

    def indexed(folder):
        """Есть ли готовая опись этой папки: тогда лица не будут обходить диск."""
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
        if not any(args.features.get(name) for name in
                   ('faces', 'visual', 'ocr', 'caption', 'adult', 'speech', 'diarize',
                    'authenticity')):
            publish(args, 'completed', 'complete', completed=summary.get('total', 0),
                    total=summary.get('total', 0),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    inventory=summary)
            return

    here = Path(__file__).resolve().parent
    workspace = here.parents[1]
    face_python = Path(sys.executable)
    vision_python = workspace / 'work' / 'vision-venv' / 'Scripts' / 'python.exe'
    ocr_python = Path(r'C:\cv-ocr\Scripts\python.exe')
    audio_python = workspace / 'work' / 'audio-venv' / 'Scripts' / 'python.exe'
    imgutils_python = workspace / 'work' / 'imgutils-venv' / 'Scripts' / 'python.exe'
    stage = args.progress_file.with_name('device-stage-progress.json')

    if args.features.get('faces') and args.path:
        parents = {}
        for path in args.path:
            parents.setdefault(path.parent, []).append(path)
        for root, paths in parents.items():
            command = [str(face_python), str(here / 'prototype.py'), 'scan',
                       '--photos', str(root), '--models', str(here / 'models' / 'buffalo_l'),
                       '--data', str(args.catalog), '--limit', str(len(paths)), '--min-side', '160',
                       '--progress-file', str(stage), '--stop-file', str(args.stop_file), *force]
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
                       *force, *indexed(root)]
            for name in sorted(EXCLUDED_NAMES):
                command.extend(('--exclude-dir-name', name))
            run_child(args, command, 'faces')
            if args.stop_file.exists():
                publish(args, 'stopped', 'faces')
                return
    elif any(args.features.get(name) for name in
             ('visual', 'ocr', 'caption', 'adult', 'speech', 'diarize', 'authenticity')):
        if inventory(args).get('stopped'):
            publish(args, 'stopped', 'inventory')
            return

    root_args = [item for root in args.root for item in ('--root', str(root))]
    root_args.extend(item for path in args.path for item in ('--path', str(path)))
    if any(args.features.get(name) for name in ('visual', 'ocr', 'caption')):
        options = catalog_settings.load(args.catalog)
        env = os.environ.copy()
        env.update(HF_HOME=r'C:\cv-models\huggingface', HF_HUB_OFFLINE='1', PYTHONUTF8='1')
        command = [str(vision_python), str(here / 'analyze_photos.py'), 'analyze',
                   '--catalog', str(args.catalog), '--limit', '100000000', '--batch-size', '24',
                   '--model', options['visual_model'],
                   '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                   *force, *root_args]
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
        os.environ.update(PADDLE_PDX_CACHE_HOME=r'C:\cv-models\paddle',
                          PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True', PYTHONUTF8='1')
        run_child(args, [str(ocr_python), str(here / 'ocr_photos.py'),
                  '--catalog', str(args.catalog), '--limit', '100000000',
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *root_args], 'ocr')
    if args.stop_file.exists():
        publish(args, 'stopped', 'ocr')
        return

    # WD/NudeNet идут до описания: тогда Qwen получает уже готовые локальные
    # рейтинги и только самые уверенные теги как вспомогательный контекст.
    if args.features.get('adult'):
        os.environ.update(HF_HOME=r'C:\cv-models\huggingface', HF_HUB_OFFLINE='1', PYTHONUTF8='1')
        run_child(args, [str(vision_python), str(here / 'adult_photos.py'),
                  '--catalog', str(args.catalog), '--limit', '100000000',
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *root_args], 'adult')
    if args.stop_file.exists():
        publish(args, 'stopped', 'adult')
        return

    if args.features.get('caption'):
        os.environ.update(HF_HOME=r'C:\cv-models\huggingface', HF_HUB_OFFLINE='1', PYTHONUTF8='1')
        options = catalog_settings.load(args.catalog)
        run_child(args, [str(vision_python), str(here / 'caption_photos.py'),
                  '--catalog', str(args.catalog), '--limit', '100000000',
                  '--backend', options['caption_backend'], '--model', options['caption_model'],
                  '--lmstudio-url', options['caption_lmstudio_url'],
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *root_args], 'caption')
    if args.stop_file.exists():
        publish(args, 'stopped', 'caption')
        return

    if args.features.get('speech'):
        options = catalog_settings.load(args.catalog)
        # Предыдущие этапы включили офлайн для Hugging Face; здесь он мешает
        # забрать модель распознавания, если её ещё нет.
        os.environ.update(HF_HOME=r'C:\cv-models\huggingface', HF_HUB_OFFLINE='0',
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
                  *force, *root_args], 'authenticity')
    if args.stop_file.exists():
        publish(args, 'stopped', 'authenticity')
        return

    if args.features.get('diarize'):
        # Тот же офлайн-манёвр, что и у речи: закачка модели должна идти в сеть.
        os.environ.update(HF_HOME=r'C:\cv-models\huggingface', HF_HUB_OFFLINE='0',
                          HF_HUB_DISABLE_SYMLINKS_WARNING='1', PYTHONUTF8='1')
        run_child(args, [str(audio_python), str(here / 'speaker_diarization.py'),
                  '--catalog', str(args.catalog), '--limit', '100000000',
                  '--progress-file', str(stage), '--stop-file', str(args.stop_file),
                  *force, *root_args], 'diarize')
    if args.stop_file.exists():
        publish(args, 'stopped', 'diarize')
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
