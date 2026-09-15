"""Local face-catalog experiment. Does not download models or upload photos."""
import argparse
import fnmatch
import hashlib
import html
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import pathrules
import settings as catalog_settings
import video as video_media
import video_tracks

PICTURES = {'.jpg', '.jpeg', '.png', '.webp'}
SUPPORTED = PICTURES | video_media.SUPPORTED
DLL_HANDLES = []


def configure_gpu_runtime():
    """Keep NVIDIA wheel DLL directories registered for this process."""
    if os.name != 'nt' or not hasattr(os, 'add_dll_directory'):
        return
    nvidia = Path(sys.prefix) / 'Lib' / 'site-packages' / 'nvidia'
    if not nvidia.is_dir():
        return
    directories = sorted(nvidia.glob('*/bin'))
    os.environ['PATH'] = os.pathsep.join([*(str(p) for p in directories), os.environ.get('PATH', '')])
    for directory in directories:
        DLL_HANDLES.append(os.add_dll_directory(str(directory)))


def database(folder, check_same_thread=True):
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / 'catalog.sqlite', check_same_thread=check_same_thread)
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS photos (
          path TEXT PRIMARY KEY, size INTEGER, modified INTEGER, model TEXT,
          status TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS faces (
          id INTEGER PRIMARY KEY, path TEXT REFERENCES photos(path),
          box TEXT, embedding BLOB, thumbnail TEXT);
        CREATE TABLE IF NOT EXISTS face_clusters (
          face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
          label INTEGER NOT NULL, probability REAL NOT NULL DEFAULT 0,
          method TEXT NOT NULL DEFAULT 'hdbscan', computed_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS face_clusters_label ON face_clusters(label);
    ''')
    # Видео пришло позже фотографий: досыпаем колонки в готовый каталог.
    columns = {row[1] for row in db.execute('PRAGMA table_info(photos)')}
    if 'kind' not in columns:
        db.execute("ALTER TABLE photos ADD COLUMN kind TEXT NOT NULL DEFAULT 'photo'")
    if 'duration' not in columns:
        db.execute('ALTER TABLE photos ADD COLUMN duration REAL')
    face_columns = {row[1] for row in db.execute('PRAGMA table_info(faces)')}
    if 'frame_time' not in face_columns:
        # Лицо из ролика помнит, на какой секунде его нашли.
        db.execute('ALTER TABLE faces ADD COLUMN frame_time REAL')
    if 'track_start' not in face_columns:
        # Трек — не один момент, а промежуток: с какой секунды по какую
        # лицо было в кадре. У фотографий и старых записей — NULL.
        db.execute('ALTER TABLE faces ADD COLUMN track_start REAL')
        db.execute('ALTER TABLE faces ADD COLUMN track_stop REAL')
    db.commit()
    return db


def doctor():
    print('Python:', sys.version)
    subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                    '--format=csv'], check=False)
    configure_gpu_runtime()
    import onnxruntime as ort
    ort.disable_telemetry_events()
    ort.preload_dlls(directory='')
    print('ONNX Runtime:', ort.__version__)
    print('Available providers:', ort.get_available_providers())
    print('Provider availability alone does not verify GPU inference. Run scan.')


def load_models(folder):
    import numpy as np
    configure_gpu_runtime()
    import onnxruntime as ort
    ort.disable_telemetry_events()
    ort.preload_dlls(directory='')
    from insightface.model_zoo.model_zoo import ModelRouter
    files = sorted(folder.glob('*.onnx'))
    if not files:
        raise ValueError('No local ONNX files. See README.md.')
    digest = hashlib.sha256()
    models = {}
    for path in files:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(folder / (path.stem + '-warmup'))
        model = ModelRouter(str(path.resolve())).get_model(
            sess_options=options, providers=['CUDAExecutionProvider'])
        if model is not None and model.taskname in {'detection', 'recognition'}:
            if 'CUDAExecutionProvider' not in model.session.get_providers():
                raise RuntimeError(f'CUDA initialization failed: {path.name}')
            models[model.taskname] = model
    if set(models) != {'detection', 'recognition'}:
        raise ValueError('Need compatible SCRFD detector and ArcFace recognition ONNX models.')
    models['detection'].prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.6)
    models['recognition'].prepare(ctx_id=0)
    # Exercise both networks even if the first photo contains no faces.
    models['detection'].detect(np.zeros((640, 640, 3), dtype=np.uint8))
    models['recognition'].get_feat(np.zeros((112, 112, 3), dtype=np.uint8))
    for name, model in models.items():
        profile = Path(model.session.end_profiling())
        if not profile.is_file():
            raise RuntimeError(f'{name}: runtime profile was not created')
        events = json.loads(profile.read_text(encoding='utf-8'))
        gpu_ops = sum(event.get('args', {}).get('provider') == 'CUDAExecutionProvider' for event in events)
        if not gpu_ops:
            raise RuntimeError(f'{name}: no GPU operations found in runtime profile')
        print(f'{name}: verified {gpu_ops} CUDA operations during warm-up')
    return models, digest.hexdigest()


def scan(args):
    import numpy as np
    from PIL import Image, ImageOps
    from insightface.app.common import Face
    root = args.photos.resolve()
    data = args.data.resolve()
    progress_file = args.progress_file.resolve() if args.progress_file else None
    stop_file = args.stop_file.resolve() if args.stop_file else None
    started_at = time.time()
    last_progress_write = 0.0
    if not root.is_dir():
        raise ValueError('Photo directory does not exist')
    if data == root or root in data.parents:
        raise ValueError('Keep catalog/cache outside the source photo directory')
    excluded_roots = [path.resolve() for path in args.exclude_path]
    excluded_names = {name.casefold() for name in args.exclude_dir_name}
    excluded_patterns = [pattern.casefold() for pattern in args.exclude_file_pattern]
    included_paths = {path.resolve() for path in args.include_path}
    if any(root != path.parent and root not in path.parents for path in included_paths):
        raise ValueError('Included photo must be inside the photo directory')
    options = catalog_settings.load(data)
    block_rules, allow_rules = pathrules.prepare(options)
    # Порог из командной строки перебивает настройку каталога.
    min_side = args.min_side or options['min_side']
    ignore_patterns = catalog_settings.patterns(options)
    excluded_patterns = excluded_patterns + ignore_patterns
    filter_settings = {
        'exclude_paths': sorted(str(path).casefold() for path in excluded_roots),
        'exclude_dir_names': sorted(excluded_names),
        'exclude_file_patterns': sorted(excluded_patterns),
        'min_side': min_side,
    }
    # Исключения каталога — общие для всех этапов, их ставят в дереве описи.
    excluded_files = set()
    try:
        from catalog_index import connect as index_db, exclusions as catalog_exclusions
        index = index_db(data)
        try:
            excluded_dirs, excluded_files = catalog_exclusions(index)
        finally:
            index.close()
        excluded_roots = excluded_roots + [Path(item) for item in excluded_dirs]
    except Exception as exc:  # каталога может ещё не быть
        print(f'Exclusions unavailable: {exc}', file=sys.stderr, flush=True)

    paths = []
    processed = ignored = skipped = errors = faces_found = videos_done = videos_total = 0

    def is_excluded(path):
        resolved = path.resolve()
        return any(resolved == excluded or excluded in resolved.parents for excluded in excluded_roots)

    def publish(status, current='', force=False):
        nonlocal last_progress_write
        if progress_file is None:
            return
        now = time.time()
        if not force and now - last_progress_write < 0.2:
            return
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'status': status, 'source': str(root), 'catalog': str(data),
            'current': current, 'total': len(paths), 'found': len(paths),
            'completed': processed + ignored + skipped + errors,
            'processed': processed, 'ignored': ignored, 'skipped': skipped,
            'errors': errors, 'faces_found': faces_found, 'videos_done': videos_done,
            'videos_total': videos_total, 'video_track_step': options['video_track_step'],
            'pid': os.getpid(), 'started_at': started_at, 'updated_at': now,
        }
        temporary = progress_file.with_suffix(progress_file.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        # Файл прогресса читает device_job.py каждые полсекунды — os.replace изредка
        # натыкается на этот момент чтения (WinError 5), несколько попыток решают дело.
        for attempt in range(5):
            try:
                os.replace(temporary, progress_file)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
        last_progress_write = now

    publish('counting', force=True)
    if args.use_inventory:
        # Опись уже обошла диск — берём готовый список вместо повторного обхода.
        from catalog_index import bounds
        index = database(data)
        low, high = bounds(root)
        for (found,) in index.execute(
                'SELECT path FROM photos WHERE path>=? AND path<? '
                "AND status NOT IN ('excluded','missing') ORDER BY path",
                (low, high)):
            candidate = Path(found)
            if pathrules.blocked(found, block_rules, allow_rules):
                continue
            if not options['video_enabled'] and video_media.is_video(candidate):
                continue
            if included_paths and candidate.resolve() not in included_paths:
                continue
            if any(fnmatch.fnmatch(candidate.name.casefold(), pattern)
                   for pattern in excluded_patterns):
                continue
            paths.append(candidate)
        index.close()
        print(f'Using catalog inventory: {len(paths)} files.', flush=True)
        publish('counting', str(root), force=True)
    else:
        seen = 0
        # Junction/симлинк может замкнуть дерево само на себя — без защиты
        # обход одной и той же папки повторяется бесконечно.
        visited = set()
        for current, directories, filenames in os.walk(root, followlinks=False):
            try:
                identity = os.stat(current)
                key = (identity.st_dev, identity.st_ino)
            except OSError:
                key = None
            if key is not None:
                if key in visited:
                    directories[:] = []
                    continue
                visited.add(key)
            directories[:] = sorted(
                directory for directory in directories
                if directory.casefold() not in excluded_names
                and not is_excluded(Path(current) / directory)
                # Правило закрывает подпапку целиком — сюда обход не спускается
                # вовсе, а не просто пропускает её файлы задним числом.
                and pathrules.enter(str(Path(current) / directory), block_rules, allow_rules)
            )
            if pathrules.blocked(current, block_rules, allow_rules):
                continue
            paths.extend(
                Path(current) / filename for filename in sorted(filenames)
                if (Path(filename).suffix.lower() in (
                        SUPPORTED if options['video_enabled'] else PICTURES)
                    and not (Path(current) / filename).is_symlink()
                    and (not included_paths or (Path(current) / filename).resolve() in included_paths)
                    and not any(fnmatch.fnmatch(filename.casefold(), pattern)
                                for pattern in excluded_patterns)
                    and not pathrules.blocked(Path(current) / filename,
                                              block_rules, allow_rules))
            )
            seen += 1
            # Обход большой папки занимает минуты: показываем, что работа идёт.
            if seen % 20 == 0:
                publish('counting', str(current), force=True)

    publish('preparing', force=True)
    try:
        models, signature = load_models(args.models.resolve())
    except Exception:
        publish('error', force=True)
        raise
    # Списки исключений решают, какие файлы обходить, и не меняют результат для
    # отдельного файла — держать их в подписи значит зря пересканировать всё.
    # Порог min-side результат меняет, поэтому он в подписи остаётся.
    def stamp(version, settings):
        payload = json.dumps(settings, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(f'{signature}|{version}|{payload}'.encode()).hexdigest()

    legacy = set()
    if any((excluded_roots, excluded_names, excluded_patterns, min_side)):
        legacy.add(stamp('scan-filters-v1', filter_settings))
        legacy.add(stamp('scan-filters-v1', {**filter_settings, 'exclude_paths': [],
                                             'exclude_dir_names': [], 'exclude_file_patterns': []}))
    if min_side:
        signature = stamp('scan-filters-v2', {'min_side': min_side})
    legacy.discard(signature)
    # У роликов своя подпись: поменяли параметры трекинга — пересчитываются
    # только они, фотографии остаются нетронутыми.
    video_signature = stamp('video-tracks-v1', {
        'step': options['video_track_step'], 'gap': options['video_track_gap'],
        'best': options['video_track_best'], 'min_side': min_side,
        'from': options['video_min_seconds'], 'to': options['video_max_seconds']})

    db = database(data)
    pathrules.ensure_column(db)
    if legacy:
        # Те же модель и порог — результат тот же, переносим отметки без пересчёта.
        with db:
            moved = db.execute(
                'UPDATE photos SET model=? WHERE model IN (' + ','.join('?' * len(legacy)) + ')',
                (signature, *sorted(legacy))).rowcount
        if moved:
            print(f'Cache signature migrated for {moved} photos.', flush=True)
    thumbs = data / 'thumbnails'
    videos_total = sum(video_media.is_video(path) for path in paths)
    thumbs.mkdir(exist_ok=True)
    publish('running', force=True)
    for path in paths:
        filename = path.name
        if stop_file and stop_file.exists():
            publish('stopped', str(path), force=True)
            print('Stopped from Web UI. Re-run scan to continue.', file=sys.stderr)
            return
        if processed + ignored + errors >= args.limit:
            print('Sample limit reached. Re-run to continue.')
            print(f'Processed={processed}, ignored={ignored}, skipped={skipped}, errors={errors}')
            publish('limited', str(path), force=True)
            return
        stat = None
        try:
                stat = path.stat()
                key = str(path)
                if key in excluded_files:
                    skipped += 1
                    publish('running', str(path))
                    continue
                is_video = video_media.is_video(path)
                kind_signature = video_signature if is_video else signature
                old = db.execute('SELECT size,modified,model,status FROM photos WHERE path=?',
                                 (key,)).fetchone()
                if old and old[3] == 'excluded':
                    skipped += 1
                    publish('running', str(path))
                    continue
                if (not args.force and old
                        and old[:3] == (stat.st_size, stat.st_mtime_ns, kind_signature)
                        and old[3] in {'ok', 'ignored'}):
                    skipped += 1
                    publish('running', str(path))
                    continue

                def find_faces(image, moment=None, tag=''):
                    """Лица одного кадра: рамка, вектор и вырезанный портрет."""
                    found = []
                    frame = np.asarray(image)[:, :, ::-1].copy()
                    boxes, landmarks = models['detection'].detect(frame)
                    for number, box in enumerate(boxes):
                        if landmarks is None:
                            continue
                        face = Face(bbox=box[:4], kps=landmarks[number], det_score=box[4])
                        models['recognition'].get(frame, face)
                        embedding = np.asarray(face.normed_embedding, dtype='<f4')
                        if not np.all(np.isfinite(embedding)):
                            continue
                        token = hashlib.sha256(
                            f'{key}:{stat.st_mtime_ns}:{kind_signature}:{tag}{number}'
                            .encode()).hexdigest()
                        thumbnail = f'thumbnails/{token}.jpg'
                        crop = image.crop(tuple(int(value) for value in box[:4]))
                        crop.thumbnail((160, 160))
                        crop.save(data / thumbnail)
                        found.append((key, json.dumps(box[:4].tolist()), embedding.tobytes(),
                                      thumbnail, moment))
                    return found

                def mark(status, note=None, kind='photo', duration=None):
                    with db:
                        db.execute(
                            'INSERT INTO photos(path,dir,size,modified,model,status,error,kind,duration) '
                            'VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET '
                            'dir=excluded.dir,size=excluded.size,modified=excluded.modified,'
                            'model=excluded.model,status=excluded.status,error=excluded.error,'
                            'kind=excluded.kind,duration=excluded.duration',
                            (key, str(path.parent), stat.st_size, stat.st_mtime_ns,
                             kind_signature, status, note, kind, duration))
                        if status == 'ignored':
                            db.execute('DELETE FROM faces WHERE path=?', (key,))

                results, duration, img = [], None, None
                if is_video:
                    info = video_media.probe(path)
                    duration = info['duration'] or None
                    short = (options['video_min_seconds']
                             and duration and duration < options['video_min_seconds'])
                    reason = short and 'ролик короче минимальной длительности' or \
                        catalog_settings.rejects(options, size=stat.st_size,
                                                 width=info['width'], height=info['height'])
                    if reason:
                        mark('ignored', reason, 'video', duration)
                        ignored += 1
                        publish('running', str(path))
                        continue
                    tracks = video_tracks.find_tracks(
                        path, models, step_seconds=options['video_track_step'],
                        gap_seconds=options['video_track_gap'],
                        best_frames=options['video_track_best'],
                        stop_seconds=options['video_max_seconds'],
                        stop_check=lambda: bool(stop_file and stop_file.exists()))
                    for number, track in enumerate(tracks):
                        token = hashlib.sha256(
                            f'{key}:{stat.st_mtime_ns}:{kind_signature}:{number}'.encode()
                        ).hexdigest()
                        thumbnail = f'thumbnails/{token}.jpg'
                        track['extra'].save(data / thumbnail)
                        results.append((
                            key, json.dumps([round(value, 1) for value in track['box']]),
                            track['embedding'].tobytes(), thumbnail,
                            round(track['frame_time'], 3),
                            round(track['start'], 3), round(track['stop'], 3)))
                    print(f'  {filename}: треков {len(tracks)}', flush=True)
                else:
                    with Image.open(path) as original:
                        img = ImageOps.exif_transpose(original).convert('RGB')
                    reason = catalog_settings.rejects(
                        options, size=stat.st_size, width=img.size[0], height=img.size[1])
                    if min_side and min(img.size) < min_side:
                        reason = reason or 'сторона меньше минимальной'
                    if reason:
                        mark('ignored', reason)
                        ignored += 1
                        publish('running', str(path))
                        continue
                    results = find_faces(img)
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise RuntimeError('File changed during processing; retry scan')
                mark('ok', None, 'video' if is_video else 'photo', duration)
                with db:
                    merge_faces(db, key, results)
                processed += 1
                faces_found += len(results)
                if is_video:
                    videos_done += 1
                print(f'{processed}: {filename}: {len(results)} faces', flush=True)
                publish('running', str(path))
        except Exception as exc:
            errors += 1
            print(f'ERROR: {path}: {exc}', file=sys.stderr, flush=True)
            with db:
                db.execute('INSERT INTO photos(path,dir,size,modified,model,status,error) VALUES(?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET status=excluded.status,error=excluded.error',
                           (str(path), str(path.parent), stat.st_size if stat else None,
                            stat.st_mtime_ns if stat else None, signature, 'error', str(exc)))
            publish('running', str(path))
    print(f'Processed={processed}, ignored={ignored}, skipped={skipped}, errors={errors}')
    publish('completed', force=True)


def overlap(first, second):
    """IoU двух рамок — по нему узнаём то же лицо после повторного прохода."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    common = (right - left) * (bottom - top)
    area = ((first[2] - first[0]) * (first[3] - first[1])
            + (second[2] - second[0]) * (second[3] - second[1]) - common)
    return common / area if area > 0 else 0.0


def same_moment(first, second, tolerance=0.25):
    """Один и тот же кадр ролика (или обе метки пустые — обычная фотография)."""
    if first is None and second is None:
        return True
    if first is None or second is None:
        return False
    return abs(float(first) - float(second)) <= tolerance


def ranges_close(first_start, first_stop, second_start, second_stop, gap=0.75):
    """Пересекаются ли два промежутка времени (с небольшим запасом)."""
    if None in (first_start, first_stop, second_start, second_stop):
        return False
    return first_start <= second_stop + gap and second_start <= first_stop + gap


def merge_faces(db, key, results, threshold=0.45, track_embedding_threshold=0.5):
    """Обновляем прежние лица файла вместо удаления: к ним привязаны имена.

    У фото и старых покадровых видеозаписей результат совпадает с прежним
    лицом по рамке в тот же момент. У треков момента одного нет — трек это
    промежуток, а рамка по ходу трека меняется, — поэтому им подошло бы
    только сравнение с прежним треком, который пересекается по времени и
    похож по голосу эмбеддинга: это тот же человек в том же куске ролика.
    """
    import numpy as np
    previous = []
    for face_id, raw_box, moment, track_start, track_stop, raw_embedding in db.execute(
            'SELECT id,box,frame_time,track_start,track_stop,embedding FROM faces '
            'WHERE path=?', (key,)):
        try:
            box = [float(value) for value in json.loads(raw_box)]
        except (TypeError, ValueError, json.JSONDecodeError):
            box = None
        embedding = np.frombuffer(raw_embedding, dtype='<f4') if raw_embedding else None
        previous.append((face_id, box, moment, track_start, track_stop, embedding))
    taken = set()
    for item in results:
        path_key, box_json, embedding, thumbnail = item[:4]
        moment = item[4] if len(item) > 4 else None
        track_start = item[5] if len(item) > 5 else None
        track_stop = item[6] if len(item) > 6 else None
        box = [float(value) for value in json.loads(box_json)]
        best, best_score = None, 0.0
        if track_start is not None:
            vector = np.frombuffer(embedding, dtype='<f4')
            for face_id, _, _, old_start, old_stop, old_vector in previous:
                if (face_id in taken or old_vector is None
                        or not ranges_close(track_start, track_stop, old_start, old_stop)):
                    continue
                denom = (np.linalg.norm(vector) * np.linalg.norm(old_vector)) or 1.0
                score = float(np.dot(vector, old_vector) / denom)
                if score >= track_embedding_threshold and score > best_score:
                    best, best_score = face_id, score
        else:
            for face_id, old_box, old_moment, old_start, _, _ in previous:
                # Кадры ролика — разные сцены: рамки сравниваем только внутри секунды.
                # Прежний трек рамкой не сравниваем — она за трек успела уехать.
                if (face_id in taken or old_box is None or old_start is not None
                        or not same_moment(moment, old_moment)):
                    continue
                score = overlap(box, old_box)
                if score >= threshold and score > best_score:
                    best, best_score = face_id, score
        if best is None:
            db.execute(
                'INSERT INTO faces(path,box,embedding,thumbnail,frame_time,'
                'track_start,track_stop) VALUES(?,?,?,?,?,?,?)',
                (path_key, box_json, embedding, thumbnail, moment, track_start, track_stop))
            continue
        taken.add(best)
        db.execute('UPDATE faces SET box=?,embedding=?,thumbnail=?,frame_time=?,'
                  'track_start=?,track_stop=? WHERE id=?',
                   (box_json, embedding, thumbnail, moment, track_start, track_stop, best))
        # Вектор изменился — метку кластера пересчитаем, имя остаётся.
        db.execute('DELETE FROM face_clusters WHERE face_id=?', (best,))
    stale = [(face_id,) for face_id, *_ in previous if face_id not in taken]
    if stale:
        db.executemany('DELETE FROM faces WHERE id=?', stale)
    return len(taken), len(results) - len(taken), len(stale)


def cluster_embeddings(matrix, algorithm='hdbscan', distance=0.35, min_cluster_size=8):
    """Cluster normalized face vectors and return labels plus membership confidence."""
    import numpy as np
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    if algorithm == 'hdbscan':
        from sklearn.cluster import HDBSCAN
        estimator = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean',
        ).fit(matrix)
        return estimator.labels_, estimator.probabilities_
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=distance, min_samples=2, metric='cosine', n_jobs=-1).fit_predict(matrix)
    return labels, np.where(labels == -1, 0.0, 1.0)


def gallery(args):
    import numpy as np
    db = database(args.data)
    # Prototype only: cap size before allocating the potentially quadratic cluster workload.
    rows = db.execute('SELECT faces.id,faces.path,embedding,thumbnail,model FROM faces JOIN photos USING(path) WHERE status="ok" ORDER BY faces.id LIMIT 5001').fetchall()
    if not rows:
        raise ValueError('No indexed faces')
    if len(rows) > 5000:
        raise ValueError('Prototype clustering is limited to 5000 faces; use a separate sample catalog')
    if len({r[4] for r in rows}) != 1:
        raise ValueError('Mixed model versions: re-scan with one model before grouping')
    matrix = np.stack([np.frombuffer(r[2], dtype='<f4') for r in rows])
    labels, probabilities = cluster_embeddings(
        matrix, args.algorithm, args.distance, args.min_cluster_size)
    groups = {}
    for row, label, probability in zip(rows, labels, probabilities):
        groups.setdefault(int(label), []).append((row, float(probability)))
    output = ['<!doctype html><html lang="ru"><meta charset="utf-8">',
              '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\' file:; style-src \'unsafe-inline\'">',
              '<title>Локальная фототека — прототип</title>',
              '<style>body{font:16px system-ui;background:#151820;color:#eee;margin:32px}section{display:flex;flex-wrap:wrap;gap:12px}img{width:120px;height:120px;object-fit:cover;border-radius:8px}figure{margin:0;width:140px}figcaption{overflow-wrap:anywhere;font-size:12px;color:#ccd}.confidence{color:#8fd;font-size:11px}</style>',
              '<h1>Предварительные группы лиц</h1>',
              f'<p>Алгоритм: {html.escape(args.algorithm)}. Группы — гипотезы, а не подтверждённые личности. Галерея работает локально.</p>']
    # Put useful groups first and the usually large noise bucket last.
    ordered = sorted(groups.items(), key=lambda item: (item[0] == -1, -len(item[1])))
    for label, members in ordered:
        members.sort(key=lambda item: item[1], reverse=True)
        name = 'Не сгруппированы' if label == -1 else f'Группа {label + 1}'
        output.append(f'<h2>{name}: {len(members)} лиц / {len({r[0][1] for r in members})} фото</h2><section>')
        for row, probability in members:
            confidence = '' if label == -1 else f'<span class="confidence">уверенность {probability:.0%}</span>'
            output.append(f'<figure title="{html.escape(row[1], quote=True)}"><img loading="lazy" src="{html.escape(row[3], quote=True)}"><figcaption>{html.escape(Path(row[1]).name)}<br>{confidence}</figcaption></figure>')
        output.append('</section>')
    output.append('</html>')
    target = args.data / 'gallery.html'
    target.write_text('\n'.join(output), encoding='utf-8')
    print(target.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    scan_parser = sub.add_parser('scan')
    scan_parser.add_argument('--photos', type=Path, required=True)
    scan_parser.add_argument('--models', type=Path, required=True)
    scan_parser.add_argument('--data', type=Path, default=Path('data'))
    scan_parser.add_argument('--limit', type=int, default=1000)
    scan_parser.add_argument('--exclude-path', action='append', type=Path, default=[],
                             help='Skip this directory tree; repeat for multiple paths')
    scan_parser.add_argument('--exclude-dir-name', action='append', default=[],
                             help='Skip directories with this name (case-insensitive); repeat as needed')
    scan_parser.add_argument('--exclude-file-pattern', action='append', default=[],
                             help='Skip matching file names, for example *__an__*')
    scan_parser.add_argument('--include-path', action='append', type=Path, default=[],
                             help='Process only this exact photo; repeat for a selection')
    scan_parser.add_argument('--use-inventory', action='store_true',
                             help='Взять список файлов из описи каталога, не обходя диск')
    scan_parser.add_argument('--force', action='store_true',
                             help='Искать лица заново, даже если файл не менялся')
    scan_parser.add_argument('--min-side', type=int, default=0,
                             help='Ignore images whose shorter side is below this pixel count')
    scan_parser.add_argument('--progress-file', type=Path,
                             help='Write machine-readable scan progress to this JSON file')
    scan_parser.add_argument('--stop-file', type=Path,
                             help='Stop safely when this file appears')
    gallery_parser = sub.add_parser('gallery')
    gallery_parser.add_argument('--data', type=Path, default=Path('data'))
    gallery_parser.add_argument('--algorithm', choices=('hdbscan', 'dbscan'), default='hdbscan')
    gallery_parser.add_argument('--distance', type=float, default=0.35)
    gallery_parser.add_argument('--min-cluster-size', type=int, default=8)
    args = parser.parse_args()
    if args.command == 'scan' and args.limit < 1:
        parser.error('--limit must be positive')
    if args.command == 'scan' and args.min_side < 0:
        parser.error('--min-side cannot be negative')
    if args.command == 'gallery' and not 0 < args.distance < 1:
        parser.error('--distance must be between 0 and 1')
    if args.command == 'gallery' and args.min_cluster_size < 2:
        parser.error('--min-cluster-size must be at least 2')
    if args.command == 'doctor':
        doctor()
    elif args.command == 'scan':
        scan(args)
    else:
        gallery(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped. Re-run scan to continue.', file=sys.stderr)
        sys.exit(130)
