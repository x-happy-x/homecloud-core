"""Превью сетки и сведения о файлах — этап ядра после описи.

Хаб показывает галерею из своих мелких превью, поэтому сетка открывается
без ядра и без источника. Превью делаются при каждом обходе источника для
нового и изменившегося (размер или время файла другие) и уходят на хаб
вместе со сведениями о файле: размеры оригинала и EXIF для просмотрщика.
В самом источнике ничего не пишется.

Размер: короткая сторона 400 px, длинная не больше 800 — хватает плиткам
сетки вплоть до крупного масштаба на экранах с двойной плотностью.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
import time

import catalogdb
import hublink
import media_metadata
import pathkeys
import sources
import video as video_media

SHORT = 400
LONG = 800
BATCH = 40


def write_progress(path, **values):
    if path is None:
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps({**values, 'updated_at': time.time(), 'pid': os.getpid()},
                                    ensure_ascii=False), encoding='utf-8')
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))


def pending(db, roots, paths, force=False):
    """Снимки задания, у которых превью нет или оно от прежней версии файла."""
    db.executescript('''CREATE TABLE IF NOT EXISTS photo_thumbs (
        path TEXT PRIMARY KEY, size INTEGER NOT NULL, modified INTEGER NOT NULL,
        width INTEGER, height INTEGER, thumb_width INTEGER, thumb_height INTEGER,
        bytes INTEGER NOT NULL DEFAULT 0, metadata_json TEXT, created_at REAL NOT NULL);''')
    where, values = pathkeys.scope_sql(roots, paths, 'photos.path')
    fresh = '' if force else (' AND (photo_thumbs.path IS NULL OR photo_thumbs.size!=photos.size'
                              ' OR photo_thumbs.modified!=photos.modified)')
    return db.execute(
        'SELECT photos.path,photos.size,photos.modified FROM photos '
        'LEFT JOIN photo_thumbs ON photo_thumbs.path=photos.path '
        # В галерее только годные и не исключённые правилами путей — им и превью.
        "WHERE photos.status='ok' AND COALESCE(photos.blocked,0)=0" + fresh + where
        + ' ORDER BY photos.path',
        values).fetchall()


def fit(width, height):
    scale = min(1.0, SHORT / max(1, min(width, height)), LONG / max(1, max(width, height)))
    return max(1, round(width * scale)), max(1, round(height * scale))


def render(key, size, modified):
    """Превью и сведения одного файла. Ошибка чтения — не повод ронять этап."""
    from PIL import Image, ImageOps
    local = sources.local(key)
    if video_media.is_video(key):
        info = video_media.probe(local)
        image = video_media.to_image(video_media.poster(local))
        width, height = info['width'] or image.width, info['height'] or image.height
    else:
        with Image.open(local) as source:
            # Размер оригинала — до draft: тот сразу уменьшает картинку при чтении.
            full = source.size
            source.draft('RGB', (SHORT * 2, SHORT * 2))
            orientation = source.getexif().get(274, 1)
            image = ImageOps.exif_transpose(source).convert('RGB')
        width, height = full
        if orientation in {5, 6, 7, 8}:
            width, height = height, width
    image = image.convert('RGB')
    image.thumbnail(fit(image.width, image.height), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, 'JPEG', quality=80, optimize=True, progressive=True)
    try:
        metadata = media_metadata.read(local)
    except Exception as exc:
        metadata = {'error': str(exc)}
    return {'path': key, 'size': size, 'modified': modified, 'width': width, 'height': height,
            'thumb_width': image.width, 'thumb_height': image.height,
            'data': base64.b64encode(stream.getvalue()).decode('ascii'), 'metadata': metadata}


def run(args):
    link = hublink.link_for(args.catalog)
    progress = args.progress_file
    if link is None:
        # У бэкенда одного компьютера превью делаются по запросу, как раньше.
        write_progress(progress, status='completed', total=0, completed=0)
        print('Каталог локальный — превью хранить негде, этап пропущен.')
        return
    db = catalogdb.connect(args.catalog, timeout=60)
    try:
        rows = pending(db, args.root, args.path, args.force)
    finally:
        db.close()
    state = {'status': 'running', 'total': len(rows), 'completed': 0, 'errors': 0,
             'current': '', 'videos_total': sum(video_media.is_video(row[0]) for row in rows),
             'videos_done': 0}
    write_progress(progress, **state)
    batch = []

    def flush():
        if batch:
            link.json('POST', '/thumbs', {'items': batch}, timeout=300)
            batch.clear()

    def task(row):
        try:
            return row, render(*row), None
        except Exception as exc:
            return row, None, exc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(rows), 200):
            if args.stop_file and args.stop_file.exists():
                flush()
                state['status'] = 'stopped'
                write_progress(progress, **state)
                return
            for row, item, error in pool.map(task, rows[offset:offset + 200]):
                state['completed'] += 1
                state['current'] = row[0]
                if video_media.is_video(row[0]):
                    state['videos_done'] += 1
                if error is not None:
                    state['errors'] += 1
                    print(f'ERROR: {row[0]}: {error}', file=sys.stderr, flush=True)
                    continue
                batch.append(item)
                if len(batch) >= BATCH:
                    flush()
                write_progress(progress, **state)
    flush()
    state['status'] = 'completed'
    write_progress(progress, **state)
    print(f"Превью: {state['completed'] - state['errors']} готово, ошибок {state['errors']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--root', action='append', type=str, default=[])
    parser.add_argument('--path', action='append', type=str, default=[])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--progress-file', type=Path)
    parser.add_argument('--stop-file', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
