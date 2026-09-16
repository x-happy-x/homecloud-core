"""Поиск дубликатов: точные копии по хешу файла и похожие кадры по dHash.

Полный хеш всех 78 тысяч файлов — это прочитать весь диск, поэтому точные копии
ищем только среди файлов одинакового размера: другие совпасть не могут. Похожие
(пережатые, изменённого размера) ищем по перцептивному хешу — для него картинку
всё-таки приходится открывать, поэтому это отдельный, необязательный проход.
"""
import hashlib
import sqlite3
import time
from pathlib import Path

SCHEMA = '''
CREATE TABLE IF NOT EXISTS photo_hashes (
  path TEXT PRIMARY KEY,
  size INTEGER NOT NULL,
  modified INTEGER NOT NULL,
  sha1 TEXT,
  dhash TEXT,
  width INTEGER,
  height INTEGER,
  computed REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS photo_hashes_sha1 ON photo_hashes(sha1);
CREATE INDEX IF NOT EXISTS photo_hashes_dhash ON photo_hashes(dhash);
'''

CHUNK = 1024 * 1024


def ensure_schema(db):
    db.executescript(SCHEMA)
    db.commit()


def connect(catalog):
    db = sqlite3.connect(Path(catalog) / 'catalog.sqlite', timeout=60)
    ensure_schema(db)
    return db


def file_sha1(path):
    digest = hashlib.sha1()
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def picture_hash(path):
    """dHash 8×8: сравнивает соседние пиксели, переживает пережатие и масштаб."""
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        source.draft('L', (64, 64))
        image = ImageOps.exif_transpose(source).convert('L')
        width, height = image.size
        small = image.resize((9, 8), Image.BILINEAR)
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(small.getpixel((x, y)) > small.getpixel((x + 1, y)))
    return f'{bits:016x}', width, height


def candidates(db, similar):
    """Что осталось посчитать: файлы без свежего хеша."""
    if similar:
        # Для похожих нужен каждый снимок: заранее отсеять нельзя.
        return db.execute('''
            SELECT photos.path,photos.size,photos.modified FROM photos
            LEFT JOIN photo_hashes USING(path)
            WHERE photos.status='ok' AND COALESCE(photos.kind,'photo')='photo'
              AND COALESCE(photos.blocked,0)=0
              AND (photo_hashes.path IS NULL OR photo_hashes.dhash IS NULL
                   OR photo_hashes.size<>photos.size
                   OR photo_hashes.modified<>photos.modified)
            ORDER BY photos.path''').fetchall()
    # Точные копии бывают только у файлов одинакового размера.
    return db.execute('''
        SELECT photos.path,photos.size,photos.modified FROM photos
        LEFT JOIN photo_hashes USING(path)
        WHERE photos.status='ok' AND COALESCE(photos.blocked,0)=0
          AND photos.size IN (SELECT size FROM photos WHERE status='ok'
                              GROUP BY size HAVING COUNT(*)>1)
          AND (photo_hashes.path IS NULL OR photo_hashes.sha1 IS NULL
               OR photo_hashes.size<>photos.size
               OR photo_hashes.modified<>photos.modified)
        ORDER BY photos.size,photos.path''').fetchall()


def compute(db, rows, similar, progress=None, stop=None, disk=None):
    """Считает хеши порциями, чтобы прогресс был виден, а база не блокировалась."""
    done = errors = 0
    batch = []
    for path, size, modified in rows:
        if stop and stop():
            break
        target = Path((disk or {}).get(path, path))
        try:
            if not target.is_file():
                raise OSError('файл не найден')
            sha1 = file_sha1(target)
            shape = (None, None, None)
            if similar:
                shape = picture_hash(target)
            batch.append((path, size, modified, sha1, shape[0], shape[1], shape[2], time.time()))
        except Exception:
            errors += 1
        done += 1
        if len(batch) >= 100:
            _flush(db, batch)
            batch = []
        if progress and done % 25 == 0:
            progress(done, errors, path)
    _flush(db, batch)
    if progress:
        progress(done, errors, '')
    return done, errors


def _flush(db, batch):
    if not batch:
        return
    with db:
        db.executemany(
            'INSERT INTO photo_hashes(path,size,modified,sha1,dhash,width,height,computed) '
            'VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET '
            'size=excluded.size,modified=excluded.modified,sha1=excluded.sha1,'
            'dhash=COALESCE(excluded.dhash,photo_hashes.dhash),'
            'width=COALESCE(excluded.width,photo_hashes.width),'
            'height=COALESCE(excluded.height,photo_hashes.height),'
            'computed=excluded.computed', batch)


def _distance(first, second):
    return bin(int(first, 16) ^ int(second, 16)).count('1')


def groups(db, similar=False, distance=4, limit=200, offset=0, skip=()):
    """Группы одинаковых снимков: сначала точные копии, потом похожие."""
    ensure_schema(db)
    skip = set(skip or ())
    found = []
    seen = set()
    # Один запрос вместо запроса на группу: групп бывают десятки тысяч.
    exact = {}
    for sha1, path in db.execute(
            'SELECT sha1,path FROM photo_hashes WHERE sha1 IN '
            '(SELECT sha1 FROM photo_hashes WHERE sha1 IS NOT NULL '
            ' GROUP BY sha1 HAVING COUNT(*)>1) ORDER BY sha1,path'):
        if path not in skip:
            exact.setdefault(sha1, []).append(path)
    for sha1, paths in sorted(exact.items(), key=lambda item: -len(item[1])):
        if len(paths) < 2:
            continue
        seen.update(paths)
        found.append({'key': f'sha1:{sha1}', 'kind': 'exact', 'paths': paths})
    if similar:
        buckets = {}
        for path, dhash in db.execute(
                'SELECT path,dhash FROM photo_hashes WHERE dhash IS NOT NULL ORDER BY path'):
            if path in skip or path in seen:
                continue
            buckets.setdefault(dhash, []).append(path)
        # Сравнивать каждый хеш с каждым нельзя: на 78 тысячах это миллиарды пар.
        # Режем хеш на четыре части: при расхождении до трёх бит хотя бы одна
        # часть обязана совпасть, поэтому кандидатов ищем по ним.
        distance = min(max(int(distance), 0), 3)
        bands = {}
        for key in buckets:
            for number in range(4):
                bands.setdefault((number, key[number * 4:number * 4 + 4]), []).append(key)
        used = set()
        for key in buckets:
            if key in used:
                continue
            near, queue = {key}, [key]
            while queue:
                current = queue.pop()
                for number in range(4):
                    for other in bands.get((number, current[number * 4:number * 4 + 4]), ()):
                        if other in near or other in used:
                            continue
                        if _distance(current, other) <= distance:
                            near.add(other)
                            queue.append(other)
            used.update(near)
            paths = sorted(path for item in near for path in buckets[item])
            if len(paths) < 2:
                continue
            found.append({'key': f'dhash:{key}', 'kind': 'similar', 'paths': paths})
    total = len(found)
    return found[offset:offset + limit], total


def stats(db):
    ensure_schema(db)
    hashed = db.execute('SELECT COUNT(*) FROM photo_hashes WHERE sha1 IS NOT NULL').fetchone()[0]
    pictured = db.execute('SELECT COUNT(*) FROM photo_hashes WHERE dhash IS NOT NULL').fetchone()[0]
    return {'hashed': hashed, 'pictured': pictured}


def forget(db, paths):
    paths = [str(path) for path in paths if path]
    if not paths:
        return
    marks = ','.join('?' * len(paths))
    with db:
        db.execute(f'DELETE FROM photo_hashes WHERE path IN ({marks})', paths)


# Файл меньше этого — скорее иконка или картинка интерфейса, чем снимок.
SMALL_FILE = 100 * 1024


def index(found, shapes):
    """Группы с решением «что оставить», весом лишнего и общей сводкой.

    shapes — {путь: (size, width, height)}. Оставляем самый крупный кадр, при
    равенстве — самый тяжёлый файл и самый короткий путь; он идёт в группе
    первым, чтобы попасть в видимые карточки.
    """
    groups = []
    folders = {}
    for group in found:
        paths = group['paths']

        def rank(path):
            size, width, height = shapes.get(path) or (0, 0, 0)
            return ((width or 0) * (height or 0), size or 0, -len(path))

        keep = max(paths, key=rank)
        sizes = {path: (shapes.get(path) or (0, 0, 0))[0] or 0 for path in paths}
        extra = sum(sizes.values()) - sizes[keep]
        for path in paths:
            if path != keep:
                folder = str(Path(path).parent)
                folders[folder] = folders.get(folder, 0) + 1
        groups.append({
            'key': group['key'], 'kind': group['kind'], 'keep': keep,
            'paths': [keep, *(path for path in paths if path != keep)],
            'count': len(paths), 'extra': extra, 'file_size': max(sizes.values(), default=0),
        })
    summary = {
        'groups': len(groups),
        'exact': sum(1 for group in groups if group['kind'] == 'exact'),
        'similar': sum(1 for group in groups if group['kind'] == 'similar'),
        'files': sum(group['count'] for group in groups),
        'extra_files': sum(group['count'] - 1 for group in groups),
        'extra_bytes': sum(group['extra'] for group in groups),
        'small_groups': sum(1 for group in groups if group['file_size'] < SMALL_FILE),
        'top_folders': [{'folder': folder, 'copies': copies} for folder, copies in
                        sorted(folders.items(), key=lambda item: (-item[1], item[0]))[:6]],
    }
    return groups, summary


def parent(path):
    """Папка файла ровно в том виде, в каком её считает сводка."""
    return str(Path(path).parent)


def in_folder(group, folder):
    """Лишние копии группы, лежащие прямо в этой папке, без вложенных."""
    return [path for path in group['paths'][1:] if parent(path) == folder]


def select(groups, kind='all', sort='size', hide_small=False, folder=''):
    """Фильтр и порядок групп для выдачи.

    folder оставляет группы, у которых лишняя копия лежит прямо в этой папке:
    те самые, что сводка посчитала в «где больше всего лишних копий».
    """
    chosen = [group for group in groups
              if (kind not in ('exact', 'similar') or group['kind'] == kind)
              and not (hide_small and group['file_size'] < SMALL_FILE)
              and (not folder or in_folder(group, folder))]
    if sort == 'count':
        chosen.sort(key=lambda group: (-group['count'], -group['extra'], group['key']))
    else:
        chosen.sort(key=lambda group: (-group['extra'], -group['count'], group['key']))
    return chosen
