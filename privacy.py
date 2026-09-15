"""Скрытый альбом: файл уезжает в личную папку и пропадает у остальных.

Ключ каталога — исходный путь снимка, он не меняется: к нему привязаны лица,
альбомы и анализ. Меняется только место файла на диске, и оно записано в
`hidden_photos.stored`. Владельцем считается логин из картотеки; администратор
видит и чужое.
"""
import hashlib
import shutil
import time
from pathlib import Path

SCHEMA = '''
CREATE TABLE IF NOT EXISTS hidden_photos (
  path TEXT PRIMARY KEY,
  owner TEXT NOT NULL,
  stored TEXT NOT NULL,
  hidden_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS hidden_photos_owner ON hidden_photos(owner);
'''


def ensure_schema(db):
    db.executescript(SCHEMA)
    db.commit()


def root(catalog, values=None):
    """Куда складывать скрытое: по умолчанию в сам каталог, рядом с превью."""
    custom = str((values or {}).get('hidden_root') or '').strip()
    return Path(custom) if custom else Path(catalog) / 'hidden'


def _slot(folder, owner, path):
    """Имя файла в хранилище: по исходному пути, чтобы не зависеть от кириллицы."""
    key = hashlib.sha1(f'{owner}|{path}'.encode('utf-8')).hexdigest()
    return folder / key[:2] / f'{key}{Path(path).suffix.lower()}'


def owners(db):
    ensure_schema(db)
    return dict(db.execute('SELECT owner,COUNT(*) FROM hidden_photos GROUP BY owner'))


def mine(db, viewer):
    ensure_schema(db)
    return {row[0] for row in db.execute(
        'SELECT path FROM hidden_photos WHERE owner=?', (viewer,))}


def foreign(db, viewer):
    """Пути, спрятанные другими: их не видно ни в галерее, ни в лицах."""
    ensure_schema(db)
    return {row[0] for row in db.execute(
        'SELECT path FROM hidden_photos WHERE owner<>?', (viewer,))}


def stored_paths(db):
    ensure_schema(db)
    return dict(db.execute('SELECT path,stored FROM hidden_photos'))


def where(viewer, admin, hidden):
    """Условие видимости для выборки снимков; None — без ограничений."""
    if hidden is None:
        return '', []
    if hidden:
        # Свой скрытый альбом; администратору показываем и чужие.
        if admin:
            return ' AND photos.path IN (SELECT path FROM hidden_photos)', []
        return (' AND photos.path IN (SELECT path FROM hidden_photos WHERE owner=?)',
                [viewer or ''])
    # В обычной галерее скрытого нет ни у кого — даже у владельца: для него это
    # отдельный альбом, который он открывает сознательно.
    return ' AND photos.path NOT IN (SELECT path FROM hidden_photos)', []


def hide(db, catalog, viewer, paths, values=None):
    """Переносим файлы в личную папку владельца и запоминаем, где они теперь."""
    ensure_schema(db)
    if not viewer:
        raise ValueError('Скрывать снимки может только вошедший пользователь')
    folder = root(catalog, values)
    moved, errors = [], []
    for raw in dict.fromkeys(str(path) for path in paths if path):
        row = db.execute("SELECT path FROM photos WHERE path=? AND status='ok'",
                         (raw,)).fetchone()
        if row is None:
            errors.append({'path': raw, 'error': 'Снимок не найден в каталоге'})
            continue
        if db.execute('SELECT 1 FROM hidden_photos WHERE path=?', (raw,)).fetchone():
            continue
        source = Path(raw)
        if not source.is_file():
            errors.append({'path': raw, 'error': 'Файл не найден на диске'})
            continue
        target = _slot(folder, viewer, raw)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        except OSError as exc:
            errors.append({'path': raw, 'error': str(exc)})
            continue
        with db:
            db.execute('INSERT INTO hidden_photos(path,owner,stored,hidden_at) '
                       'VALUES(?,?,?,?)', (raw, viewer, str(target), time.time()))
        moved.append(raw)
    return {'hidden': len(moved), 'paths': moved, 'errors': errors}


def reveal(db, viewer, admin, paths):
    """Возвращаем файл на исходное место; чужое может вернуть только админ."""
    ensure_schema(db)
    back, errors = [], []
    for raw in dict.fromkeys(str(path) for path in paths if path):
        row = db.execute('SELECT owner,stored FROM hidden_photos WHERE path=?',
                         (raw,)).fetchone()
        if row is None:
            continue
        owner, stored = row
        if owner != viewer and not admin:
            errors.append({'path': raw, 'error': 'Это чужой скрытый снимок'})
            continue
        source, target = Path(stored), Path(raw)
        try:
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise OSError('На исходном месте уже есть файл')
                shutil.move(str(source), str(target))
        except OSError as exc:
            errors.append({'path': raw, 'error': str(exc)})
            continue
        with db:
            db.execute('DELETE FROM hidden_photos WHERE path=?', (raw,))
        back.append(raw)
    return {'revealed': len(back), 'paths': back, 'errors': errors}


def forget(db, paths):
    """Снимок удалили — запись о тайнике больше не нужна."""
    paths = [str(path) for path in paths if path]
    if not paths:
        return
    marks = ','.join('?' * len(paths))
    with db:
        db.execute(f'DELETE FROM hidden_photos WHERE path IN ({marks})', paths)
