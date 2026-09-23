"""Альбомы и дерево папок каталога.

Альбом — ручная подборка: «2010 год», внутри «Семья», «Отпуск». Вложенность
произвольная, один снимок может лежать в любом числе альбомов. Папки берём из
самого каталога (`photos.dir`) и показываем той же иерархией, что на диске.
"""
import time

import pathkeys

SCHEMA = '''
CREATE TABLE IF NOT EXISTS albums (
  id INTEGER PRIMARY KEY,
  parent_id INTEGER NOT NULL DEFAULT 0,
  title TEXT NOT NULL,
  created REAL NOT NULL,
  UNIQUE(parent_id, title)
);
CREATE TABLE IF NOT EXISTS album_photos (
  album_id INTEGER NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  added REAL NOT NULL,
  PRIMARY KEY(album_id, path)
);
CREATE INDEX IF NOT EXISTS album_photos_path ON album_photos(path);
CREATE INDEX IF NOT EXISTS albums_parent ON albums(parent_id);
'''

MAX_DEPTH = 8


def ensure_schema(db):
    db.executescript(SCHEMA)
    db.commit()


def _rows(db):
    return [{'id': row[0], 'parent_id': row[1], 'title': row[2], 'created': row[3]}
            for row in db.execute(
                'SELECT id,parent_id,title,created FROM albums ORDER BY title')]


def _children_map(rows):
    children = {0: []}
    for row in rows:
        children.setdefault(row['id'], [])
        children.setdefault(row['parent_id'], []).append(row['id'])
    return children


def descendants(db, album_id):
    """Сам альбом и всё, что внутри него: для фильтра «с вложенными»."""
    children = _children_map(_rows(db))
    found, queue = [], [int(album_id)]
    while queue:
        current = queue.pop()
        if current in found:
            continue
        found.append(current)
        queue.extend(children.get(current, ()))
    return found


def tree(db):
    """Плоский список альбомов сверху вниз: путь, счётчики и обложка."""
    rows = _rows(db)
    by_id = {row['id']: row for row in rows}
    children = _children_map(rows)
    direct = dict(db.execute('SELECT album_id,COUNT(*) FROM album_photos GROUP BY album_id'))
    covers = {}
    for album_id, path in db.execute(
            'SELECT album_id,path FROM album_photos ORDER BY added DESC'):
        covers.setdefault(album_id, path)

    items = {}

    def walk(album_id, depth, trail):
        row = by_id[album_id]
        names = [*trail, row['title']]
        kids = sorted(children.get(album_id, ()), key=lambda item: by_id[item]['title'])
        total = direct.get(album_id, 0)
        for kid in kids:
            total += walk(kid, depth + 1, names)
            covers.setdefault(album_id, covers.get(kid))
        items[album_id] = {
            'id': album_id, 'parent_id': row['parent_id'], 'title': row['title'],
            'depth': depth, 'trail': ' / '.join(names), 'children': kids,
            'photos': direct.get(album_id, 0), 'total': total,
            'cover': covers.get(album_id) or '',
        }
        return total

    top = sorted(children.get(0, ()), key=lambda item: by_id[item]['title'])
    for album_id in top:
        walk(album_id, 0, [])

    # Обход считает детей раньше родителей — выдачу собираем «сверху вниз».
    result = []

    def emit(album_id):
        result.append(items[album_id])
        for kid in items[album_id]['children']:
            emit(kid)

    for album_id in top:
        emit(album_id)
    return result


def _depth(db, parent_id):
    depth, current = 0, int(parent_id)
    while current:
        row = db.execute('SELECT parent_id FROM albums WHERE id=?', (current,)).fetchone()
        if row is None:
            raise KeyError('Родительский альбом не найден')
        current, depth = row[0], depth + 1
        if depth > MAX_DEPTH:
            break
    return depth


def create(db, title, parent_id=0):
    title = ' '.join(str(title or '').split())[:120]
    if not title:
        raise ValueError('Название альбома не может быть пустым')
    parent_id = int(parent_id or 0)
    if parent_id and _depth(db, parent_id) >= MAX_DEPTH:
        raise ValueError('Слишком глубокая вложенность альбомов')
    exists = db.execute('SELECT id FROM albums WHERE parent_id=? AND title=?',
                        (parent_id, title)).fetchone()
    if exists:
        return exists[0]
    with db:
        cursor = db.execute('INSERT INTO albums(parent_id,title,created) VALUES(?,?,?)',
                            (parent_id, title, time.time()))
    return cursor.lastrowid


def rename(db, album_id, title):
    title = ' '.join(str(title or '').split())[:120]
    if not title:
        raise ValueError('Название альбома не может быть пустым')
    with db:
        db.execute('UPDATE albums SET title=? WHERE id=?', (title, int(album_id)))


def move(db, album_id, parent_id=0):
    album_id, parent_id = int(album_id), int(parent_id or 0)
    if parent_id in descendants(db, album_id):
        raise ValueError('Альбом нельзя вложить в самого себя')
    if parent_id and _depth(db, parent_id) >= MAX_DEPTH:
        raise ValueError('Слишком глубокая вложенность альбомов')
    with db:
        db.execute('UPDATE albums SET parent_id=? WHERE id=?', (parent_id, album_id))


def remove(db, album_id):
    """Удаляем альбом вместе с вложенными; сами файлы не трогаем."""
    ids = descendants(db, album_id)
    marks = ','.join('?' * len(ids))
    with db:
        db.execute(f'DELETE FROM album_photos WHERE album_id IN ({marks})', ids)
        db.execute(f'DELETE FROM albums WHERE id IN ({marks})', ids)
    return len(ids)


def set_photos(db, album_id, add=(), drop=()):
    album_id = int(album_id)
    if db.execute('SELECT 1 FROM albums WHERE id=?', (album_id,)).fetchone() is None:
        raise KeyError('Альбом не найден')
    add = list(dict.fromkeys(str(path) for path in add if path))
    drop = list(dict.fromkeys(str(path) for path in drop if path))
    stamp = time.time()
    with db:
        if add:
            db.executemany(
                'INSERT INTO album_photos(album_id,path,added) VALUES(?,?,?) '
                'ON CONFLICT(album_id,path) DO NOTHING',
                [(album_id, path, stamp) for path in add])
        if drop:
            marks = ','.join('?' * len(drop))
            db.execute(f'DELETE FROM album_photos WHERE album_id=? AND path IN ({marks})',
                       [album_id, *drop])
    return db.execute('SELECT COUNT(*) FROM album_photos WHERE album_id=?',
                      (album_id,)).fetchone()[0]


def photo_albums(db, paths):
    """Для каждого пути — альбомы, в которых он лежит."""
    result = {path: [] for path in paths}
    if not paths:
        return result
    known = {item['id']: item for item in tree(db)}
    for offset in range(0, len(paths), 400):
        batch = paths[offset:offset + 400]
        marks = ','.join('?' * len(batch))
        for path, album_id in db.execute(
                f'SELECT path,album_id FROM album_photos WHERE path IN ({marks})', batch):
            item = known.get(album_id)
            if item and path in result:
                result[path].append({'id': album_id, 'title': item['title'],
                                     'trail': item['trail']})
    for values in result.values():
        values.sort(key=lambda item: item['trail'])
    return result


# ---------- папки ----------

def _escape(value):
    return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


def folder_clause(path, deep=True):
    """Условие SQL «снимок лежит в этой папке» (по желанию — и во вложенных)."""
    path = str(path)
    trimmed = path.rstrip('\\/') or path
    if not deep:
        return ' AND photos.dir=?', [trimmed]
    separator = '/' if ('/' in path and '\\' not in path) else '\\'
    prefix = path if path.endswith(('\\', '/')) else trimmed + separator
    return (" AND (photos.dir=? OR photos.dir LIKE ? ESCAPE '\\')",
            [trimmed, _escape(prefix) + '%'])


class Folders:
    """Дерево папок каталога. Пересобирается, когда каталог менялся.

    root_label — подпись корня: у хаба это «PC-X · D:» или «Netcraze · /HDD».
    """

    def __init__(self, db, root_label=None):
        self.db = db
        self.root_label = root_label
        self.stamp = None
        self.nodes = {}
        self.roots = []

    def refresh(self, force=False):
        stamp = self.db.execute(
            "SELECT COUNT(*),MAX(modified) FROM photos WHERE status='ok'").fetchone()
        if stamp == self.stamp and not force:
            return
        nodes, roots = {}, []
        for folder, count in self.db.execute(
                "SELECT dir,COUNT(*) FROM photos WHERE status='ok' AND dir IS NOT NULL "
                'GROUP BY dir'):
            chain = pathkeys.chain(folder)
            folder = chain[-1]
            for index, key in enumerate(chain):
                node = nodes.get(key)
                if node is None:
                    parent = chain[index - 1] if index else ''
                    name = pathkeys.name(key) or key
                    if not index and self.root_label:
                        name = self.root_label(key)
                    node = nodes[key] = {
                        'path': key, 'parent': parent, 'direct': 0, 'total': 0,
                        'name': name, 'children': []}
                    if parent:
                        nodes[parent]['children'].append(key)
                    else:
                        roots.append(key)
                node['total'] += count
            nodes[folder]['direct'] += count
        for node in nodes.values():
            node['children'].sort(key=lambda key: nodes[key]['name'].casefold())
        self.nodes, self.roots = nodes, sorted(roots)
        self.stamp = stamp

    def children(self, path=''):
        """Дети одной папки; пустой путь — корни дисков."""
        self.refresh()
        keys = self.roots if not path else self.nodes.get(path, {}).get('children', [])
        return [{'path': key, 'name': self.nodes[key]['name'],
                 'photos': self.nodes[key]['total'],
                 'direct': self.nodes[key]['direct'],
                 'folders': len(self.nodes[key]['children'])} for key in keys]

    def trail(self, path):
        """Цепочка от корня до папки — для «хлебных крошек»."""
        self.refresh()
        chain, current = [], path
        while current and current in self.nodes:
            node = self.nodes[current]
            chain.append({'path': node['path'], 'name': node['name']})
            current = node['parent']
        chain.reverse()
        return chain
