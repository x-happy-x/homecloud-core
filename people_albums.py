"""Альбомы для людей и групп лиц — та же вложенная папка, что у фото
(`albums.py`), только участники — не пути к снимкам, а ключи групп из
раздела «Люди» (`person:id`/`auto:label`/`noise`/`excluded`): под одним
альбомом можно держать и названных людей, и ещё не разобранные автогруппы.

Скрытие альбома — решение владельца картотеки, а не про приватность
конкретного снимка (для этого уже есть `hidden_photos`): скрытый альбом
(и всё вложенное в него) пропадает из «Люди» у всех, кроме админа. Право
скрывать/показывать проверяется на уровне маршрута в `web_server.py`, этот
модуль ему доверяет.
"""
import time

SCHEMA = '''
CREATE TABLE IF NOT EXISTS people_albums (
  id INTEGER PRIMARY KEY,
  parent_id INTEGER NOT NULL DEFAULT 0,
  title TEXT NOT NULL,
  created REAL NOT NULL,
  hidden INTEGER NOT NULL DEFAULT 0,
  UNIQUE(parent_id, title)
);
CREATE TABLE IF NOT EXISTS people_album_members (
  album_id INTEGER NOT NULL REFERENCES people_albums(id) ON DELETE CASCADE,
  group_key TEXT NOT NULL,
  added REAL NOT NULL,
  PRIMARY KEY(album_id, group_key)
);
CREATE INDEX IF NOT EXISTS people_album_members_key ON people_album_members(group_key);
CREATE INDEX IF NOT EXISTS people_albums_parent ON people_albums(parent_id);
'''

MAX_DEPTH = 8


def ensure_schema(db):
    db.executescript(SCHEMA)
    db.commit()


def _rows(db):
    return [{'id': row[0], 'parent_id': row[1], 'title': row[2], 'created': row[3],
             'hidden': bool(row[4])}
            for row in db.execute(
                'SELECT id,parent_id,title,created,hidden FROM people_albums ORDER BY title')]


def _children_map(rows):
    children = {0: []}
    for row in rows:
        children.setdefault(row['id'], [])
        children.setdefault(row['parent_id'], []).append(row['id'])
    return children


def descendants(db, album_id):
    """Сам альбом и всё, что внутри него: для удаления и проверки вложенности."""
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
    """Плоский список альбомов сверху вниз: путь, участники, видимость."""
    rows = _rows(db)
    by_id = {row['id']: row for row in rows}
    children = _children_map(rows)
    direct = dict(db.execute(
        'SELECT album_id,COUNT(*) FROM people_album_members GROUP BY album_id'))
    own_members = {}
    for album_id, group_key in db.execute('SELECT album_id,group_key FROM people_album_members'):
        own_members.setdefault(album_id, []).append(group_key)

    items = {}

    def walk(album_id, depth, trail, ancestor_hidden):
        row = by_id[album_id]
        names = [*trail, row['title']]
        # Скрытый родитель прячет и всё вложенное — своя отметка не нужна на
        # каждом уровне, достаточно унаследовать решение сверху.
        effectively_hidden = ancestor_hidden or row['hidden']
        kids = sorted(children.get(album_id, ()), key=lambda item: by_id[item]['title'])
        all_members = list(own_members.get(album_id, ()))
        total = len(all_members)
        for kid in kids:
            kid_total, kid_members = walk(kid, depth + 1, names, effectively_hidden)
            total += kid_total
            all_members.extend(kid_members)
        items[album_id] = {
            'id': album_id, 'parent_id': row['parent_id'], 'title': row['title'],
            'depth': depth, 'trail': ' / '.join(names), 'children': kids,
            'groups': direct.get(album_id, 0), 'total': total,
            'hidden': row['hidden'], 'effectively_hidden': effectively_hidden,
            'member_keys': all_members,
        }
        return total, all_members

    top = sorted(children.get(0, ()), key=lambda item: by_id[item]['title'])
    for album_id in top:
        walk(album_id, 0, [], False)

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
        row = db.execute('SELECT parent_id FROM people_albums WHERE id=?', (current,)).fetchone()
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
    exists = db.execute('SELECT id FROM people_albums WHERE parent_id=? AND title=?',
                        (parent_id, title)).fetchone()
    if exists:
        return exists[0]
    with db:
        cursor = db.execute(
            'INSERT INTO people_albums(parent_id,title,created,hidden) VALUES(?,?,?,0)',
            (parent_id, title, time.time()))
    return cursor.lastrowid


def rename(db, album_id, title):
    title = ' '.join(str(title or '').split())[:120]
    if not title:
        raise ValueError('Название альбома не может быть пустым')
    with db:
        db.execute('UPDATE people_albums SET title=? WHERE id=?', (title, int(album_id)))


def move(db, album_id, parent_id=0):
    album_id, parent_id = int(album_id), int(parent_id or 0)
    if parent_id in descendants(db, album_id):
        raise ValueError('Альбом нельзя вложить в самого себя')
    if parent_id and _depth(db, parent_id) >= MAX_DEPTH:
        raise ValueError('Слишком глубокая вложенность альбомов')
    with db:
        db.execute('UPDATE people_albums SET parent_id=? WHERE id=?', (parent_id, album_id))


def remove(db, album_id):
    """Удаляем альбом вместе с вложенными; сами группы и людей не трогаем."""
    ids = descendants(db, album_id)
    marks = ','.join('?' * len(ids))
    with db:
        db.execute(f'DELETE FROM people_album_members WHERE album_id IN ({marks})', ids)
        db.execute(f'DELETE FROM people_albums WHERE id IN ({marks})', ids)
    return len(ids)


def set_hidden(db, album_id, hidden):
    """Показать/скрыть альбом — кто имеет на это право, решает маршрут."""
    with db:
        db.execute('UPDATE people_albums SET hidden=? WHERE id=?',
                   (1 if hidden else 0, int(album_id)))


def set_members(db, album_id, add=(), drop=()):
    album_id = int(album_id)
    if db.execute('SELECT 1 FROM people_albums WHERE id=?', (album_id,)).fetchone() is None:
        raise KeyError('Альбом не найден')
    add = list(dict.fromkeys(str(key) for key in add if key))
    drop = list(dict.fromkeys(str(key) for key in drop if key))
    stamp = time.time()
    with db:
        if add:
            db.executemany(
                'INSERT INTO people_album_members(album_id,group_key,added) VALUES(?,?,?) '
                'ON CONFLICT(album_id,group_key) DO NOTHING',
                [(album_id, key, stamp) for key in add])
        if drop:
            marks = ','.join('?' * len(drop))
            db.execute(
                f'DELETE FROM people_album_members WHERE album_id=? AND group_key IN ({marks})',
                [album_id, *drop])
    return db.execute('SELECT COUNT(*) FROM people_album_members WHERE album_id=?',
                      (album_id,)).fetchone()[0]


def group_albums(db, group_keys):
    """Для каждого ключа группы — альбомы, в которых он лежит."""
    group_keys = list(group_keys)
    result = {key: [] for key in group_keys}
    if not group_keys:
        return result
    known = {item['id']: item for item in tree(db)}
    marks = ','.join('?' * len(group_keys))
    for group_key, album_id in db.execute(
            f'SELECT group_key,album_id FROM people_album_members WHERE group_key IN ({marks})',
            group_keys):
        item = known.get(album_id)
        if item and group_key in result:
            result[group_key].append({'id': album_id, 'title': item['title'],
                                      'trail': item['trail']})
    for values in result.values():
        values.sort(key=lambda item: item['trail'])
    return result


def hidden_group_keys(db):
    """Ключи групп в скрытом альбоме — своём или у кого-то из предков."""
    hidden = set()
    for item in tree(db):
        if item['effectively_hidden']:
            hidden.update(item['member_keys'])
    return hidden
