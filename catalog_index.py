"""Опись файлов каталога: что нашли, что изменилось и что исключено вручную.

Сканирование разделено на два шага. Сначала опись: обход выбранных папок,
сравнение с прошлым разом и дерево найденного. Потом уже этапы обработки —
они работают только по тому, что осталось после исключений.
"""
from collections import defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3

import pathrules
import video as video_media

PICTURES = {'.jpg', '.jpeg', '.png', '.webp'}
# Опись обходит и ролики: лица ищем и в них.
SUPPORTED = PICTURES | video_media.SUPPORTED
# Служебные каталоги: кэши, окружения, системные папки и типичные папки игр.
EXCLUDED_NAMES = {
    '.git', '.gradle', '.idea', '.svn', '.venv', '__pycache__', 'appdata',
    'bin', 'cache', 'caches', 'node_modules', 'obj', 'packages',
    'program files', 'program files (x86)', 'site-packages', 'temp', 'tmp',
    'venv', 'virtual machines', 'windows', 'my games', 'игра',
    # Корзина и системные служебные папки Windows — не библиотека фотографий,
    # а место для удалённого и восстановления системы; их содержимое не должно
    # ни попадать в галерею, ни считаться «удалённым», если его не сканировать.
    '$recycle.bin', 'recycler', 'system volume information',
}
# Состояния относительно предыдущей описи.
STATES = ('new', 'changed', 'known', 'missing', 'excluded')


def ensure_schema(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS photos (
          path TEXT PRIMARY KEY, size INTEGER, modified INTEGER, model TEXT,
          status TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS photo_dirs (
          path TEXT PRIMARY KEY, parent TEXT, name TEXT,
          files INTEGER NOT NULL DEFAULT 0, subtree INTEGER NOT NULL DEFAULT 0,
          new INTEGER NOT NULL DEFAULT 0, changed INTEGER NOT NULL DEFAULT 0,
          missing INTEGER NOT NULL DEFAULT 0, excluded INTEGER NOT NULL DEFAULT 0,
          bytes INTEGER NOT NULL DEFAULT 0, updated_at TEXT);
        CREATE INDEX IF NOT EXISTS photo_dirs_parent ON photo_dirs(parent);
        CREATE TABLE IF NOT EXISTS scan_exclusions (
          path TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS scan_roots (
          path TEXT PRIMARY KEY, first_run TEXT NOT NULL, last_run TEXT NOT NULL,
          files INTEGER NOT NULL DEFAULT 0, new INTEGER NOT NULL DEFAULT 0,
          changed INTEGER NOT NULL DEFAULT 0, missing INTEGER NOT NULL DEFAULT 0,
          excluded INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0);
    ''')
    columns = {row[1] for row in db.execute('PRAGMA table_info(photos)')}
    for name in ('dir', 'state', 'last_run'):
        if name not in columns:
            db.execute(f'ALTER TABLE photos ADD COLUMN {name} TEXT')
    db.execute('CREATE INDEX IF NOT EXISTS photos_dir ON photos(dir)')
    db.commit()
    if 'dir' not in columns:
        backfill_dirs(db)
    return db


def backfill_dirs(db):
    """Старые записи не знают своей папки — проставляем один раз."""
    rows = db.execute('SELECT path FROM photos WHERE dir IS NULL').fetchall()
    db.executemany('UPDATE photos SET dir=? WHERE path=?',
                   [(str(Path(path).parent), path) for (path,) in rows])
    db.commit()


def bounds(directory):
    """Границы диапазона путей внутри папки: works для обычного сравнения строк."""
    prefix = str(directory).rstrip('\\/') + os.sep
    return prefix, prefix + '￿'


def exclusions(db):
    dirs, files = [], set()
    for path, kind in db.execute('SELECT path,kind FROM scan_exclusions'):
        (dirs.append(path) if kind == 'dir' else files.add(path))
    return dirs, files


def _builtin_excluded(path):
    """Путь лежит внутри служебной папки вроде node_modules или корзины."""
    return any(part.casefold() in EXCLUDED_NAMES for part in Path(path).parts)


def is_excluded(path, excluded_dirs, excluded_files):
    if path in excluded_files:
        return True
    return any(path == item or path.startswith(item.rstrip('\\/') + os.sep)
               for item in excluded_dirs)


def walk(roots, excluded_dirs, report=None, stop=None, rules=None):
    """Находит поддерживаемые файлы, обходя служебные каталоги стороной."""
    found = []
    seen_dirs = 0
    block, allow = rules or ([], [])
    skip = {item.rstrip('\\/').casefold() for item in excluded_dirs}
    # Junction/симлинк может замкнуть дерево само на себя (частый случай на
    # Windows — библиотеки Steam, папки облачных синхронизаций): без защиты
    # обход одной и той же папки повторяется бесконечно и опись не завершится.
    visited = set()
    for root in roots:
        for current, directories, files in os.walk(root, followlinks=False):
            if stop and stop():
                return found, True
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
            directories[:] = [name for name in directories
                              if name.casefold() not in EXCLUDED_NAMES
                              and str(Path(current) / name).casefold() not in skip
                              # Заблокированная папка не обходится вовсе, а не
                              # просто теряет свои файлы после полного обхода.
                              and pathrules.enter(str(Path(current) / name), block, allow)]
            for name in files:
                target = str(Path(current) / name)
                if (Path(name).suffix.casefold() in SUPPORTED
                        and not pathrules.blocked(target, block, allow)):
                    found.append(target)
            seen_dirs += 1
            if report and seen_dirs % 25 == 0:
                report(len(found), str(current))
    return found, False


def take_inventory(db, roots, report=None, stop=None, rules=None, batch=3000):
    """Сверяет папки с каталогом: что новое, что изменилось, что пропало.

    Найденное пишется в базу порциями по ходу дела, а не одной транзакцией
    в самом конце: обход всего диска может идти долго, и если его остановить
    или он упадёт на середине, то, что уже нашли, не должно пропадать.
    """
    ensure_schema(db)
    run = datetime.now(timezone.utc).isoformat()
    excluded_dirs, excluded_files = exclusions(db)
    roots = [str(Path(root).resolve()) for root in roots]
    found, walk_stopped = walk(roots, excluded_dirs, report=report, stop=stop,
                               rules=rules)

    known = {}
    for root in roots:
        low, high = bounds(root)
        known.update({row[0]: row[1:] for row in db.execute(
            'SELECT path,size,modified,status FROM photos WHERE path>=? AND path<?',
            (low, high))})

    summary = dict.fromkeys(STATES, 0)
    summary['bytes'] = 0
    per_dir = defaultdict(lambda: dict.fromkeys(('files', 'bytes', *STATES), 0))
    total = len(found)
    pending = []
    processed_stopped = False

    def flush():
        if not pending:
            return
        with db:
            db.executemany('''INSERT INTO photos(path,dir,size,modified,status,state,last_run,model)
                VALUES(?,?,?,?,?,?,?,'inventory-v2') ON CONFLICT(path) DO UPDATE SET
                dir=excluded.dir,size=excluded.size,modified=excluded.modified,
                status=excluded.status,state=excluded.state,last_run=excluded.last_run''', pending)
        pending.clear()

    for index, path in enumerate(found, 1):
        if stop and stop():
            processed_stopped = True
            break
        try:
            info = os.stat(path)
            size, modified = info.st_size, info.st_mtime_ns
        except OSError:
            size, modified = 0, 0
        previous = known.get(path)
        if is_excluded(path, excluded_dirs, excluded_files):
            state, status = 'excluded', 'excluded'
        elif previous is None:
            state, status = 'new', 'ok'
        elif (previous[0], previous[1]) != (size, modified):
            state, status = 'changed', 'ok'
        else:
            state = 'known'
            status = previous[2] if previous[2] in {'ok', 'ignored'} else 'ok'
        folder = str(Path(path).parent)
        pending.append((path, folder, size, modified, status, state, run))
        summary[state] += 1
        summary['bytes'] += size
        counters = per_dir[folder]
        counters['files'] += 1
        counters['bytes'] += size
        counters[state] += 1
        if len(pending) >= batch:
            flush()
        if report and (index % 200 == 0 or index == total):
            report(index, path, total)
    flush()

    # Найденное видно в дереве сразу — даже если обход прервали на середине.
    rebuild_dirs(db, roots, per_dir, run)
    stopped = walk_stopped or processed_stopped
    # «Пропало» имеет смысл только для полного прохода: на частичном обходе
    # это пометило бы ещё не дошедшие файлы как missing.
    if not stopped:
        block, allow = rules or ([], [])
        with db:
            for root in roots:
                low, high = bounds(root)
                stale = [row[0] for row in db.execute(
                    'SELECT path FROM photos WHERE path>=? AND path<? '
                    'AND (last_run IS NULL OR last_run!=?)', (low, high, run))]
                # Файл под правилом исключения путей (или в служебной папке
                # вроде корзины) не пропал — его просто не обходили; не путать
                # с настоящим missing, когда файл действительно исчез с диска.
                excluded_now, missing_now = [], []
                for path in stale:
                    reason = (pathrules.blocked(path, block, allow)
                             or _builtin_excluded(path)
                             or is_excluded(path, excluded_dirs, excluded_files))
                    (excluded_now if reason else missing_now).append(path)
                # last_run обязательно обновляем и здесь — иначе эти же строки
                # снова попадут в «устаревшие» на следующей описи и будут
                # пересчитываться заново при каждом скане без всякой пользы.
                if excluded_now:
                    db.executemany(
                        "UPDATE photos SET status='excluded',state='excluded',last_run=? "
                        'WHERE path=?', [(run, path) for path in excluded_now])
                    summary['excluded'] += len(excluded_now)
                if missing_now:
                    db.executemany(
                        "UPDATE photos SET status='missing',state='missing',last_run=? "
                        'WHERE path=?', [(run, path) for path in missing_now])
                    summary['missing'] += len(missing_now)
    # Корень записываем в любом случае — иначе остановленный обход показал бы
    # уже сохранённые файлы в базе, но не давал бы до них добраться через
    # дерево «Что найдено» (оно строится по списку scan_roots).
    with db:
        for root in roots:
            db.execute('''INSERT INTO scan_roots(path,first_run,last_run,files,new,changed,
                missing,excluded,bytes) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET last_run=excluded.last_run,files=excluded.files,
                new=excluded.new,changed=excluded.changed,missing=excluded.missing,
                excluded=excluded.excluded,bytes=excluded.bytes''',
                (root, run, run, *root_totals(db, root)))
    if not stopped:
        # Прошлые отдельные корни внутри только что полностью просканированной
        # папки больше не нужны отдельной строкой — родитель уже их покрыл,
        # и без этого «Что найдено» копит дубликаты вроде D:\xxx рядом с D:\.
        with db:
            for root in roots:
                low, high = bounds(root)
                db.execute('DELETE FROM scan_roots WHERE path>=? AND path<? AND path!=?',
                          (low, high, root))
    return {'run': run, 'roots': roots, 'stopped': stopped, **summary,
            'total': sum(summary[state] for state in STATES)}


def root_totals(db, root):
    low, high = bounds(root)
    row = db.execute('''SELECT COUNT(*),
        SUM(state='new'),SUM(state='changed'),SUM(state='missing'),SUM(state='excluded'),
        SUM(size) FROM photos WHERE path>=? AND path<?''', (low, high)).fetchone()
    return tuple(value or 0 for value in row)


def rebuild_dirs(db, roots, per_dir=None, run=None):
    """Складывает счётчики по папкам и поднимает их вверх по дереву."""
    run = run or datetime.now(timezone.utc).isoformat()
    if per_dir is None:
        per_dir = defaultdict(lambda: dict.fromkeys(('files', 'bytes', *STATES), 0))
        for root in roots:
            low, high = bounds(root)
            for folder, size, state in db.execute(
                    'SELECT dir,size,state FROM photos WHERE path>=? AND path<?', (low, high)):
                counters = per_dir[folder or root]
                counters['files'] += 1
                counters['bytes'] += size or 0
                if state in STATES:
                    counters[state] += 1

    totals = defaultdict(lambda: dict.fromkeys(('files', 'bytes', 'subtree', *STATES), 0))
    tops = [str(Path(root).resolve()).rstrip('\\/') for root in roots]
    for folder, counters in per_dir.items():
        current = Path(folder)
        chain = [str(current)]
        # Выше выбранной папки не поднимаемся: там лежат чужие файлы.
        while str(current).rstrip('\\/') not in tops:
            parent = current.parent
            if parent == current:
                break
            current = parent
            chain.append(str(current))
        own = totals[chain[0]]
        for key in ('files', 'bytes', *STATES):
            own[key] += counters[key]
        for level in chain:
            totals[level]['subtree'] += counters['files']
            if level != chain[0]:
                for key in ('bytes', *STATES):
                    totals[level][key] += counters[key]

    with db:
        for root in roots:
            low, high = bounds(root)
            db.execute('DELETE FROM photo_dirs WHERE path>=? AND path<?', (low, high))
            db.execute('DELETE FROM photo_dirs WHERE path=?', (str(Path(root).resolve()),))
        db.executemany('''INSERT INTO photo_dirs(path,parent,name,files,subtree,new,changed,
            missing,excluded,bytes,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET parent=excluded.parent,name=excluded.name,
            files=excluded.files,subtree=excluded.subtree,new=excluded.new,
            changed=excluded.changed,missing=excluded.missing,excluded=excluded.excluded,
            bytes=excluded.bytes,updated_at=excluded.updated_at''',
            [(path, str(Path(path).parent) if str(Path(path).parent) != path else None,
              Path(path).name or path, values['files'], values['subtree'], values['new'],
              values['changed'], values['missing'], values['excluded'], values['bytes'], run)
             for path, values in totals.items()])


def set_exclusions(db, add=(), remove=()):
    """Исключение — свойство каталога: его учитывают и опись, и все этапы."""
    ensure_schema(db)
    now = datetime.now(timezone.utc).isoformat()
    touched = set()
    with db:
        for path in add:
            path = str(Path(path))
            kind = 'dir' if Path(path).is_dir() else 'file'
            db.execute('INSERT INTO scan_exclusions(path,kind,created_at) VALUES(?,?,?) '
                       'ON CONFLICT(path) DO UPDATE SET kind=excluded.kind', (path, kind, now))
            if kind == 'dir':
                low, high = bounds(path)
                db.execute("UPDATE photos SET status='excluded',state='excluded' "
                           'WHERE path>=? AND path<?', (low, high))
            else:
                db.execute("UPDATE photos SET status='excluded',state='excluded' "
                           'WHERE path=?', (path,))
            touched.add(path)
        for path in remove:
            path = str(Path(path))
            db.execute('DELETE FROM scan_exclusions WHERE path=?', (path,))
            low, high = bounds(path)
            db.execute("UPDATE photos SET status='ok',state='known' "
                       "WHERE state='excluded' AND (path=? OR (path>=? AND path<?))",
                       (path, low, high))
            touched.add(path)
    # Заново считаем счётчики затронутых корней.
    roots = {row[0] for row in db.execute('SELECT path FROM scan_roots')}
    affected = {root for root in roots
                for path in touched
                if path == root or path.startswith(root.rstrip('\\/') + os.sep)
                or root.startswith(path.rstrip('\\/') + os.sep)}
    if affected:
        rebuild_dirs(db, sorted(affected))
        with db:
            for root in affected:
                db.execute('''UPDATE scan_roots SET files=?,new=?,changed=?,missing=?,
                    excluded=?,bytes=? WHERE path=?''', (*root_totals(db, root), root))
    return {'ok': True, 'excluded': db.execute(
        'SELECT COUNT(*) FROM scan_exclusions').fetchone()[0]}


def tree(db, path='', limit=400):
    """Дерево найденного: подпапки со счётчиками и файлы текущей папки."""
    ensure_schema(db)
    if not path:
        roots = [dict(zip(('path', 'first_run', 'last_run', 'files', 'new', 'changed',
                           'missing', 'excluded', 'bytes'), row))
                 for row in db.execute('''SELECT path,first_run,last_run,files,new,changed,
                     missing,excluded,bytes FROM scan_roots ORDER BY path''')]
        return {'path': '', 'parent': None, 'roots': roots, 'directories': [], 'files': []}
    path = str(Path(path))
    excluded_dirs, excluded_files = exclusions(db)
    directories = [dict(zip(('path', 'name', 'files', 'subtree', 'new', 'changed', 'missing',
                             'excluded', 'bytes'), row))
                   for row in db.execute('''SELECT path,name,files,subtree,new,changed,missing,
                       excluded,bytes FROM photo_dirs WHERE parent=? ORDER BY name''', (path,))]
    for item in directories:
        item['off'] = is_excluded(item['path'], excluded_dirs, excluded_files)
    files = [dict(zip(('path', 'size', 'state', 'status'), row))
             for row in db.execute('''SELECT path,size,state,status FROM photos
                 WHERE dir=? ORDER BY path LIMIT ?''', (path, limit))]
    for item in files:
        item['name'] = Path(item['path']).name
        item['off'] = item['status'] == 'excluded'
    own = db.execute('''SELECT files,subtree,new,changed,missing,excluded,bytes
        FROM photo_dirs WHERE path=?''', (path,)).fetchone()
    parent = str(Path(path).parent)
    return {'path': path, 'parent': None if parent == path else parent,
            'directories': directories, 'files': files,
            'totals': dict(zip(('files', 'subtree', 'new', 'changed', 'missing',
                                'excluded', 'bytes'), own)) if own else None,
            'truncated': len(files) >= limit}


def connect(catalog):
    db = sqlite3.connect(Path(catalog) / 'catalog.sqlite', timeout=30)
    return ensure_schema(db)
