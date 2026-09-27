"""Копии файлов: не обрабатывать одно и то же дважды.

В каталоге много точных копий (одни и те же снимки в разных папках и на
разных дисках). Этапы анализа и лиц теперь считают только оригинал, а
готовые результаты переносятся на копии.

* ``keys`` — сразу после описи: быстрый ключ файла (размер + sha1 первых и
  последних 64 КБ) — только для файлов, у которых размер совпадает с чьим-то
  ещё: другие копиями быть не могут. Читается начало и конец через драйвер
  источника, файл целиком не копируется. Затем пересобирается photo_copies:
  у каждой группы одинаковых ключей оригинал — первый по пути, остальные —
  копии.
* ``propagate`` — после этапов: строки таблиц анализа оригинала
  копируются на его копии, которых там ещё нет.

Этапы берут файлы через ``pathkeys.analysis_scope_sql`` — без копий.
Превью сетки делаются для всех файлов: галерее нужны и копии.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import catalogdb
import pathkeys
import sources

EDGE = 64 * 1024
WORKERS = 8
SCHEMA = '''
CREATE TABLE IF NOT EXISTS photo_keys (
  path TEXT PRIMARY KEY, size INTEGER NOT NULL, modified INTEGER NOT NULL,
  quick TEXT NOT NULL, computed REAL NOT NULL);
CREATE INDEX IF NOT EXISTS photo_keys_quick ON photo_keys(quick);
CREATE TABLE IF NOT EXISTS photo_copies (
  path TEXT PRIMARY KEY, original TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS photo_copies_original ON photo_copies(original);
'''
# Таблицы результатов по файлу: их строки оригинала переносятся на копии.
# Лица сюда не входят: у копии своих лиц нет, иначе одно фото давало бы
# несколько одинаковых лиц в группах.
RESULT_TABLES = ('photo_analysis', 'photo_embeddings', 'photo_adult_analysis', 'photo_curation',
                 'video_speech', 'video_speech_segments', 'video_diarization', 'video_speakers',
                 'video_speaker_turns', 'router_predictions')


def ensure_schema(db):
    db.executescript(SCHEMA)


def write_progress(path, **values):
    if path is None:
        return
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')  # свой у каждого процесса
    temporary.write_text(json.dumps({**values, 'updated_at': time.time(), 'pid': os.getpid()},
                                    ensure_ascii=False), encoding='utf-8')
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))


def quick_key(stream, size):
    """sha1 от размера, первых и последних EDGE байт. Совпадение — копия."""
    digest = hashlib.sha1(str(size).encode('ascii'))
    digest.update(stream.read(EDGE))
    if size > EDGE:
        stream.seek(max(EDGE, size - EDGE))
        digest.update(stream.read(EDGE))
    return digest.hexdigest()


def open_file(key):
    """Поток файла без копирования целиком: драйвер источника или локальный путь."""
    if not pathkeys.is_key(key):
        return open(key, 'rb')
    driver, native = sources.core_access().resolve(key)
    local = driver.local_path(native)
    return open(local, 'rb') if local else driver.open(native)


def pending(db, roots, paths, force=False):
    """Файлы задания без свежего ключа, у которых размер совпадает с чьим-то ещё."""
    where, values = pathkeys.scope_sql(roots, paths, 'photos.path')
    stale = '' if force else (
        ' AND NOT EXISTS (SELECT 1 FROM photo_keys k WHERE k.path=photos.path '
        'AND k.size=photos.size AND k.modified=photos.modified)')
    return db.execute(
        "SELECT photos.path,photos.size,photos.modified FROM photos WHERE photos.status='ok' "
        'AND photos.size>0 AND photos.size IN (SELECT size FROM photos '
        "WHERE status='ok' GROUP BY size HAVING COUNT(*)>1)" + where + stale +
        ' ORDER BY photos.path', values).fetchall()


def rebuild_copies(db):
    """Оригинал группы одинаковых ключей — первый по пути, остальные — копии.
    Старое содержимое удаляется в той же транзакции, что вставляется новое."""
    with db:
        db.execute('DELETE FROM photo_copies')
        db.execute(
            'INSERT INTO photo_copies(path,original) '
            'SELECT k.path, g.original FROM photo_keys k '
            "JOIN photos p ON p.path=k.path AND p.status='ok' AND p.size=k.size "
            'AND p.modified=k.modified '
            'JOIN (SELECT k2.quick, MIN(k2.path) AS original FROM photo_keys k2 '
            "JOIN photos p2 ON p2.path=k2.path AND p2.status='ok' AND p2.size=k2.size "
            'AND p2.modified=k2.modified GROUP BY k2.quick HAVING COUNT(*)>1) g '
            'ON g.quick=k.quick WHERE k.path<>g.original')
    return db.execute('SELECT COUNT(*) FROM photo_copies').fetchone()[0]


def keys(args):
    db = catalogdb.connect(args.catalog, timeout=60)
    try:
        ensure_schema(db)
        rows = pending(db, args.root, args.path, args.force)
        state = {'status': 'running', 'total': len(rows), 'completed': 0, 'errors': 0,
                 'current': ''}
        write_progress(args.progress_file, **state)
        started = time.time()

        # Источник, который не отвечает, пропускается целиком до следующего
        # задания: иначе каждый его файл ждал бы таймаута подключения.
        dead = set()

        def one(row):
            key, size, modified = row
            source = pathkeys.source_of(key)
            if source in dead:
                return key, size, modified, None, 'источник недоступен'
            try:
                with open_file(key) as stream:
                    return key, size, modified, quick_key(stream, size), ''
            except FileNotFoundError as exc:
                return key, size, modified, None, str(exc)
            except (sources.SourceError, OSError) as exc:
                if source and not isinstance(exc, PermissionError):
                    dead.add(source)
                return key, size, modified, None, str(exc)
            except Exception as exc:
                return key, size, modified, None, str(exc)

        found = []
        with ThreadPoolExecutor(WORKERS) as pool:
            for key, size, modified, quick, error in pool.map(one, rows):
                if args.stop_file and args.stop_file.exists():
                    break
                state['completed'] += 1
                state['current'] = key
                if quick is None:
                    state['errors'] += 1
                    if error != 'источник недоступен':
                        print(f'{key}: {error}', flush=True)
                else:
                    found.append((key, size, modified, quick, time.time()))
                if len(found) >= 500:
                    save(db, found)
                    found.clear()
                if state['completed'] % 50 == 0:
                    write_progress(args.progress_file, **state)
        save(db, found)
        copies = rebuild_copies(db)
        print(f'Ключи: {state["completed"]} файлов за {time.time() - started:.0f} с, '
              f'ошибок {state["errors"]}; копий в каталоге: {copies}'
              + (f'; недоступны: {", ".join(sorted(dead))}' if dead else ''), flush=True)
        write_progress(args.progress_file, **{**state, 'status': 'completed', 'copies': copies})
    finally:
        db.close()


def save(db, rows):
    if not rows:
        return
    with db:
        db.executemany(
            'INSERT INTO photo_keys(path,size,modified,quick,computed) VALUES(?,?,?,?,?) '
            'ON CONFLICT(path) DO UPDATE SET size=excluded.size,modified=excluded.modified,'
            'quick=excluded.quick,computed=excluded.computed', rows)


def propagate(db):
    """Строки таблиц результатов оригинала — на его копии, где их ещё нет."""
    ensure_schema(db)
    moved = {}
    existing = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in RESULT_TABLES:
        if table not in existing:
            continue
        columns = [row[1] for row in db.execute(f'PRAGMA table_info({table})')]
        if 'path' not in columns:
            continue
        names = ','.join(columns)
        picked = ','.join('c.path' if column == 'path' else f't.{column}' for column in columns)
        with db:
            cursor = db.execute(
                f'INSERT OR IGNORE INTO {table}({names}) SELECT {picked} FROM {table} t '
                'JOIN photo_copies c ON c.original=t.path '
                f'WHERE NOT EXISTS (SELECT 1 FROM {table} x WHERE x.path=c.path)')
        moved[table] = cursor.rowcount
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('keys', 'propagate'))
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--root', action='append', default=[])
    parser.add_argument('--path', action='append', default=[])
    parser.add_argument('--progress-file', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.command == 'keys':
        keys(args)
        return
    db = catalogdb.connect(args.catalog, timeout=60)
    try:
        write_progress(args.progress_file, status='running', total=1, completed=0)
        moved = propagate(db)
        print('Перенесено на копии:', moved, flush=True)
        write_progress(args.progress_file, status='completed', total=1, completed=1, moved=moved)
    finally:
        db.close()


if __name__ == '__main__':
    sys.exit(main())
