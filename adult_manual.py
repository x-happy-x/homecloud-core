"""Ручная отметка 18+: человек поправляет автоматическую оценку.

Отметка пишется в photo_adult_analysis (её читают все фильтры, замыливание и
подборки), а прежняя автоматическая оценка сохраняется в photo_adult_manual,
чтобы её можно было вернуть. Этап 18+ и пересчёт отмеченные файлы не трогают.
Отметка действует на всю группу копий (photo_copies): это один и тот же кадр.
"""
from datetime import datetime, timezone
import json

SCHEMA = '''
CREATE TABLE IF NOT EXISTS photo_adult_manual (
  path TEXT PRIMARY KEY, rating TEXT NOT NULL, author TEXT NOT NULL DEFAULT '',
  marked_at TEXT NOT NULL, previous_json TEXT);
'''
# safe — «это не 18+», explicit — «это 18+»; None — вернуть автоматическую.
RATINGS = {'safe', 'explicit'}
_PREVIOUS = ('size', 'modified', 'rating', 'adult_score', 'tags_json', 'regions_json',
             'description', 'detector_model', 'tagger_model', 'status', 'error', 'analyzed_at')


def ensure_schema(db):
    db.executescript(SCHEMA)


def _table(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                      (name,)).fetchone() is not None


def copy_group(db, paths):
    """Пути вместе со всеми их копиями и оригиналами."""
    group = {path for path in paths if path}
    if not group or not _table(db, 'photo_copies'):
        return sorted(group)
    for start in range(0, len(paths), 400):
        batch = list(paths)[start:start + 400]
        marks = ','.join('?' * len(batch))
        originals = {row[0] for row in db.execute(
            f'SELECT original FROM photo_copies WHERE path IN ({marks})', batch)}
        originals |= set(batch)
        omarks = ','.join('?' * len(originals))
        group |= originals
        group |= {row[0] for row in db.execute(
            f'SELECT path FROM photo_copies WHERE original IN ({omarks})', list(originals))}
    return sorted(group)


def mark(db, paths, rating, author=''):
    """Поставить (rating = safe|explicit) или снять (None) ручную отметку. Вернёт пути."""
    if rating is not None and rating not in RATINGS:
        raise ValueError('Отметка 18+ бывает только safe или explicit')
    ensure_schema(db)
    targets = [path for path in copy_group(db, paths)
               if db.execute("SELECT 1 FROM photos WHERE path=?", (path,)).fetchone()]
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with db:
        for path in targets:
            existing = db.execute('SELECT previous_json FROM photo_adult_manual WHERE path=?',
                                  (path,)).fetchone()
            if rating is None:
                if existing is None:
                    continue
                db.execute('DELETE FROM photo_adult_analysis WHERE path=?', (path,))
                previous = json.loads(existing[0]) if existing[0] else None
                if previous:
                    columns = ','.join(('path', *_PREVIOUS))
                    db.execute(f'INSERT INTO photo_adult_analysis({columns}) '
                               f'VALUES({",".join("?" * (len(_PREVIOUS) + 1))})',
                               (path, *(previous.get(name) for name in _PREVIOUS)))
                db.execute('DELETE FROM photo_adult_manual WHERE path=?', (path,))
                continue
            if existing is None:
                row = db.execute(f'SELECT {",".join(_PREVIOUS)} FROM photo_adult_analysis '
                                 'WHERE path=?', (path,)).fetchone()
                previous = json.dumps(dict(zip(_PREVIOUS, row)), ensure_ascii=False) if row else None
            else:
                previous = existing[0]
            size, modified = db.execute('SELECT size,modified FROM photos WHERE path=?',
                                        (path,)).fetchone()
            db.execute(
                'INSERT INTO photo_adult_analysis(path,size,modified,rating,adult_score,'
                'regions_json,detector_model,status,analyzed_at) '
                "VALUES(?,?,?,?,?,'[]','manual','ok',?) ON CONFLICT(path) DO UPDATE SET "
                'rating=excluded.rating,adult_score=excluded.adult_score,'
                "regions_json='[]',detector_model='manual',status='ok',error=NULL,"
                'analyzed_at=excluded.analyzed_at',
                (path, size or 0, modified or 0, rating, 1.0 if rating == 'explicit' else 0.0, now))
            db.execute(
                'INSERT INTO photo_adult_manual(path,rating,author,marked_at,previous_json) '
                'VALUES(?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET rating=excluded.rating,'
                'author=excluded.author,marked_at=excluded.marked_at',
                (path, rating, author or '', now, previous))
    return targets


def manual_ratings(db, paths):
    """{путь: safe|explicit} для отмеченных вручную."""
    if not paths or not _table(db, 'photo_adult_manual'):
        return {}
    found = {}
    paths = list(paths)
    for start in range(0, len(paths), 400):
        batch = paths[start:start + 400]
        found.update(db.execute(
            f'SELECT path,rating FROM photo_adult_manual WHERE path IN ({",".join("?" * len(batch))})',
            batch))
    return found
