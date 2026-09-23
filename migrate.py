"""Перенос каталогов отдельных компьютеров в общий каталог хаба.

До хаба у каждого компьютера был свой каталог, и снимок в нём назывался
просто путём: `D:\\Фото\\a.jpg`. В общем каталоге путь не уникален, поэтому
ключ — `источник:путь` (см. pathkeys.py).

* `prefix(db, source)` — на месте переписывает пути каталога в ключи этого
  источника: сами снимки, папки описи, исключения, историю заданий и правила
  путей в настройках. Каталог первого компьютера так и становится общим.
* `merge(target, source_file, source)` — вливает каталог ещё одного
  компьютера в общий: лица, группы, имена, альбомы и анализ. Номера лиц,
  групп и видеоличностей сдвигаются, люди с тем же именем сливаются в одного,
  подборки не переносятся — их пересоберёт ядро.

Запуск руками:

    python migrate.py prefix каталог.sqlite pc-x
    python migrate.py merge общий.sqlite чужой.sqlite pc-a
"""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import pathkeys

# Таблица → колонки с путём снимка или папки.
PATH_COLUMNS = {
    'photos': ('path', 'dir'),
    'faces': ('path',),
    'photo_analysis': ('path',),
    'photo_adult_analysis': ('path',),
    'photo_curation': ('path',),
    'photo_embeddings': ('path',),
    'photo_hashes': ('path',),
    'photo_thumbs': ('path',),
    'photo_dirs': ('path', 'parent'),
    'scan_roots': ('path',),
    'scan_exclusions': ('path',),
    'album_photos': ('path',),
    'highlight_groups': ('cover_path',),
    'highlight_photos': ('path',),
    'hidden_photos': ('path', 'stored'),
    'router_batch_items': ('path',),
    'router_predictions': ('path',),
    'router_reviews': ('path',),
    'router_skips': ('path',),
    'router_training_labels': ('path',),
    'video_diarization': ('path',),
    'video_identities': ('path',),
    'video_people_hints': ('path',),
    'video_speaker_faces': ('path',),
    'video_speaker_turns': ('path',),
    'video_speakers': ('path',),
    'video_speech': ('path',),
    'video_speech_segments': ('path',),
}
RULE_SETTINGS = ('block_paths', 'allow_paths')


def _tables(db, schema='main'):
    return {row[0] for row in db.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type='table'")}


def _columns(db, table, schema='main'):
    return [row[1] for row in db.execute(f'PRAGMA {schema}.table_info("{table}")')]


def _legacy(column):
    """Путь без источника: двоеточия нет или это буква диска."""
    return f"instr({column},':')<=2"


def prefix(db, source):
    """Пути каталога одного компьютера → ключи источника `source`."""
    if not pathkeys.SOURCE_ID.match(source):
        raise ValueError(f'Некорректный id источника: {source!r}')
    head = source + ':'
    tables = _tables(db)
    counts = {}
    db.execute('PRAGMA foreign_keys=OFF')
    with db:
        for table, columns in PATH_COLUMNS.items():
            if table not in tables:
                continue
            present = set(_columns(db, table))
            for column in columns:
                if column not in present:
                    continue
                changed = db.execute(
                    f'UPDATE "{table}" SET "{column}"=?||"{column}" '
                    f'WHERE "{column}" IS NOT NULL AND "{column}"<>\'\' AND {_legacy(column)}',
                    (head,)).rowcount
                counts[f'{table}.{column}'] = changed
        if 'scan_runs' in tables:
            for run_id, roots, paths in db.execute(
                    'SELECT id,roots_json,paths_json FROM scan_runs').fetchall():
                fixed = [json.dumps([_key(head, item) for item in json.loads(raw or '[]')],
                                    ensure_ascii=False) for raw in (roots, paths)]
                db.execute('UPDATE scan_runs SET roots_json=?,paths_json=? WHERE id=?',
                           (*fixed, run_id))
        if 'settings' in tables:
            for key in RULE_SETTINGS:
                row = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
                if row:
                    db.execute('UPDATE settings SET value=? WHERE key=?',
                               (json.dumps(_rules(head, json.loads(row[0])), ensure_ascii=False),
                                key))
    return counts


def _key(head, value):
    value = str(value)
    return value if pathkeys.is_key(value) else head + value


def _rules(head, raw):
    """Правила путей тоже становятся правилами этого источника."""
    lines = [line.strip() for line in str(raw or '').replace('\r', '').split('\n')]
    return '\n'.join(_key(head, line) if line else line for line in lines).strip('\n')


def _copy(db, table, transform=None, conflict='IGNORE', where=''):
    """Строки src.table → main.table по общим колонкам; transform: колонка → выражение."""
    if table not in _tables(db, 'src') or table not in _tables(db):
        return 0
    wanted = set(_columns(db, table))
    columns = [column for column in _columns(db, table, 'src') if column in wanted]
    transform = transform or {}
    expressions = [transform.get(column, f'"{column}"') for column in columns]
    names = ','.join(f'"{column}"' for column in columns)
    return db.execute(
        f'INSERT OR {conflict} INTO main."{table}" ({names}) '
        f'SELECT {",".join(expressions)} FROM src."{table}" {where}').rowcount


def merge(target, source_file, source):
    """Вливает каталог компьютера в общий каталог хаба. Возвращает счётчики."""
    target, source_file = Path(target), Path(source_file)
    work = Path(tempfile.mkdtemp(prefix='homecloud-merge-', dir=target.parent))
    try:
        copy = work / 'source.sqlite'
        shutil.copyfile(source_file, copy)
        prepared = sqlite3.connect(copy)
        try:
            prefix(prepared, source)
        finally:
            prepared.close()
        if not target.exists() or target.stat().st_size == 0:
            # Общего каталога ещё нет — им становится каталог этого компьютера.
            shutil.copyfile(copy, target)
            return {'created': True}
        return _merge(target, copy)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _merge(target, copy):
    db = sqlite3.connect(target, timeout=120)
    db.execute('PRAGMA foreign_keys=OFF')
    db.execute('ATTACH DATABASE ? AS src', (str(copy),))
    counts = {}
    try:
        with db:
            main_tables = _tables(db)
            face_offset = db.execute('SELECT COALESCE(MAX(id),0) FROM faces').fetchone()[0] \
                if 'faces' in main_tables else 0
            label_offset = (db.execute('SELECT COALESCE(MAX(label),-1)+1 FROM face_clusters')
                            .fetchone()[0] if 'face_clusters' in main_tables else 0)
            identity_offset = (db.execute('SELECT COALESCE(MAX(id),0) FROM video_identities')
                               .fetchone()[0] if 'video_identities' in main_tables else 0)
            db.execute('CREATE TEMP TABLE map_people (old INTEGER PRIMARY KEY, new INTEGER)')
            db.execute('CREATE TEMP TABLE map_albums (old INTEGER PRIMARY KEY, new INTEGER)')
            db.execute('CREATE TEMP TABLE map_people_albums (old INTEGER PRIMARY KEY, new INTEGER)')

            # Люди: одно имя — один человек; ссылку на картотеку берём, если своей нет.
            if 'people' in _tables(db, 'src'):
                for person_id, name, created, bigfam_id in db.execute(
                        'SELECT id,name,created_at,bigfam_id FROM src.people').fetchall():
                    row = db.execute('SELECT id,bigfam_id FROM people WHERE name=?',
                                     (name,)).fetchone()
                    if row is None:
                        new = db.execute('INSERT INTO people(name,created_at,bigfam_id) '
                                         'VALUES(?,?,?)', (name, created, bigfam_id)).lastrowid
                    else:
                        new = row[0]
                        if bigfam_id and not row[1]:
                            db.execute('UPDATE people SET bigfam_id=? WHERE id=?',
                                       (bigfam_id, new))
                    db.execute('INSERT INTO map_people VALUES(?,?)', (person_id, new))
            person = '(SELECT new FROM map_people WHERE old={})'
            face = f'("face_id"+{face_offset})'

            def group_key(column):
                return (f"CASE WHEN {column} LIKE 'person:%' THEN 'person:'||"
                        f"(SELECT new FROM map_people WHERE old=CAST(substr({column},8) AS INTEGER)) "
                        f"WHEN {column} LIKE 'auto:%' THEN 'auto:'||"
                        f"(CAST(substr({column},6) AS INTEGER)+{label_offset}) ELSE {column} END")

            for table in ('photos', 'photo_analysis', 'photo_adult_analysis', 'photo_curation',
                          'photo_embeddings', 'photo_hashes', 'photo_thumbs', 'photo_dirs',
                          'scan_roots', 'scan_exclusions', 'hidden_photos', 'router_batches',
                          'router_batch_items', 'router_predictions', 'router_reviews',
                          'router_skips', 'router_training_labels', 'router_models',
                          'curation_prompts', 'video_diarization', 'video_people_hints',
                          'video_speaker_turns', 'video_speech', 'video_speech_segments'):
                counts[table] = _copy(db, table)
            counts['faces'] = _copy(db, 'faces', {'id': f'("id"+{face_offset})'})
            for table in ('face_authenticity', 'face_exclusions', 'face_quality',
                          'face_track_data', 'face_track_samples'):
                counts[table] = _copy(db, table, {'face_id': face})
            counts['face_clusters'] = _copy(db, 'face_clusters', {
                'face_id': face,
                'label': f'CASE WHEN "label">=0 THEN "label"+{label_offset} ELSE "label" END'})
            counts['face_people'] = _copy(db, 'face_people', {
                'face_id': face, 'person_id': person.format('"person_id"')})
            counts['face_identity_conflicts'] = _copy(db, 'face_identity_conflicts', {
                'face_a': f'("face_a"+{face_offset})', 'face_b': f'("face_b"+{face_offset})',
                'history_id': 'NULL'})
            counts['video_identities'] = _copy(db, 'video_identities', {
                'id': f'("id"+{identity_offset})'})
            counts['face_track_identities'] = _copy(db, 'face_track_identities', {
                'face_id': face,
                'identity_id': f'CASE WHEN "identity_id" IS NULL THEN NULL '
                               f'ELSE "identity_id"+{identity_offset} END'})
            counts['video_speaker_faces'] = _copy(db, 'video_speaker_faces', {'face_id': face})
            counts['video_speakers'] = _copy(db, 'video_speakers', {
                'suggested_person_id': person.format('"suggested_person_id"'),
                'assigned_person_id': person.format('"assigned_person_id"')})
            counts['voice_prints'] = _copy(db, 'voice_prints', {
                'person_id': person.format('"person_id"')})
            counts['person_age_profiles'] = _copy(db, 'person_age_profiles', {
                'person_id': person.format('"person_id"')})
            counts['group_avatars'] = _copy(db, 'group_avatars', {
                'group_key': group_key('"group_key"'), 'face_id': face})

            # Альбомы: одноимённый альбом в том же месте — тот же альбом.
            counts['albums'] = _merge_tree(db, 'albums', 'map_albums',
                                           ('created',))
            counts['album_photos'] = _copy(db, 'album_photos', {
                'album_id': '(SELECT new FROM map_albums WHERE old="album_id")'})
            counts['people_albums'] = _merge_tree(db, 'people_albums', 'map_people_albums',
                                                  ('created', 'hidden'))
            counts['people_album_members'] = _copy(db, 'people_album_members', {
                'album_id': '(SELECT new FROM map_people_albums WHERE old="album_id")',
                'group_key': group_key('"group_key"')})

            if 'scan_runs' in _tables(db, 'src') and 'scan_runs' in main_tables:
                columns = [column for column in _columns(db, 'scan_runs', 'src') if column != 'id']
                names = ','.join(columns)
                counts['scan_runs'] = db.execute(
                    f'INSERT OR IGNORE INTO scan_runs ({names}) '
                    f'SELECT {names} FROM src.scan_runs').rowcount

            # Правила путей источника добавляются к общим.
            if 'settings' in main_tables and 'settings' in _tables(db, 'src'):
                for key in RULE_SETTINGS:
                    theirs = db.execute('SELECT value FROM src.settings WHERE key=?',
                                        (key,)).fetchone()
                    if not theirs or not json.loads(theirs[0]):
                        continue
                    ours = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
                    lines = [line for line in (json.loads(ours[0]) if ours else '').split('\n')
                             if line.strip()]
                    for line in json.loads(theirs[0]).split('\n'):
                        if line.strip() and line not in lines:
                            lines.append(line)
                    db.execute('INSERT INTO settings(key,value,changed) VALUES(?,?,0) '
                               'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                               (key, json.dumps('\n'.join(lines), ensure_ascii=False)))
            counts['offsets'] = {'faces': face_offset, 'labels': label_offset,
                                 'identities': identity_offset}
    finally:
        db.execute('DETACH DATABASE src')
        db.close()
    return counts


def _merge_tree(db, table, mapping, extra):
    """Дерево альбомов: родитель раньше детей, совпадение — по (родитель, название)."""
    if table not in _tables(db, 'src') or table not in _tables(db):
        return 0
    present = set(_columns(db, table))
    fields = [name for name in extra if name in present
              and name in _columns(db, table, 'src')]
    rows = db.execute(f'SELECT id,parent_id,title{"".join(","+name for name in fields)} '
                      f'FROM src."{table}"').fetchall()
    known = {0: 0}
    pending = list(rows)
    added = 0
    while pending:
        progress = False
        for row in list(pending):
            old, parent, title = row[0], row[1], row[2]
            if parent not in known:
                continue
            new_parent = known[parent]
            found = db.execute(f'SELECT id FROM "{table}" WHERE parent_id=? AND title=?',
                               (new_parent, title)).fetchone()
            if found:
                known[old] = found[0]
            else:
                names = ','.join(['parent_id', 'title', *fields])
                marks = ','.join('?' * (2 + len(fields)))
                known[old] = db.execute(f'INSERT INTO "{table}" ({names}) VALUES({marks})',
                                        (new_parent, title, *row[3:])).lastrowid
                added += 1
            db.execute(f'INSERT OR REPLACE INTO {mapping} VALUES(?,?)', (old, known[old]))
            pending.remove(row)
            progress = True
        if not progress:
            break
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    first = sub.add_parser('prefix')
    first.add_argument('catalog', type=Path)
    first.add_argument('source')
    second = sub.add_parser('merge')
    second.add_argument('target', type=Path)
    second.add_argument('catalog', type=Path)
    second.add_argument('source')
    args = parser.parse_args()
    if args.command == 'prefix':
        db = sqlite3.connect(args.catalog)
        try:
            print(json.dumps(prefix(db, args.source), ensure_ascii=False, indent=1))
        finally:
            db.close()
    else:
        print(json.dumps(merge(args.target, args.catalog, args.source), ensure_ascii=False,
                         indent=1))


if __name__ == '__main__':
    main()
