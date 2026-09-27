"""Автоматические подборки: лучшее за месяц и год, события, «в этот день».

Работает только по SQLite: оценки берёт из `photo_curation` (см.
photo_curation.py), векторы — из `photo_embeddings`. Файлы не читаются,
модели не грузятся, так что пересборка стоит секунды.

Как отбирается подборка:

1. **Кандидаты** — годные по оценке снимки с надёжным временем съёмки, не
   спрятанные, не исключённые правилами и проверенные на 18+.
2. **События** — кадры по времени: большой разрыв начинает новое событие,
   средний — если сменилась сцена (средние SigLIP-векторы по краям разрыва
   непохожи) или место (координаты, когда они есть).
3. **Серии** — почти одинаковые кадры (dHash, очень близкий SigLIP, короткая
   очередь) схлопываются в лучший из них; остальные запоминаются как «серия».
4. **Разнообразие** — жадный MMR: на каждом шаге берётся кадр с наибольшим
   `оценка − λ·похожесть_на_уже_взятые − штраф_за_перебор_события/дня`.
   Кадр той же сцены, что уже взятый, не берётся вовсе.

Итог хранится в `highlight_groups` / `highlight_photos`, отдельно от
пользовательских альбомов. Замена старых подборок — одна транзакция.
"""
import argparse
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

import catalogdb
import highlight_themes
import pathkeys
import photo_curation
import places
import sources

SCHEMA = '''
CREATE TABLE IF NOT EXISTS highlight_groups (
  id INTEGER PRIMARY KEY,
  key TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  subtitle TEXT NOT NULL DEFAULT '',
  period_start TEXT, period_end TEXT,
  score REAL NOT NULL DEFAULT 0,
  photo_count INTEGER NOT NULL DEFAULT 0,
  cover_path TEXT,
  generated_at TEXT NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS highlight_groups_kind ON highlight_groups(kind, period_start);
CREATE TABLE IF NOT EXISTS highlight_photos (
  group_id INTEGER NOT NULL REFERENCES highlight_groups(id) ON DELETE CASCADE,
  position INTEGER NOT NULL,
  path TEXT NOT NULL,
  score REAL NOT NULL,
  pick INTEGER NOT NULL,
  reasons_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(group_id, position)
);
CREATE INDEX IF NOT EXISTS highlight_photos_path ON highlight_photos(path);
'''

KINDS = ('month', 'year', 'event', 'trip', 'theme', 'place', 'on-this-day')

# Все коэффициенты в одном месте: их удобно крутить и видно в meta подборки.
PARAMS = {
    # кандидаты
    'min_time_trust': 0.6,        # mtime-only в этом каталоге — почти всегда скачанное
    'require_adult_check': True,  # без анализа 18+ снимок в подборку не идёт
    'min_base_score': 0.35,
    'personal_weight': 0.15,      # добавка за крупные лица узнанных людей
    # события
    'event_hard_gap': 5 * 3600,
    'event_soft_gap': 60 * 60,
    'event_scene_break': 0.60,    # косинус средних векторов по краям разрыва
    'event_place_km': 25.0,
    'event_place_gap': 20 * 60,
    # серии и одна сцена
    'dup_hamming': 3,             # dHash: столько бит и меньше — копия
    'dup_hamming_loose': 6,       # …или столько, если и SigLIP ≥ dup_cos_loose
    'dup_cos_loose': 0.85,
    'dup_cos': 0.955,
    'burst_seconds': 15, 'burst_cos': 0.90,
    'scene_cos': 0.90,            # такое сходство с уже взятым — та же сцена, не берём
    'scene_burst_seconds': 120, 'scene_burst_cos': 0.84,
    # MMR
    'mmr_lambda': 0.55,
    'redundancy_from': 0.60, 'redundancy_to': 0.90,
    'min_gain': 0.35,
    'quality_floor': 0.50,        # ниже — в подборку не берём, даже чтобы добрать размер
    'shortlist_factor': 6,
    'min_event_score': 0.50,      # событие из одних проходных кадров не показываем
    # темы и места
    'theme_share': 0.40,          # такая доля кадров события в одной теме — событие называется по ней
    'place_min': 10,              # столько кандидатов в городе — у него своя подборка
    'trip_km': 100.0,             # событие дальше этого от дома — поездка
    'theme_year_min': 15,         # столько кадров темы за год — ещё и подборка «Котики · 2023»
}

# Меньше стольких кадров после отбора подборка выглядит случайной — не сохраняем.
MIN_PHOTOS = {'month': 4, 'year': 8, 'event': 3, 'trip': 3, 'theme': 6, 'place': 6, 'on-this-day': 2}

BUCKET_WEIGHTS = {
    'month': {'event': 0.25, 'day': 0.08},
    'year': {'event': 0.25, 'month': 0.20},
    'event': {'hour': 0.04},
    'trip': {'hour': 0.04, 'day': 0.04},
    'theme': {'event': 0.35, 'month': 0.10},
    'place': {'event': 0.30, 'month': 0.10},
    'on-this-day': {'event': 0.25},
}

MONTHS = ('январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август',
          'сентябрь', 'октябрь', 'ноябрь', 'декабрь')
MONTHS_GENITIVE = ('января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа',
                   'сентября', 'октября', 'ноября', 'декабря')


def ensure_schema(db):
    db.executescript(SCHEMA)
    db.commit()


def connect(catalog, check_same_thread=True):
    db = catalogdb.connect(catalog, timeout=60, check_same_thread=check_same_thread)
    photo_curation.ensure_schema(db)
    ensure_schema(db)
    return db


def ramp(value, start, stop):
    return max(0.0, min(1.0, (value - start) / (stop - start)))


def _table_exists(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                      (name,)).fetchone() is not None


# ---------------------------------------------------------------- кандидаты

class Candidate:
    __slots__ = ('path', 'taken', 'ts', 'source', 'trust', 'base', 'visual', 'technical',
                 'personal', 'score', 'dhash', 'model', 'vector', 'lat', 'lon', 'people',
                 'faces', 'event', 'series', 'place', 'theme', 'theme_z')

    def __init__(self):
        # Место и тема ставятся позже, по всем кандидатам сразу.
        self.place, self.theme, self.theme_z = None, None, 0.0

    def reasons(self):
        return {'base': round(self.base, 3),
                'visual': None if self.visual is None else round(self.visual, 3),
                'technical': round(self.technical, 3), 'personal': round(self.personal, 3),
                'time_source': self.source}


def load_candidates(db, params):
    """Годные снимки, отсортированные по времени, и счётчики отсева."""
    params = {**PARAMS, **(params or {})}
    adult = _table_exists(db, 'photo_adult_analysis')
    hidden = _table_exists(db, 'hidden_photos')
    rows = db.execute(f'''
        SELECT c.path, c.taken_at, c.taken_ts, c.taken_source, c.base_score, c.visual_score,
               c.technical_score, c.personal_score, c.dhash, c.embedding_model,
               c.latitude, c.longitude, c.people_json, c.face_count,
               {"a.rating, a.status" if adult else "NULL, NULL"},
               {"h.path IS NOT NULL" if hidden else "0"}
        FROM photo_curation c JOIN photos p ON p.path=c.path
        {"LEFT JOIN photo_adult_analysis a ON a.path=c.path" if adult else ""}
        {"LEFT JOIN hidden_photos h ON h.path=c.path" if hidden else ""}
        WHERE c.eligible=1 AND c.status='ok' AND p.status='ok'
          AND COALESCE(p.blocked,0)=0 AND p.size=c.size AND p.modified=c.modified
        ORDER BY c.taken_ts, c.path''').fetchall()
    stats = {'eligible': len(rows), 'hidden': 0, 'adult_unchecked': 0, 'adult': 0,
             'untrusted_time': 0, 'low_score': 0}
    found = []
    for (path, taken_at, ts, source, base, visual, technical, personal, dhash, model,
         lat, lon, people_json, face_count, rating, adult_status, is_hidden) in rows:
        if is_hidden:
            stats['hidden'] += 1
            continue
        # Оценка могла устареть относительно анализа 18+ — проверяем по месту.
        if rating is not None and adult_status == 'ok' and rating not in photo_curation.SAFE_RATINGS:
            stats['adult'] += 1
            continue
        if params['require_adult_check'] and (rating is None or adult_status != 'ok'):
            stats['adult_unchecked'] += 1
            continue
        trust = photo_curation.TIME_TRUST.get(source, 0.0)
        if trust < params['min_time_trust'] or ts is None:
            stats['untrusted_time'] += 1
            continue
        if base is None or base < params['min_base_score']:
            stats['low_score'] += 1
            continue
        item = Candidate()
        item.path, item.ts, item.source, item.trust = path, float(ts), source, trust
        item.taken = datetime.fromisoformat(taken_at)
        item.base, item.visual, item.technical = base, visual, technical or 0.0
        item.personal = personal or 0.0
        item.score = base + params['personal_weight'] * item.personal
        item.dhash = int(dhash, 16) if dhash else None
        item.model, item.vector = model, None
        item.lat, item.lon = lat, lon
        entries = json.loads(people_json or '[]')
        item.people = sorted({entry['person_id'] for entry in entries
                              if entry.get('person_id') is not None and entry.get('prominence', 0) > 0})
        item.faces = face_count or 0
        item.event = None
        item.series = []
        found.append(item)
    _attach_vectors(db, found)
    _attach_coords(db, found)
    stats['with_place'] = _attach_places(found)
    vectors = {model: highlight_themes.theme_vectors(db, model)
               for model in {item.model for item in found if item.vector is not None}}
    stats['themes'] = highlight_themes.assign(
        found, {model: value for model, value in vectors.items() if value}, params)
    stats['candidates'] = len(found)
    return found, stats


def _attach_coords(db, items):
    """Координат нет в оценке — берём из EXIF, который ядро собрало для превью сетки."""
    missing = {item.path: item for item in items if item.lat is None or item.lon is None}
    if not missing or not _table_exists(db, 'photo_thumbs'):
        return
    keys = list(missing)
    for offset in range(0, len(keys), 500):
        batch = keys[offset:offset + 500]
        marks = ','.join('?' * len(batch))
        for path, raw in db.execute(
                f"SELECT path,metadata_json FROM photo_thumbs WHERE path IN ({marks}) "
                "AND metadata_json LIKE '%latitude%'", batch):
            try:
                coords = (json.loads(raw) or {}).get('coords') or {}
            except (TypeError, ValueError):
                continue
            if coords.get('latitude') is not None and coords.get('longitude') is not None:
                missing[path].lat = float(coords['latitude'])
                missing[path].lon = float(coords['longitude'])


def _attach_places(items):
    """Ближайший город по координатам; справочника нет — мест просто не будет."""
    for item in items:
        item.place = None
    located = [item for item in items if item.lat is not None and item.lon is not None]
    if not located:
        return 0
    try:
        for item in located:
            item.place = places.nearest(item.lat, item.lon)
    except OSError:
        return 0
    return sum(item.place is not None for item in located)


def _attach_vectors(db, items):
    by_key = {(item.path, item.model): item for item in items if item.model}
    keys = list(by_key)
    for offset in range(0, len(keys), 500):
        batch = keys[offset:offset + 500]
        marks = ','.join('?' * len(batch))
        for path, model, blob in db.execute(
                f'SELECT path,model,embedding FROM photo_embeddings WHERE path IN ({marks})',
                [path for path, _ in batch]):
            item = by_key.get((path, model))
            if item is None:
                continue
            vector = np.frombuffer(blob, dtype='<f4').astype(np.float32)
            item.vector = vector / max(float(np.linalg.norm(vector)), 1e-12)


# ---------------------------------------------------------------- сходство

def cosine(first, second):
    if first.vector is None or second.vector is None or first.model != second.model:
        return None
    return float(first.vector @ second.vector)


def hamming(first, second):
    if first.dhash is None or second.dhash is None:
        return None
    return bin(first.dhash ^ second.dhash).count('1')


def relation(first, second, params, cos=None):
    """Насколько кадры повторяют друг друга: 'duplicate', 'scene' или None, и мягкая похожесть."""
    cos = cosine(first, second) if cos is None else cos
    ham = hamming(first, second)
    gap = abs(first.ts - second.ts)
    duplicate = (
        (ham is not None and ham <= params['dup_hamming'])
        or (ham is not None and cos is not None and ham <= params['dup_hamming_loose']
            and cos >= params['dup_cos_loose'])
        or (cos is not None and cos >= params['dup_cos'])
        or (cos is not None and gap <= params['burst_seconds'] and cos >= params['burst_cos']))
    if duplicate:
        return 'duplicate', 1.0, cos, ham
    scene = cos is not None and (
        cos >= params['scene_cos']
        or (gap <= params['scene_burst_seconds'] and cos >= params['scene_burst_cos']))
    soft = 0.0 if cos is None else ramp(cos, params['redundancy_from'], params['redundancy_to'])
    if cos is None and gap <= 60:
        soft = 0.5   # нечем сравнить, но снято почти одновременно
    return ('scene' if scene else None), (1.0 if scene else soft), cos, ham


# ---------------------------------------------------------------- события

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    value = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(min(1.0, value)))


def _mean_vector(items):
    vectors = [item.vector for item in items if item.vector is not None]
    models = {item.model for item in items if item.vector is not None}
    if not vectors or len(models) > 1:
        return None, None
    mean = np.mean(vectors, axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-12), models.pop()


def detect_events(items, params=None):
    """Разбивает отсортированные по времени кадры на события. Возвращает список списков."""
    params = {**PARAMS, **(params or {})}
    events = []
    current = []
    for index, item in enumerate(items):
        if not current:
            current.append(item)
            continue
        previous = current[-1]
        gap = item.ts - previous.ts
        why = None
        if gap > params['event_hard_gap']:
            why = 'time'
        elif gap > params['event_soft_gap']:
            before, model_before = _mean_vector(current[-5:])
            after, model_after = _mean_vector(items[index:index + 5])
            if before is not None and after is not None and model_before == model_after \
                    and float(before @ after) < params['event_scene_break']:
                why = 'scene'
        if why is None and gap > params['event_place_gap'] and None not in (
                previous.lat, previous.lon, item.lat, item.lon):
            if haversine_km(previous.lat, previous.lon, item.lat, item.lon) > params['event_place_km']:
                why = 'place'
        if why:
            events.append(current)
            current = []
        current.append(item)
    if current:
        events.append(current)
    for members in events:
        key = 'event:' + members[0].taken.strftime('%Y%m%d-%H%M%S')
        for item in members:
            item.event = key
    return events


# ---------------------------------------------------------------- отбор

def collapse_series(items, params, trace=None):
    """Схлопывает копии и очереди в лучший кадр. Возвращает представителей по убыванию оценки."""
    ordered = sorted(items, key=lambda item: (-item.score, item.path))
    representatives = []
    by_model = {}
    for item in ordered:
        item.series = []
        match = None
        pool = by_model.get(item.model, ([], None))
        members, matrix = pool
        cosines = (matrix @ item.vector) if (matrix is not None and item.vector is not None) else None
        # Сначала кандидаты по вектору (быстро), потом те, у кого вектора нет.
        order = (np.argsort(-cosines) if cosines is not None else range(len(members)))
        for position in list(order)[:25]:
            other = members[position]
            kind, _, cos, ham = relation(item, other, params,
                                         None if cosines is None else float(cosines[position]))
            if kind == 'duplicate':
                match = (other, cos, ham)
                break
        if match is None:
            for other in representatives:
                if other.vector is not None and item.vector is not None and other.model == item.model:
                    continue   # уже проверены по матрице
                kind, _, cos, ham = relation(item, other, params)
                if kind == 'duplicate':
                    match = (other, cos, ham)
                    break
        if match:
            other, cos, ham = match
            other.series.append(item.path)
            if trace is not None:
                trace[item.path] = {'decision': 'duplicate', 'of': other.path,
                                    'cos': None if cos is None else round(cos, 3), 'hamming': ham}
            continue
        representatives.append(item)
        if item.vector is not None:
            members.append(item)
            matrix = (item.vector[None, :] if matrix is None
                      else np.vstack([matrix, item.vector[None, :]]))
            by_model[item.model] = (members, matrix)
    return representatives


def bucket_keys(item):
    return {'event': item.event, 'day': item.taken.strftime('%Y-%m-%d'),
            'month': item.taken.strftime('%Y-%m'), 'hour': item.taken.strftime('%Y-%m-%d %H')}


def diversify(representatives, target, weights, params, trace=None, minimum=1):
    """Жадный MMR с запретом на ту же сцену и штрафом за перебор события/дня."""
    if not representatives or target <= 0:
        return []
    ordered = sorted(representatives, key=lambda item: (-item.score, item.path))
    # Короткий список: лучшие по оценке плюс лучшие два кадра каждого события —
    # иначе маленькое, но хорошее событие не доживёт до MMR.
    size = max(target * params['shortlist_factor'], 60)
    shortlist = ordered[:size]
    chosen = {item.path for item in shortlist}
    per_event = {}
    for item in ordered[size:]:
        count = per_event.get(item.event, 0)
        if count < 2 and sum(1 for other in shortlist if other.event == item.event) < 2:
            shortlist.append(item)
            chosen.add(item.path)
        per_event[item.event] = count + 1
    if trace is not None:
        for rank, item in enumerate(ordered):
            if item.path not in chosen:
                trace[item.path] = {'decision': 'not_shortlisted', 'rank': rank + 1,
                                    'shortlist': len(shortlist)}
    sizes = {}
    for item in representatives:
        for name, value in bucket_keys(item).items():
            sizes[(name, value)] = sizes.get((name, value), 0) + 1
    counts = {}
    selected = []
    redundancy = {item.path: (0.0, None) for item in shortlist}
    blocked = {}
    remaining = list(shortlist)
    while remaining and len(selected) < target:
        best = None
        for item in remaining:
            if item.path in blocked:
                continue
            soft, nearest = redundancy[item.path]
            penalty = 0.0
            for name, weight in weights.items():
                key = (name, bucket_keys(item)[name])
                taken = counts.get(key, 0)
                if taken:
                    penalty += weight * taken / (1 + math.log2(sizes.get(key, 1)))
            gain = item.score - params['mmr_lambda'] * soft - penalty
            if best is None or gain > best[0] + 1e-12 or (
                    abs(gain - best[0]) <= 1e-12 and item.path < best[1].path):
                best = (gain, item, soft, penalty, nearest)
        if best is None:
            break
        gain, item, soft, penalty, nearest = best
        if gain < params['min_gain'] and len(selected) >= minimum:
            if trace is not None:
                trace['__stop__'] = {'reason': 'min_gain', 'best_gain': round(gain, 3),
                                     'best': item.path}
            break
        selected.append((item, {'gain': round(gain, 3), 'redundancy': round(soft, 3),
                                'bucket_penalty': round(penalty, 3), 'nearest': nearest,
                                'pick': len(selected) + 1}))
        remaining.remove(item)
        for name in weights:
            key = (name, bucket_keys(item)[name])
            counts[key] = counts.get(key, 0) + 1
        for other in remaining:
            if other.path in blocked:
                continue
            kind, soft, cos, _ = relation(other, item, params)
            if kind in ('duplicate', 'scene'):
                blocked[other.path] = (item.path, cos)
            elif soft > redundancy[other.path][0]:
                redundancy[other.path] = (soft, item.path)
    if trace is not None:
        for path, (by, cos) in blocked.items():
            trace[path] = {'decision': 'same_scene', 'as': by,
                           'cos': None if cos is None else round(cos, 3)}
        picked = {item.path for item, _ in selected}
        for item in remaining:
            if item.path not in blocked and item.path not in picked:
                soft, nearest = redundancy[item.path]
                trace[item.path] = {'decision': 'lost_to_diversity', 'score': round(item.score, 3),
                                    'redundancy': round(soft, 3), 'nearest': nearest}
        for item, info in selected:
            trace[item.path] = {'decision': 'selected', **info}
    return selected


def target_size(count, factor, low, high):
    return int(max(low, min(high, round(factor * math.sqrt(max(count, 0))))))


def build_group(kind, key, title, subtitle, items, target_range, params, trace=None,
                min_representatives=1, extra_meta=None):
    """Одна подборка из набора кандидатов. None — если набирать не из чего."""
    representatives = collapse_series(items, params, trace)
    strong = [item for item in representatives if item.score >= params['quality_floor']]
    if trace is not None:
        for item in representatives:
            if item.score < params['quality_floor']:
                trace[item.path] = {'decision': 'below_quality_floor', 'score': round(item.score, 3),
                                    'floor': params['quality_floor']}
    if len(strong) < min_representatives:
        return None
    factor, low, high = target_range
    target = min(target_size(len(strong), factor, low, high), len(strong))
    selected = diversify(strong, target, BUCKET_WEIGHTS[kind], params, trace,
                         minimum=min(low, len(strong)))
    if len(selected) < MIN_PHOTOS.get(kind, 1):
        if trace is not None:
            trace['__too_few__'] = {'selected': len(selected), 'minimum': MIN_PHOTOS.get(kind, 1)}
        return None
    selected_sorted = sorted(selected, key=lambda pair: (pair[0].ts, pair[0].path))
    scores = sorted((item.score for item, _ in selected), reverse=True)
    top = scores[:5]
    breadth = min(1.0, math.log(len(representatives) + 1) / math.log(100))
    group_score = (sum(top) / len(top)) * (0.75 + 0.25 * breadth)
    cover = max(selected, key=lambda pair: (pair[0].score, -pair[0].ts))[0]
    people = {}
    for item, _ in selected:
        for person in item.people:
            people[person] = people.get(person, 0) + 1
    photos = []
    for position, (item, info) in enumerate(selected_sorted):
        photos.append({'path': item.path, 'score': round(item.score, 4), 'pick': info['pick'],
                       'position': position,
                       'reasons': {**item.reasons(), **info, 'event': item.event,
                                   'series': len(item.series),
                                   'series_sample': item.series[:3],
                                   'people': item.people}})
    events = sorted({item.event for item in items})
    return {
        'key': key, 'kind': kind, 'title': title, 'subtitle': subtitle,
        'period_start': min(item.taken for item in items).isoformat(timespec='seconds'),
        'period_end': max(item.taken for item in items).isoformat(timespec='seconds'),
        'score': round(group_score, 4), 'cover_path': cover.path, 'photos': photos,
        'meta': {'candidates': len(items), 'representatives': len(representatives),
                 'strong': len(strong),
                 'collapsed': len(items) - len(representatives), 'target': target,
                 'events': len(events), 'people': people,
                 'time_sources': _count(item.source for item in items), **(extra_meta or {})},
    }


def _count(values):
    found = {}
    for value in values:
        found[value] = found.get(value, 0) + 1
    return found


# ---------------------------------------------------------------- виды подборок

def day_range_title(start, end):
    if start.date() == end.date():
        return f'{start.day} {MONTHS_GENITIVE[start.month - 1]} {start.year}'
    if (start.year, start.month) == (end.year, end.month):
        return f'{start.day}–{end.day} {MONTHS_GENITIVE[start.month - 1]} {start.year}'
    if start.year == end.year:
        return (f'{start.day} {MONTHS_GENITIVE[start.month - 1]} – '
                f'{end.day} {MONTHS_GENITIVE[end.month - 1]} {start.year}')
    return (f'{start.day} {MONTHS_GENITIVE[start.month - 1]} {start.year} – '
            f'{end.day} {MONTHS_GENITIVE[end.month - 1]} {end.year}')


def years_ago(count):
    if count % 10 == 1 and count % 100 != 11:
        word = 'год'
    elif count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        word = 'года'
    else:
        word = 'лет'
    return f'{count} {word} назад'


def month_groups(items, params, only=None, trace=None):
    by_month = {}
    for item in items:
        by_month.setdefault(item.taken.strftime('%Y-%m'), []).append(item)
    groups = []
    for month, members in sorted(by_month.items()):
        if only and month != only:
            continue
        year, number = int(month[:4]), int(month[5:])
        group = build_group('month', f'month:{month}', f'{MONTHS[number - 1].capitalize()} {year}',
                            'Лучшее за месяц', members, (2.2, 6, 30), params, trace,
                            min_representatives=6)
        if group:
            groups.append(group)
    return groups


def year_groups(items, params, only=None, trace=None):
    by_year = {}
    for item in items:
        by_year.setdefault(item.taken.year, []).append(item)
    groups = []
    for year, members in sorted(by_year.items()):
        if only and str(year) != str(only):
            continue
        group = build_group('year', f'year:{year}', f'{year} год', 'Лучшее за год',
                            members, (2.5, 12, 40), params, trace, min_representatives=12)
        if group:
            groups.append(group)
    return groups


def _majority(values, share, minimum):
    """Самое частое значение, если оно набирает долю share и не меньше minimum раз."""
    counts = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    best, count = max(counts.items(), key=lambda pair: pair[1])
    return best if count >= minimum and count >= share * len(values) else None


def home_place(items):
    """«Дом» — город, где снято больше всего кандидатов."""
    counts = {}
    for item in items:
        if item.place is not None:
            counts[item.place.id] = counts.get(item.place.id, 0) + 1
    if not counts:
        return None
    ident = max(counts, key=counts.get)
    return next(item.place for item in items if item.place is not None and item.place.id == ident)


def event_groups(events, params, only=None, trace=None, home=None):
    """События и поездки. Событие далеко от дома — поездка; главная тема — в названии."""
    groups = []
    for members in events:
        key = members[0].event
        start, end = members[0].taken, members[-1].taken
        located = [item.place for item in members if item.place is not None]
        place_id = _majority([place.id for place in located], 0.5, 2)
        place = next((item for item in located if item.id == place_id), None)
        theme = _majority([item.theme for item in members], params['theme_share'], 3)
        away = bool(place and home and places.distance_km(
            place.lat, place.lon, home.lat, home.lon) >= params['trip_km'])
        dates = day_range_title(start, end)
        if away:
            kind, group_key = 'trip', 'trip:' + key.split(':', 1)[1]
            title = f'{place.name} · {dates}'
            subtitle = place.country_name if place.country != home.country else 'Поездка'
        else:
            kind, group_key = 'event', key
            title = f'{highlight_themes.TITLES[theme]} · {dates}' if theme else dates
            subtitle = 'Событие'
        if only and only not in (key, group_key):
            continue
        extra = {'duration_hours': round((members[-1].ts - members[0].ts) / 3600, 2),
                 'theme': theme, 'place': place.name if place else None,
                 'place_id': place.id if place else None}
        group = build_group(kind, group_key, title, subtitle, members, (1.6, 4, 16), params, trace,
                            min_representatives=4, extra_meta=extra)
        if group and group['score'] >= params['min_event_score']:
            groups.append(group)
    return groups


def theme_groups(items, params, only=None, trace=None):
    """«Котики» за всё время и, если кадров много, по годам."""
    by_theme = {}
    for item in items:
        if item.theme:
            by_theme.setdefault(item.theme, []).append(item)
    groups = []
    for theme, members in sorted(by_theme.items()):
        title = highlight_themes.TITLES[theme]
        wanted = [(f'theme:{theme}', title, 'Лучшее за всё время', members, (2.2, 6, 30))]
        by_year = {}
        for item in members:
            by_year.setdefault(item.taken.year, []).append(item)
        for year, part in sorted(by_year.items()):
            if len(part) >= params['theme_year_min'] and len(by_year) > 1:
                wanted.append((f'theme:{theme}:{year}', f'{title} · {year}', f'Лучшее за {year} год',
                               part, (1.8, 6, 20)))
        for key, group_title, subtitle, part, size in wanted:
            if only and key != only:
                continue
            group = build_group('theme', key, group_title, subtitle, part, size, params, trace,
                                min_representatives=6, extra_meta={'theme': theme})
            if group:
                groups.append(group)
    return groups


def place_groups(items, params, only=None, trace=None, home=None):
    """Подборка на город, где снято достаточно."""
    by_place = {}
    for item in items:
        if item.place is not None:
            by_place.setdefault(item.place.id, []).append(item)
    groups = []
    for ident, members in by_place.items():
        if len(members) < params['place_min']:
            continue
        place = members[0].place
        key = f'place:{ident}'
        if only and key != only:
            continue
        is_home = bool(home and home.id == ident)
        group = build_group('place', key, place.name, 'Чаще всего снимали здесь' if is_home else place.country_name,
                            members, (2.0, 6, 30), params, trace, min_representatives=6,
                            extra_meta={'place_id': ident, 'country': place.country,
                                        'lat': place.lat, 'lon': place.lon, 'home': is_home})
        if group:
            groups.append(group)
    return groups


def on_this_day_groups(items, params, today, window_days=1, trace=None):
    """Снимки прошлых лет в окрестности сегодняшней даты — по подборке на год."""
    by_year = {}
    for item in items:
        if item.taken.year >= today.year:
            continue
        try:
            anniversary = item.taken.date().replace(year=today.year)
        except ValueError:   # 29 февраля
            anniversary = date(today.year, 2, 28)
        if abs((anniversary - today).days) <= window_days:
            by_year.setdefault(item.taken.year, []).append(item)
    groups = []
    for year, members in sorted(by_year.items(), reverse=True):
        title = years_ago(today.year - year)
        group = build_group('on-this-day', f'on-this-day:{today:%m-%d}:{year}', title,
                            f'В этот день · {today.day} {MONTHS_GENITIVE[today.month - 1]} {year}',
                            members, (1.5, 3, 10), params, trace, min_representatives=3,
                            extra_meta={'for_date': today.isoformat(), 'window_days': window_days})
        if group:
            groups.append(group)
    return groups


def generate(db, kinds=KINDS, today=None, params=None, stop=None, log=None):
    """Все подборки в памяти. Ничего не пишет. Возвращает (группы, сводка)."""
    params = {**PARAMS, **(params or {})}
    today = today or date.today()
    started = time.time()
    items, stats = load_candidates(db, params)
    events = detect_events(items, params)
    stats['events'] = len(events)
    if log:
        log(f"highlights: {stats['candidates']} candidates, {len(events)} events; filtered {stats}")
    groups = []
    home = home_place(items)
    stats['home'] = home.name if home else None
    eventish = []
    if 'event' in kinds or 'trip' in kinds:
        eventish = event_groups(events, params, home=home)
    builders = {
        'month': lambda: month_groups(items, params),
        'year': lambda: year_groups(items, params),
        'event': lambda: [group for group in eventish if group['kind'] == 'event'],
        'trip': lambda: [group for group in eventish if group['kind'] == 'trip'],
        'theme': lambda: theme_groups(items, params),
        'place': lambda: place_groups(items, params, home=home),
        'on-this-day': lambda: on_this_day_groups(items, params, today),
    }
    for kind in kinds:
        if stop and stop():
            return None, {**stats, 'stopped': True}
        built = builders[kind]()
        stats[kind] = len(built)
        groups.extend(built)
    stats['seconds'] = round(time.time() - started, 2)
    stats['params'] = params
    return groups, stats


def save(db, groups, kinds):
    """Атомарная замена подборок указанных видов.

    Всё в одной транзакции: обновление существующих по ключу (id остаётся
    прежним — ссылки из интерфейса не ломаются), новые фото, и только потом
    удаление подборок, которые больше не набрались. Прерванная пересборка
    оставляет старые подборки нетронутыми.
    """
    now = datetime.now(timezone.utc).isoformat()
    keys = [group['key'] for group in groups]
    with db:
        for group in groups:
            db.execute(
                'INSERT INTO highlight_groups(key,kind,title,subtitle,period_start,period_end,score,'
                'photo_count,cover_path,generated_at,meta_json) VALUES(?,?,?,?,?,?,?,?,?,?,?) '
                'ON CONFLICT(key) DO UPDATE SET kind=excluded.kind,title=excluded.title,'
                'subtitle=excluded.subtitle,period_start=excluded.period_start,'
                'period_end=excluded.period_end,score=excluded.score,'
                'photo_count=excluded.photo_count,cover_path=excluded.cover_path,'
                'generated_at=excluded.generated_at,meta_json=excluded.meta_json',
                (group['key'], group['kind'], group['title'], group['subtitle'],
                 group['period_start'], group['period_end'], group['score'],
                 len(group['photos']), group['cover_path'], now,
                 json.dumps(group['meta'], ensure_ascii=False)))
            group_id = db.execute('SELECT id FROM highlight_groups WHERE key=?',
                                  (group['key'],)).fetchone()[0]
            group['id'] = group_id
            db.execute('DELETE FROM highlight_photos WHERE group_id=?', (group_id,))
            db.executemany(
                'INSERT INTO highlight_photos(group_id,position,path,score,pick,reasons_json) '
                'VALUES(?,?,?,?,?,?)',
                [(group_id, photo['position'], photo['path'], photo['score'], photo['pick'],
                  json.dumps(photo['reasons'], ensure_ascii=False)) for photo in group['photos']])
        db.execute('CREATE TEMP TABLE IF NOT EXISTS highlight_keep(key TEXT PRIMARY KEY)')
        db.execute('DELETE FROM highlight_keep')
        db.executemany('INSERT OR IGNORE INTO highlight_keep(key) VALUES(?)', [(key,) for key in keys])
        marks = ','.join('?' * len(kinds))
        stale = f'kind IN ({marks}) AND key NOT IN (SELECT key FROM highlight_keep)'
        db.execute(f'DELETE FROM highlight_photos WHERE group_id IN '
                   f'(SELECT id FROM highlight_groups WHERE {stale})', list(kinds))
        removed = db.execute(f'DELETE FROM highlight_groups WHERE {stale}', list(kinds)).rowcount
        db.execute('DELETE FROM highlight_keep')
    return {'saved': len(groups), 'removed': removed}


def regenerate(catalog, kinds=KINDS, today=None, params=None, stop=None, log=None, db=None):
    """Собрать и сохранить. Своё соединение, если его не дали."""
    own = db is None
    db = db or connect(catalog)
    try:
        groups, stats = generate(db, kinds, today, params, stop, log)
        if groups is None:
            return stats
        if not stats['candidates']:
            # Ни одного кандидата — скорее не досчитаны оценки или проверка 18+,
            # чем действительно пустой каталог. Прежние подборки не стираем.
            stats.pop('params', None)
            return {**stats, 'saved': 0, 'removed': 0, 'kept_previous': True}
        stats.update(save(db, groups, kinds))
        stats.pop('params', None)
        return stats
    finally:
        if own:
            db.close()


# ---------------------------------------------------------------- чтение для API

GROUP_COLUMNS = ('id', 'key', 'kind', 'title', 'subtitle', 'period_start', 'period_end', 'score',
                 'photo_count', 'cover_path', 'generated_at', 'meta_json')


def _group_row(row):
    group = dict(zip(GROUP_COLUMNS, row))
    group['meta'] = json.loads(group.pop('meta_json') or '{}')
    return group


def list_groups(db, kind='', limit=50, offset=0, order='recent'):
    where, values = '', []
    if kind:
        where, values = ' WHERE kind=?', [kind]
    sort = 'score DESC, period_start DESC' if order == 'score' else 'period_start DESC, score DESC'
    total = db.execute(f'SELECT COUNT(*) FROM highlight_groups{where}', values).fetchone()[0]
    rows = db.execute(f'SELECT {",".join(GROUP_COLUMNS)} FROM highlight_groups{where} '
                      f'ORDER BY {sort}, id LIMIT ? OFFSET ?', [*values, limit, offset]).fetchall()
    return [_group_row(row) for row in rows], total


def get_group(db, ident):
    """По числовому id или по ключу вида month:2024-07."""
    column = 'id' if str(ident).isdigit() else 'key'
    row = db.execute(f'SELECT {",".join(GROUP_COLUMNS)} FROM highlight_groups WHERE {column}=?',
                     (int(ident) if column == 'id' else str(ident),)).fetchone()
    if row is None:
        return None
    group = _group_row(row)
    group['photos'] = [
        {'path': path, 'position': position, 'score': score, 'pick': pick,
         'reasons': json.loads(reasons or '{}')}
        for position, path, score, pick, reasons in db.execute(
            'SELECT position,path,score,pick,reasons_json FROM highlight_photos '
            'WHERE group_id=? ORDER BY position', (group['id'],))]
    return group


# ---------------------------------------------------------------- диагностика

def explain(db, path, params=None, today=None):
    """Почему снимок попал или не попал в подборки своего месяца, года и события."""
    params = {**PARAMS, **(params or {})}
    report = {'path': path, 'curation': photo_curation.explain_row(db, path)}
    if report['curation'] is None:
        report['verdict'] = 'нет оценки: снимок не прошёл визуальный анализ или этап curation не запускался'
        return report
    items, stats = load_candidates(db, params)
    by_path = {item.path: item for item in items}
    item = by_path.get(path)
    report['stored'] = [
        {'group': key, 'title': title, 'pick': pick, 'reasons': json.loads(reasons)}
        for key, title, pick, reasons in db.execute(
            'SELECT g.key,g.title,p.pick,p.reasons_json FROM highlight_photos p '
            'JOIN highlight_groups g ON g.id=p.group_id WHERE p.path=?', (path,))]
    if item is None:
        curation = report['curation']
        reasons = list(curation['rejections'] or [])
        if curation['eligible']:
            trust = photo_curation.TIME_TRUST.get(curation['taken_source'], 0.0)
            if trust < params['min_time_trust']:
                reasons.append(f"time_source:{curation['taken_source']} (доверие {trust} < {params['min_time_trust']})")
            if (curation['base_score'] or 0) < params['min_base_score']:
                reasons.append(f"base_score {curation['base_score']:.3f} < {params['min_base_score']}")
            if not reasons:
                reasons.append('скрыт, исключён правилами, устарела оценка или нет проверки 18+')
        report['verdict'] = 'не кандидат'
        report['reasons'] = reasons
        return report
    events = detect_events(items, params)
    members = next(group for group in events if group[0].event == item.event)
    report['event'] = {'key': item.event, 'size': len(members),
                       'start': members[0].taken.isoformat(), 'end': members[-1].taken.isoformat()}
    report['decisions'] = {}
    for label, builder in (
            ('month', lambda trace: month_groups(items, params, item.taken.strftime('%Y-%m'), trace)),
            ('year', lambda trace: year_groups(items, params, item.taken.year, trace)),
            ('event', lambda trace: event_groups(events, params, item.event, trace))):
        trace = {}
        built = builder(trace)
        decision = trace.get(path) or ({'decision': 'group_too_small'} if not built else
                                       {'decision': 'unknown'})
        if decision.get('decision') == 'lost_to_diversity' and '__stop__' in trace:
            decision['stopped'] = trace['__stop__']
        report['decisions'][label] = decision
    report['score'] = {'context': round(item.score, 4), **item.reasons()}
    return report


def contact_sheet(group, output, cell=220):
    """Картинка-лист подборки с оценками — глазами проверить результат."""
    from PIL import Image, ImageDraw, ImageOps
    photos = group['photos']
    columns = 6
    rows = max(1, math.ceil(len(photos) / columns))
    sheet = Image.new('RGB', (columns * cell, rows * (cell + 30) + 40), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), f"{group['title']} · {group['kind']} · {group['meta']}"[:180], fill='black')
    for index, photo in enumerate(photos):
        x, y = (index % columns) * cell, 40 + (index // columns) * (cell + 30)
        try:
            with Image.open(sources.local(photo['path'])) as source:
                source.draft('RGB', (cell * 2, cell * 2))
                image = ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((cell - 6, cell - 6))
            sheet.paste(image, (x + 3, y + 3))
        except Exception:
            draw.rectangle((x + 3, y + 3, x + cell - 3, y + cell - 3), outline='red')
        reasons = photo['reasons']
        draw.text((x + 4, y + cell), f"#{photo['pick']} s={photo['score']:.2f} v={reasons.get('visual') or 0:.2f} "
                                     f"t={reasons['technical']:.2f} x{reasons.get('series', 0) + 1}",
                  fill='black')
        draw.text((x + 4, y + cell + 13), pathkeys.name(photo['path'])[:34], fill='gray')
    sheet.save(output, quality=88)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('generate', help='пересобрать и сохранить подборки')
    run.add_argument('--catalog', type=Path, required=True)
    run.add_argument('--kinds', default=','.join(KINDS))
    run.add_argument('--today', help='дата для «в этот день», YYYY-MM-DD')
    run.add_argument('--dry-run', action='store_true', help='только показать, ничего не сохранять')
    run.add_argument('--allow-unchecked-adult', action='store_true', default=None,
                     help='брать снимки, которые ещё не прошли проверку 18+ '
                          '(по умолчанию — настройка highlights_require_adult_check)')
    run.add_argument('--force', action='store_true',
                     help='совместимость со стадиями: подборки и так строятся заново целиком')
    run.add_argument('--progress-file', type=Path)
    run.add_argument('--stop-file', type=Path)
    preview = sub.add_parser('preview', help='собрать одну подборку и нарисовать лист')
    preview.add_argument('--catalog', type=Path, required=True)
    preview.add_argument('--allow-unchecked-adult', action='store_true')
    preview.add_argument('--key', required=True, help='month:2019-12, year:2019, event:…')
    preview.add_argument('--sheet', type=Path)
    why = sub.add_parser('explain', help='почему снимок попал или не попал в подборки')
    why.add_argument('--catalog', type=Path, required=True)
    why.add_argument('--allow-unchecked-adult', action='store_true')
    why.add_argument('--path', required=True)
    listing = sub.add_parser('list', help='сохранённые подборки')
    listing.add_argument('--catalog', type=Path, required=True)
    listing.add_argument('--kind', default='')
    args = parser.parse_args()

    db = connect(args.catalog)
    try:
        if args.command == 'generate':
            kinds = [kind for kind in args.kinds.split(',') if kind]
            unknown = set(kinds) - set(KINDS)
            if unknown:
                parser.error(f'неизвестные виды: {", ".join(sorted(unknown))}')
            today = date.fromisoformat(args.today) if args.today else None
            progress = args.progress_file.resolve() if args.progress_file else None
            stop_path = args.stop_file.resolve() if args.stop_file else None
            photo_curation.write_progress(progress, status='running', total=1, completed=0)
            stop = (lambda: stop_path.exists()) if stop_path else None
            if args.allow_unchecked_adult is None:
                import settings as catalog_settings
                params = {'require_adult_check':
                          bool(catalog_settings.load(args.catalog)['highlights_require_adult_check'])}
            else:
                params = {'require_adult_check': False}
            if args.dry_run:
                groups, stats = generate(db, kinds, today, params, stop=stop, log=print)
                for group in groups or []:
                    print(f"{group['key']:<32} {group['title']:<28} score={group['score']:.3f} "
                          f"photos={len(group['photos'])} meta={group['meta']}")
                stats.pop('params', None)
            else:
                stats = regenerate(args.catalog, kinds, today, params, stop=stop, log=print, db=db)
            photo_curation.write_progress(progress, status='stopped' if stats.get('stopped') else 'completed',
                                          total=1, completed=1, **{k: v for k, v in stats.items()
                                                                   if k not in ('stopped',)})
            print(json.dumps(stats, ensure_ascii=False))
        elif args.command == 'preview':
            params = {**PARAMS, 'require_adult_check': not args.allow_unchecked_adult}
            items, _ = load_candidates(db, params)
            events = detect_events(items, params)
            kind, _, value = args.key.partition(':')
            builders = {'month': lambda: month_groups(items, params, value),
                        'year': lambda: year_groups(items, params, value),
                        'event': lambda: event_groups(events, params, args.key)}
            if kind not in builders:
                parser.error('preview умеет month, year и event')
            built = builders[kind]()
            if not built:
                print('подборка не набралась')
                return
            group = built[0]
            for photo in group['photos']:
                print(json.dumps({'path': photo['path'], 'score': photo['score'],
                                  **photo['reasons']}, ensure_ascii=False))
            print(json.dumps(group['meta'], ensure_ascii=False))
            if args.sheet:
                contact_sheet(group, args.sheet)
                print(f'лист: {args.sheet}')
        elif args.command == 'explain':
            print(json.dumps(explain(db, args.path, {'require_adult_check': not args.allow_unchecked_adult}),
                             ensure_ascii=False, indent=2, default=str))
        elif args.command == 'list':
            groups, total = list_groups(db, args.kind, limit=1000)
            for group in groups:
                print(f"{group['id']:>5} {group['key']:<32} {group['title']:<30} "
                      f"score={group['score']:.3f} photos={group['photo_count']}")
            print(f'всего: {total}')
    finally:
        db.close()


if __name__ == '__main__':
    sys.exit(main())
