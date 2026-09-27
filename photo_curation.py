"""Оценка снимков как кандидатов в автоматические подборки («лучшее за месяц» и т. п.).

Новых тяжёлых моделей здесь нет — только то, что каталог уже посчитал:

- **визуальная оценка** — готовый эмбеддинг SigLIP, сравненный с несколькими
  текстовыми описаниями «удачного» и «неудачного» кадра. Векторы описаний
  считаются один раз на модель (подкоманда `prompts`, окружение vision-venv)
  и лежат в `curation_prompts`; сама оценка — скалярные произведения;
- **техническая оценка** — резкость из `photo_analysis.blur_score`, разрешение,
  экспозиция и контраст по уменьшенной копии, сжатие (байт на пиксель);
- **личная значимость** — лица: насколько они крупные и узнан ли человек.
  Хранится отдельно и в базовую оценку НЕ входит: лицо в кадре — не признак
  хорошего снимка;
- время съёмки (EXIF → имя файла → mtime) и координаты из EXIF — для событий.

Этап инкрементальный в два слоя. Файл читается (EXIF, уменьшенная копия,
dHash, если его нет в `photo_hashes`) только когда поменялись размер или mtime.
Всё остальное — производные от других таблиц: подпись их состояния (`inputs`)
пересчитывается дёшево, и строка переоценивается, если лица переименовали или
анализ 18+ досчитался, без чтения файла.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

import numpy as np

import catalogdb
import pathkeys
import sources

# Поменять формулу оценки — поднять SCORE_VERSION (файлы не перечитываются).
# Поменять то, что достаётся из самого файла, — поднять FILE_VERSION.
SCORE_VERSION = 2
FILE_VERSION = 3

SCHEMA = '''
CREATE TABLE IF NOT EXISTS photo_curation (
  path TEXT PRIMARY KEY REFERENCES photos(path) ON DELETE CASCADE,
  size INTEGER NOT NULL, modified INTEGER NOT NULL,
  file_version INTEGER NOT NULL, score_version INTEGER NOT NULL,
  inputs TEXT NOT NULL,
  taken_at TEXT, taken_ts REAL, taken_source TEXT,
  latitude REAL, longitude REAL, camera TEXT,
  width INTEGER, height INTEGER, dhash TEXT,
  brightness REAL, contrast REAL, clipped REAL, colorfulness REAL,
  embedding_model TEXT, aesthetic_raw REAL,
  visual_score REAL, technical_score REAL, base_score REAL,
  face_count INTEGER NOT NULL DEFAULT 0, known_people INTEGER NOT NULL DEFAULT 0,
  personal_score REAL NOT NULL DEFAULT 0, people_json TEXT NOT NULL DEFAULT '[]',
  eligible INTEGER NOT NULL DEFAULT 0, rejections_json TEXT NOT NULL DEFAULT '[]',
  details_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL, error TEXT, computed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS photo_curation_taken ON photo_curation(eligible, taken_ts);
CREATE TABLE IF NOT EXISTS curation_prompts (
  model TEXT NOT NULL, name TEXT NOT NULL, weight REAL NOT NULL, prompt TEXT NOT NULL,
  embedding BLOB NOT NULL, dims INTEGER NOT NULL, computed_at TEXT NOT NULL,
  PRIMARY KEY(model, name)
);
'''

# Описания для zero-shot оценки. Вес со знаком: плюс — «такой кадр хочется
# показать», минус — «такой кадр в подборке не нужен».
PROMPTS = {
    'good': (1.0, 'a high quality, well-composed photograph'),
    'album': (1.0, 'a beautiful photo worth keeping in a family photo album'),
    'scenery': (0.5, 'a stunning landscape or travel photo'),
    'moment': (0.5, 'a candid photo of happy people enjoying a moment together'),
    'blurry': (-1.0, 'a blurry, out of focus, shaky photo'),
    'dark': (-0.5, 'a dark, underexposed, noisy photo'),
    'accidental': (-1.0, 'an accidental photo of a pocket, a floor or a ceiling'),
    'mundane': (-1.0, 'a photo of a document, a receipt, a price tag or a whiteboard'),
    'screen': (-1.0, 'a screenshot or a photo of a computer screen'),
}

# Калибровка «сырой» разницы похожестей в 0..1 логистой: (центр, масштаб).
# SigLIP2 на реальном каталоге: у снимков с типом photo медиана +0.009
# (5–95%: −0.022…+0.032), у скриншотов и документов −0.043.
# Jina сюда намеренно не входит: на 700 файлах её разница почти не отличала
# снимки с камерой от остального (AUC 0.63), и оценка вышла бы шумом с видом
# точного числа. Для таких снимков визуальной оценки просто нет.
AESTHETIC_CALIBRATION = {
    'google/siglip2-base-patch16-224': (0.003, 0.012),
    'google/siglip2-base-patch16-256': (0.003, 0.012),
}
# Модели, чей тип содержимого (photo/screenshot/…) годится как решение. У Jina
# в реальном каталоге 99% файлов оказались graphics или game, включая DCIM.
RELIABLE_CONTENT_MODELS = tuple(AESTHETIC_CALIBRATION)
# Визуальная оценка, которую получает кадр без откалиброванной модели.
VISUAL_PRIOR = 0.25
# Тип ошибается и у SigLIP: снимок из DCIM бывает «game». Если в EXIF записан
# производитель камеры (у скриншотов его нет), такой тип не приговор.
OVERRIDABLE_TYPES = ('game', 'graphics', 'meme')

# Доверие к источнику времени съёмки: событиям и «в этот день» оно важно.
TIME_TRUST = {'exif': 1.0, 'exif-datetime': 0.9, 'filename': 0.9, 'epoch-name': 0.8,
              'filename-date': 0.6, 'mtime': 0.3}

# Кто на снимке: названный человек весит больше, чем повторяющаяся безымянная
# гроздь, и тем более больше случайного лица из «шума».
IDENTITY_WEIGHT = {'person': 1.0, 'cluster': 0.6, 'noise': 0.2}
SAFE_RATINGS = ('safe', 'unknown')


def connect(catalog):
    db = catalogdb.connect(catalog, timeout=60)
    ensure_schema(db)
    return db


def ensure_schema(db):
    db.executescript(SCHEMA)
    if 'camera' not in {row[1] for row in db.execute('PRAGMA table_info(photo_curation)')}:
        # Столбец появился позже: старые строки перечитаются по FILE_VERSION.
        db.execute('ALTER TABLE photo_curation ADD COLUMN camera TEXT')
    db.commit()


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def ramp(value, start, stop):
    """Линейно 0 → 1 между start и stop."""
    if value is None:
        return None
    return clamp((value - start) / (stop - start))


def write_progress(path, **payload):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload['updated_at'] = time.time()
    payload['pid'] = os.getpid()
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    # Прогресс читают параллельно — редкий WinError 5 на os.replace не должен
    # ронять этап.
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


# ---------------------------------------------------------------- время съёмки

EPOCH = datetime(1970, 1, 1)
_DATE = r'((?:19|20)\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])'
_TIME = r'([01]\d|2[0-3])[-_.:]?([0-5]\d)[-_.:]?([0-5]\d)'
FILENAME_DATETIME = re.compile(
    r'(?<!\d)' + _DATE + r'(?:[ _T-]|\s+(?:at|в)\s+|_?-?)' + _TIME + r'(?:\d{1,3})?(?!\d)',
    re.IGNORECASE)
FILENAME_DATE = re.compile(r'(?<!\d)' + _DATE + r'(?!\d)')
EPOCH_MS = re.compile(r'(?<!\d)(1[0-9]{12})(?!\d)')


def wall_ts(moment):
    """Секунды «настенного» времени: наивное местное время как будто в UTC.

    Для событий важны только разности, а у EXIF и имён файлов часового пояса
    нет — так все источники остаются в одной шкале.
    """
    return (moment - EPOCH).total_seconds()


def plausible(moment, now=None):
    now = now or datetime.now()
    return datetime(1990, 1, 1) <= moment <= now + timedelta(days=2)


def parse_exif_datetime(raw):
    if not raw:
        return None
    text = str(raw).strip().replace('\x00', '')[:19]
    for pattern in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y:%m:%d %H:%M', '%Y-%m-%dT%H:%M:%S'):
        try:
            moment = datetime.strptime(text, pattern)
            return moment if plausible(moment) else None
        except ValueError:
            continue
    return None


def time_from_name(path):
    """Дата и время из имени: IMG_20191218_131104, WhatsApp Image 2021-12-27 at
    18.16.46, «Изображение WhatsApp 2024-02-21 в 17.56.22», 1573130586733.jpg.

    Если в имени несколько дат (IMG-20180525-WA0008_1572848274767 — дата
    WhatsApp и метка пересохранения), берётся самая ранняя: снимок не может
    быть сделан позже, чем его сохранили.
    """
    name = Path(path).stem
    found = []
    for match in FILENAME_DATETIME.finditer(name):
        try:
            moment = datetime(*(int(part) for part in match.groups()))
        except ValueError:
            continue
        if plausible(moment):
            found.append((moment, 'filename'))
    for match in EPOCH_MS.finditer(name):
        moment = datetime.fromtimestamp(int(match.group(1)) / 1000)
        if datetime(2005, 1, 1) <= moment and plausible(moment):
            found.append((moment, 'epoch-name'))
    for match in FILENAME_DATE.finditer(name):
        try:
            moment = datetime(*(int(part) for part in match.groups()), 12, 0, 0)
        except ValueError:
            continue
        if plausible(moment) and not any(other.date() == moment.date() for other, _ in found):
            found.append((moment, 'filename-date'))
    if not found:
        return None, None
    return min(found, key=lambda item: item[0])


def taken_time(path, exif_moment, exif_source, modified_ns):
    """Лучшее, что известно о моменте съёмки, и откуда это известно."""
    if exif_moment:
        return exif_moment, exif_source
    moment, source = time_from_name(path)
    if moment:
        if source == 'filename-date' and modified_ns:
            # Дата из имени без времени: если mtime в тот же день, берём его время.
            mtime = datetime.fromtimestamp(modified_ns / 1e9)
            if mtime.date() == moment.date():
                return mtime, 'filename'
        return moment, source
    if modified_ns:
        return datetime.fromtimestamp(modified_ns / 1e9), 'mtime'
    return None, None


def _rational(value):
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        try:
            return value[0] / value[1]
        except Exception:
            return None


def gps_from_exif(gps):
    if not gps:
        return None, None

    def degrees(parts, ref):
        try:
            d, m, s = (_rational(item) for item in parts)
            value = d + m / 60 + s / 3600
        except Exception:
            return None
        return -value if str(ref).upper().startswith(('S', 'W')) else value
    latitude = degrees(gps.get(2), gps.get(1, 'N')) if gps.get(2) else None
    longitude = degrees(gps.get(4), gps.get(3, 'E')) if gps.get(4) else None
    if latitude is None or longitude is None or (abs(latitude) < 1e-6 and abs(longitude) < 1e-6):
        return None, None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None, None
    return latitude, longitude


# ---------------------------------------------------------------- чтение файла

def read_file(path, target, modified_ns, need_dhash):
    """Всё, что нужно достать из самого файла, за одно-два открытия."""
    from PIL import Image, ImageOps
    result = {}
    with Image.open(target) as source:
        exif_moment = exif_source = None
        try:
            exif = source.getexif()
            sub = exif.get_ifd(0x8769)
            exif_moment = parse_exif_datetime(sub.get(36867) or sub.get(36868))
            exif_source = 'exif' if exif_moment else None
            if not exif_moment:
                exif_moment = parse_exif_datetime(exif.get(306))
                exif_source = 'exif-datetime' if exif_moment else None
            result['latitude'], result['longitude'] = gps_from_exif(exif.get_ifd(0x8825))
            camera = ' '.join(str(exif.get(tag) or '').strip('\x00 ') for tag in (271, 272)).strip()
            result['camera'] = camera[:120] or None
        except Exception:
            result['latitude'] = result['longitude'] = result['camera'] = None
        source.draft('RGB', (320, 320))
        image = ImageOps.exif_transpose(source)
        image = image.convert('RGB')
    image.thumbnail((256, 256))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    rg, yb = red - green, 0.5 * (red + green) - blue
    result.update(
        brightness=float(luma.mean()), contrast=float(luma.std()),
        clipped=float(((luma < 0.02) | (luma > 0.98)).mean()),
        # Цветность по Hasler–Süsstrunk, в долях от 0..1.
        colorfulness=float(math.hypot(rg.std(), yb.std()) +
                           0.3 * math.hypot(rg.mean(), yb.mean())))
    moment, source_name = taken_time(path, exif_moment, exif_source, modified_ns)
    result['taken_at'] = moment.isoformat(timespec='seconds') if moment else None
    result['taken_ts'] = wall_ts(moment) if moment else None
    result['taken_source'] = source_name
    result['dhash'] = None
    if need_dhash:
        import duplicates
        result['dhash'] = duplicates.picture_hash(target)[0]
    return result


# ---------------------------------------------------------------- оценка

def prompt_vectors(db, model):
    rows = db.execute('SELECT name,weight,prompt,embedding FROM curation_prompts WHERE model=?',
                      (model,)).fetchall()
    known = {name: (weight, prompt, np.frombuffer(blob, dtype='<f4'))
             for name, weight, prompt, blob in rows}
    if any(name not in known or known[name][1] != text for name, (_, text) in PROMPTS.items()):
        return None
    return {name: (PROMPTS[name][0], known[name][2]) for name in PROMPTS}


def aesthetic_raw(vector, prompts):
    """Средняя похожесть на «хорошие» описания минус на «плохие», с весами."""
    vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
    positive = negative = positive_weight = negative_weight = 0.0
    for weight, text_vector in prompts.values():
        similarity = float(vector @ text_vector)
        if weight > 0:
            positive += weight * similarity
            positive_weight += weight
        else:
            negative += -weight * similarity
            negative_weight += -weight
    return positive / max(positive_weight, 1e-9) - negative / max(negative_weight, 1e-9)


def face_entries(faces, width, height):
    """Лица снимка → значимость каждого с учётом размера и того, кто это."""
    entries = []
    area = float((width or 0) * (height or 0))
    for face in faces:
        if face['excluded'] or face['drawn']:
            continue
        box = face['box']
        ratio = 0.0
        if box and area > 0:
            ratio = math.sqrt(max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / area)
        # Лицо размером в 3% стороны кадра — фон; от 20% — уже портрет.
        prominence = ramp(ratio, 0.03, 0.20)
        if face['person_id'] is not None:
            identity = 'person'
        elif face['label'] is not None and face['label'] >= 0:
            identity = 'cluster'
        else:
            identity = 'noise'
        entries.append({'face_id': face['id'], 'person_id': face['person_id'],
                        'label': face['label'], 'identity': identity,
                        'ratio': round(ratio, 4), 'prominence': round(prominence, 3),
                        'weight': round(prominence * IDENTITY_WEIGHT[identity], 3)})
    return entries


def score(row, files, prompts_by_model, embedding):
    """Производные оценки строки. Ничего не читает с диска.

    row — входы из каталога, files — поля, достанные из файла, embedding —
    (модель, вектор) или None. Возвращает словарь колонок photo_curation.
    """
    width = row['width'] or files.get('width')
    height = row['height'] or files.get('height')
    megapixels = (width or 0) * (height or 0) / 1e6
    rejections = []

    # --- техника
    blur = row['blur_score']
    sharp = ramp(math.log10(max(blur, 1e-3)), math.log10(25), math.log10(600)) if blur is not None else None
    resolution = ramp(math.log2(megapixels / 0.2), 0.0, math.log2(4 / 0.2)) if megapixels > 0 else 0.0
    brightness = files.get('brightness')
    exposure = None
    if brightness is not None:
        exposure = 1 - ramp(abs(brightness - 0.45), 0.15, 0.45)
        exposure *= 1 - 0.7 * ramp(files.get('clipped') or 0, 0.05, 0.35)
    contrast = ramp(files.get('contrast'), 0.05, 0.20)
    compression = ramp(row['size'] / max(megapixels * 1e6, 1), 0.03, 0.15) if megapixels > 0 else 0.0
    parts = [(0.40, sharp), (0.20, resolution), (0.20, exposure), (0.10, contrast), (0.10, compression)]
    known = [(weight, value) for weight, value in parts if value is not None]
    technical = sum(w * v for w, v in known) / sum(w for w, _ in known) if known else 0.0

    # --- визуальная оценка по эмбеддингу
    raw = visual = None
    model = None
    if embedding is not None:
        model, vector = embedding
        prompts = prompts_by_model.get(model)
        if prompts:
            raw = aesthetic_raw(vector, prompts)
            if model in AESTHETIC_CALIBRATION:
                center, scale = AESTHETIC_CALIBRATION[model]
                visual = 1 / (1 + math.exp(-(raw - center) / scale))
    # Без визуальной оценки нельзя считать кадр средним: технически безупречное
    # фото конспекта или экрана обгоняло бы настоящие снимки. Такой кадр идёт в
    # подборку, только когда лучше ничего нет.
    base = 0.55 * (VISUAL_PRIOR if visual is None else visual) + 0.45 * technical

    # --- лица и люди, отдельно от «хорошести»
    entries = face_entries(row['faces'], width, height)
    personal = 1.0
    for entry in entries:
        personal *= 1 - 0.85 * entry['weight']
    personal = 1 - personal
    visible = [entry for entry in entries if entry['prominence'] > 0]
    known_people = sorted({entry['person_id'] for entry in visible if entry['person_id'] is not None})

    # --- кто вообще может попасть в подборку
    camera = files.get('camera')
    reliable = row.get('analysis_model') in RELIABLE_CONTENT_MODELS
    override = False
    if reliable:
        override = bool(camera) and row['content_type'] in OVERRIDABLE_TYPES
        if row['content_type'] != 'photo' and not override:
            rejections.append(f"content:{row['content_type'] or 'unknown'}")
    elif not camera:
        # Типу этой модели верить нельзя, а камеры в EXIF нет — скриншот,
        # скачанная картинка или пересланное фото, отличить нечем.
        rejections.append('content:unverified')
    if row['rating'] not in SAFE_RATINGS and row['rating'] is not None:
        rejections.append(f"adult:{row['rating']}")
    if not width or not height or min(width, height) < 480 or megapixels < 0.2:
        rejections.append('too_small')
    if blur is not None and blur < 15:
        rejections.append('blurry')
    if width and height and max(width, height) / max(min(width, height), 1) > 4:
        rejections.append('odd_aspect')
    if brightness is not None and not 0.05 <= brightness <= 0.96:
        rejections.append('exposure')
    if not files.get('taken_ts'):
        rejections.append('no_time')

    details = {
        'technical': {'sharpness': sharp, 'resolution': round(resolution, 3),
                      'exposure': exposure, 'contrast': contrast,
                      'compression': round(compression, 3), 'blur_score': blur,
                      'megapixels': round(megapixels, 2)},
        'visual': {'raw': raw, 'model': model, 'prompts': bool(visual is not None)},
        'content': {'type': row['content_type'], 'confidence': row['content_confidence'],
                    'adult_rating': row['rating'], 'override': override,
                    'model': row.get('analysis_model'), 'reliable': reliable, 'camera': camera},
        'time_trust': TIME_TRUST.get(files.get('taken_source'), 0.0),
    }
    rounded = {key: (round(value, 4) if isinstance(value, float) else value)
               for key, value in details['technical'].items()}
    details['technical'] = rounded
    return {
        'width': width, 'height': height, 'embedding_model': model,
        'aesthetic_raw': raw, 'visual_score': visual, 'technical_score': technical,
        'base_score': base, 'face_count': len(visible), 'known_people': len(known_people),
        'personal_score': personal,
        'people_json': json.dumps(entries, ensure_ascii=False),
        'eligible': int(not rejections),
        'rejections_json': json.dumps(rejections),
        'details_json': json.dumps(details, ensure_ascii=False),
    }


# ---------------------------------------------------------------- входы из каталога

def _table_exists(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                      (name,)).fetchone() is not None


def load_faces(db, paths=None):
    """Лица снимков (не роликов) с людьми, группами и отметками исключения."""
    if not _table_exists(db, 'faces'):
        return {}
    joins, columns = [], ['faces.id', 'faces.path', 'faces.box']
    for table, column, join in (
            ('face_clusters', 'face_clusters.label', 'LEFT JOIN face_clusters ON face_clusters.face_id=faces.id'),
            ('face_people', 'face_people.person_id', 'LEFT JOIN face_people ON face_people.face_id=faces.id'),
            ('face_exclusions', 'face_exclusions.face_id', 'LEFT JOIN face_exclusions ON face_exclusions.face_id=faces.id'),
            ('face_authenticity', 'face_authenticity.real_score,face_authenticity.anime_score',
             'LEFT JOIN face_authenticity ON face_authenticity.face_id=faces.id')):
        if _table_exists(db, table):
            joins.append(join)
            columns.append(column)
        else:
            columns.append('NULL,NULL' if table == 'face_authenticity' else 'NULL')
    faces_columns = {row[1] for row in db.execute('PRAGMA table_info(faces)')}
    video_filter = ' WHERE faces.frame_time IS NULL' if 'frame_time' in faces_columns else ''
    found = {}
    for face_id, path, raw_box, label, person_id, excluded, real, anime in db.execute(
            f'SELECT {",".join(columns)} FROM faces {" ".join(joins)}{video_filter} ORDER BY faces.id'):
        if paths is not None and path not in paths:
            continue
        try:
            box = [float(value) for value in json.loads(raw_box)][:4]
        except (TypeError, ValueError):
            box = None
        found.setdefault(path, []).append({
            'id': face_id, 'box': box, 'label': label, 'person_id': person_id,
            'excluded': excluded is not None,
            'drawn': real is not None and anime is not None and anime > real})
    return found


def signature(row, prompts_stamp):
    faces = [(face['id'], face['label'], face['person_id'], face['excluded'], face['drawn'], face['box'])
             for face in row['faces']]
    raw = json.dumps([SCORE_VERSION, row['analyzed_at'], row['content_type'], row['blur_score'],
                      row['width'], row['height'], row['embeddings'], row['rating'],
                      row['adult_at'], row['hash_dhash'], faces, prompts_stamp,
                      row.get('analysis_model')],
                     default=str, sort_keys=True)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]


def candidates(db, roots=(), paths=(), limit=None):
    """Снимки с готовым визуальным анализом — только у них есть тип и эмбеддинг."""
    # Копии файлов не оцениваем: оценку им переносит filekeys propagate.
    scope = pathkeys.analysis_scope_sql if _table_exists(db, 'photo_copies') else pathkeys.scope_sql
    where, values = scope(roots, paths, 'photos.path')
    adult = _table_exists(db, 'photo_adult_analysis')
    hashes = _table_exists(db, 'photo_hashes')
    sql = f'''
        SELECT photos.path, photos.size, photos.modified,
               photo_analysis.analyzed_at, photo_analysis.content_type,
               photo_analysis.content_confidence, photo_analysis.blur_score,
               photo_analysis.width, photo_analysis.height, photo_analysis.embedding_model,
               {"photo_adult_analysis.rating, photo_adult_analysis.analyzed_at" if adult else "NULL, NULL"},
               {"photo_hashes.dhash, photo_hashes.size, photo_hashes.modified" if hashes else "NULL, NULL, NULL"}
        FROM photos JOIN photo_analysis USING(path)
        {"LEFT JOIN photo_adult_analysis ON photo_adult_analysis.path=photos.path AND photo_adult_analysis.status='ok'" if adult else ""}
        {"LEFT JOIN photo_hashes ON photo_hashes.path=photos.path" if hashes else ""}
        WHERE photos.status='ok' AND COALESCE(photos.kind,'photo')='photo'
          AND COALESCE(photos.blocked,0)=0 AND photo_analysis.status='ok'{where}
        ORDER BY photos.path'''
    if limit:
        sql += ' LIMIT ?'
        values.append(int(limit))
    import video as video_media
    rows = []
    for (path, size, modified, analyzed_at, content_type, confidence, blur, width, height,
         analysis_model, rating, adult_at, dhash, hash_size, hash_modified) in db.execute(sql, values):
        if video_media.is_video(path):
            continue
        fresh_hash = dhash if (hash_size == size and hash_modified == modified) else None
        rows.append({'path': path, 'size': size, 'modified': modified,
                     'analyzed_at': analyzed_at, 'content_type': content_type,
                     'content_confidence': confidence, 'blur_score': blur,
                     'analysis_model': analysis_model,
                     'width': width, 'height': height, 'rating': rating, 'adult_at': adult_at,
                     'hash_dhash': fresh_hash, 'faces': [], 'embeddings': []})
    return rows


def embeddings_for(db, paths):
    """{путь: {модель: вектор}} — порциями, чтобы не тянуть весь индекс сразу."""
    found = {}
    paths = list(paths)
    for offset in range(0, len(paths), 500):
        batch = paths[offset:offset + 500]
        marks = ','.join('?' * len(batch))
        for path, model, blob, analyzed_at in db.execute(
                f'SELECT path,model,embedding,analyzed_at FROM photo_embeddings '
                f'WHERE path IN ({marks})', batch):
            found.setdefault(path, {})[model] = (np.frombuffer(blob, dtype='<f4'), analyzed_at)
    return found


def pick_embedding(available, preferred, prompts_by_model):
    """Сначала откалиброванная модель (лучше из настроек), потом любая с описаниями."""
    if not available:
        return None
    calibrated = [model for model in available
                  if model in AESTHETIC_CALIBRATION and model in prompts_by_model]
    if calibrated:
        model = preferred if preferred in calibrated else sorted(calibrated)[0]
        return model, available[model][0]
    if preferred in available:
        return preferred, available[preferred][0]
    for model in sorted(available):
        if model in prompts_by_model:
            return model, available[model][0]
    model = sorted(available)[0]
    return model, available[model][0]


# ---------------------------------------------------------------- этап

COLUMNS = ('path', 'size', 'modified', 'file_version', 'score_version', 'inputs',
           'taken_at', 'taken_ts', 'taken_source', 'latitude', 'longitude', 'camera',
           'width', 'height', 'dhash', 'brightness', 'contrast', 'clipped', 'colorfulness',
           'embedding_model', 'aesthetic_raw', 'visual_score', 'technical_score', 'base_score',
           'face_count', 'known_people', 'personal_score', 'people_json',
           'eligible', 'rejections_json', 'details_json', 'status', 'error', 'computed_at')
FILE_FIELDS = ('taken_at', 'taken_ts', 'taken_source', 'latitude', 'longitude', 'camera', 'dhash',
               'brightness', 'contrast', 'clipped', 'colorfulness')


def _save(db, batch):
    if not batch:
        return
    marks = ','.join('?' * len(COLUMNS))
    updates = ','.join(f'{name}=excluded.{name}' for name in COLUMNS[1:])
    with db:
        db.executemany(f'INSERT INTO photo_curation({",".join(COLUMNS)}) VALUES({marks}) '
                       f'ON CONFLICT(path) DO UPDATE SET {updates}',
                       [tuple(item.get(name) for name in COLUMNS) for item in batch])


def curate(catalog, roots=(), paths=(), force=False, limit=None, progress=None, stop=None,
           workers=6, log=print):
    """Инкрементальная оценка. Возвращает сводку счётчиков."""
    import privacy
    import settings as catalog_settings
    catalog = Path(catalog).resolve()
    db = connect(catalog)
    try:
        preferred = catalog_settings.load(catalog)['visual_model']
        rows = candidates(db, roots, paths, limit)
        by_path = {row['path']: row for row in rows}
        faces = load_faces(db, set(by_path))
        models = {model for (model,) in db.execute('SELECT DISTINCT model FROM photo_embeddings')}
        prompts_by_model = {model: vectors for model in models
                            if (vectors := prompt_vectors(db, model)) is not None}
        prompts_stamp = {model: db.execute('SELECT MAX(computed_at) FROM curation_prompts WHERE model=? '
                                           "AND name NOT LIKE 'theme:%'",
                                           (model,)).fetchone()[0] for model in prompts_by_model}
        vectors = embeddings_for(db, by_path)
        existing = {row[0]: dict(zip(COLUMNS, row)) for row in db.execute(
            f'SELECT {",".join(COLUMNS)} FROM photo_curation')}
        disk = privacy.stored_paths(db)

        work = []
        for row in rows:
            row['faces'] = faces.get(row['path'], [])
            available = vectors.get(row['path'], {})
            row['embeddings'] = sorted((model, stamp) for model, (_, stamp) in available.items())
            embedding = pick_embedding(available, preferred, prompts_by_model)
            inputs = signature(row, prompts_stamp.get(embedding[0]) if embedding else None)
            old = existing.get(row['path'])
            read = (force or old is None or old['size'] != row['size']
                    or old['modified'] != row['modified'] or old['file_version'] != FILE_VERSION
                    or old['status'] != 'ok')
            if not read and old['inputs'] == inputs and old['score_version'] == SCORE_VERSION:
                continue
            work.append((row, embedding, inputs, old, read))

        state = {'total': len(work), 'completed': 0, 'read': sum(item[4] for item in work),
                 'errors': 0, 'eligible': 0, 'skipped': len(rows) - len(work),
                 'prompts': sorted(prompts_by_model)}
        if progress:
            progress(status='running', current='', **state)
        log(f"curation: {len(rows)} candidates, {len(work)} to update "
            f"({state['read']} need file read); prompts for {sorted(prompts_by_model) or 'no model'}")

        def load(item):
            row, _, _, old, read = item
            if not read:
                return {name: old[name] for name in FILE_FIELDS}, None
            try:
                target = sources.local(disk.get(row['path'], row['path']))
                return read_file(row['path'], target, row['modified'],
                                 need_dhash=row['hash_dhash'] is None), None
            except Exception as exc:
                return None, str(exc)

        now = datetime.now(timezone.utc).isoformat()
        batch = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for offset in range(0, len(work), 64):
                if stop and stop():
                    break
                chunk = work[offset:offset + 64]
                for (row, embedding, inputs, old, read), (files, error) in zip(
                        chunk, pool.map(load, chunk)):
                    record = {'path': row['path'], 'size': row['size'], 'modified': row['modified'],
                              'file_version': FILE_VERSION, 'score_version': SCORE_VERSION,
                              'inputs': inputs, 'computed_at': now}
                    if files is None:
                        record.update(status='error', error=error, eligible=0,
                                      rejections_json='["read_error"]')
                        state['errors'] += 1
                    else:
                        files['dhash'] = row['hash_dhash'] or files.get('dhash')
                        record.update(files)
                        record.update(score(row, files, prompts_by_model, embedding))
                        record.update(status='ok', error=None)
                        state['eligible'] += record['eligible']
                    batch.append(record)
                    state['completed'] += 1
                if len(batch) >= 256:
                    _save(db, batch)
                    batch = []
                if progress:
                    progress(status='running', current=chunk[-1][0]['path'], **state)
        _save(db, batch)
        state['stopped'] = bool(stop and stop())
        return state
    finally:
        db.close()


def compute_prompts(catalog, model, force=False):
    """Векторы текстовых описаний для модели: оценки кадра и тем подборок. Запускается в vision-venv."""
    import highlight_themes
    catalog = Path(catalog).resolve()
    db = connect(catalog)
    try:
        wanted = []
        if force or prompt_vectors(db, model) is None:
            wanted += [(name, weight, text) for name, (weight, text) in PROMPTS.items()]
        if force or not highlight_themes.prompts_up_to_date(db, model):
            wanted += [(name, 0.0, text) for name, text in highlight_themes.prompt_rows()]
        if not wanted:
            print(f'prompts for {model} are up to date')
            return False
        from analyze_photos import VisualEncoder
        encoder = VisualEncoder(model)
        vectors = encoder.texts([text for _, _, text in wanted]).float().cpu().numpy()
        now = datetime.now(timezone.utc).isoformat()
        with db:
            db.executemany(
                'INSERT INTO curation_prompts(model,name,weight,prompt,embedding,dims,computed_at) '
                'VALUES(?,?,?,?,?,?,?) ON CONFLICT(model,name) DO UPDATE SET weight=excluded.weight,'
                'prompt=excluded.prompt,embedding=excluded.embedding,dims=excluded.dims,'
                'computed_at=excluded.computed_at',
                [(model, name, weight, text, np.asarray(vector, dtype='<f4').tobytes(), len(vector), now)
                 for (name, weight, text), vector in zip(wanted, vectors)])
        print(f'prompts for {model}: {len(wanted)} stored')
        return True
    finally:
        db.close()


def explain_row(db, path):
    """Всё, что этап знает о снимке, в читаемом виде; None — оценки нет."""
    row = db.execute(f'SELECT {",".join(COLUMNS)} FROM photo_curation WHERE path=?',
                     (path,)).fetchone()
    if row is None:
        return None
    record = dict(zip(COLUMNS, row))
    for name in ('people_json', 'rejections_json', 'details_json'):
        record[name[:-5]] = json.loads(record.pop(name) or 'null')
    return record


def explain(catalog, path):
    db = connect(catalog)
    try:
        return explain_row(db, path)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('curate', help='оценить снимки (окружение .venv)')
    run.add_argument('--catalog', type=Path, required=True)
    run.add_argument('--root', action='append', type=str, default=[])
    run.add_argument('--path', action='append', type=str, default=[])
    run.add_argument('--limit', type=int, default=0)
    run.add_argument('--workers', type=int, default=6)
    run.add_argument('--force', action='store_true',
                     help='Перечитать файлы и переоценить даже неизменившиеся снимки')
    run.add_argument('--progress-file', type=Path)
    run.add_argument('--stop-file', type=Path)
    prompts = sub.add_parser('prompts', help='векторы описаний (окружение vision-venv)')
    prompts.add_argument('--catalog', type=Path, required=True)
    prompts.add_argument('--model', action='append', default=[])
    prompts.add_argument('--indexed', action='store_true',
                         help='и для всех моделей, чьи векторы уже лежат в photo_embeddings')
    prompts.add_argument('--force', action='store_true')
    show = sub.add_parser('explain', help='показать оценку одного снимка')
    show.add_argument('--catalog', type=Path, required=True)
    show.add_argument('--path', required=True)
    args = parser.parse_args()

    if args.command == 'prompts':
        models = list(args.model)
        if args.indexed:
            db = connect(args.catalog)
            try:
                models += [model for (model,) in db.execute(
                    'SELECT DISTINCT model FROM photo_embeddings ORDER BY model')]
            finally:
                db.close()
        failed = 0
        for model in dict.fromkeys(models):
            try:
                compute_prompts(args.catalog, model, args.force)
            except Exception as exc:
                # Модель могли удалить с диска: без её описаний визуальная
                # оценка просто не считается, этап оценки это переживёт.
                failed += 1
                print(f'prompts for {model} failed: {exc}', file=sys.stderr)
        return 1 if failed and failed == len(set(models)) else 0
    if args.command == 'explain':
        record = explain(args.catalog, args.path)
        print(json.dumps(record, ensure_ascii=False, indent=2) if record else 'нет оценки для этого пути')
        return
    progress_path = args.progress_file.resolve() if args.progress_file else None
    stop_path = args.stop_file.resolve() if args.stop_file else None
    write_progress(progress_path, status='preparing', total=0, completed=0)
    try:
        state = curate(args.catalog, args.root, args.path, args.force, args.limit or None,
                       progress=lambda **values: write_progress(progress_path, **values),
                       stop=(lambda: stop_path.exists()) if stop_path else None,
                       workers=args.workers)
    except Exception as exc:
        write_progress(progress_path, status='error', error=str(exc), total=0, completed=0)
        raise
    write_progress(progress_path, status='stopped' if state.get('stopped') else 'completed',
                   current='', **{key: value for key, value in state.items() if key != 'stopped'})
    print(json.dumps(state, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
