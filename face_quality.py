"""Резкость лиц: какие кадры слишком мыльные, чтобы по ним узнавать человека.

Мыльное лицо — это не только размытый снимок. Чаще это крошечное лицо на
заднем плане (20–30 точек), тёмный кадр из ролика или смазанный поворот
головы. ArcFace выдаёт для них вектор, который мало похож на самого человека и
при этом «похож на всех понемногу» — такие лица склеивают разные группы между
собой и забивают карточку человека кашей.

Оценка считается по уже сохранённой миниатюре лица, а не по оригиналу:
перечитывать тысячи снимков и роликов ради неё не нужно. Миниатюра приводится
к одному размеру 96×96 — крошечное лицо при этом растягивается и честно
выглядит мыльным, — и на ней меряется метрика Crété-Roffet (2007): кадр
размывается ещё раз, и смотрится, насколько упали перепады между соседними
точками. Резкий кадр от повторного размытия теряет много, мыльный — почти
ничего. Шкала от 0 (резко) до 1 (мыло); от контраста и освещения она зависит
намного меньше, чем дисперсия лапласиана.

Подбор на реальном каталоге (5713 лиц, просмотр выборок глазами):
дисперсия лапласиана путала гладкую кожу и тёмные кадры с мылом, а Crété
отделяет чисто: от 0.80 — сплошь мыло, 0.76–0.80 — почти всё мыло,
0.70–0.76 — вперемешку, ниже 0.70 — нормальные лица. Поэтому порог 0.76;
из 915 подписанных вручную лиц за него уходит 13.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import catalogfiles
import face_crops

# Поменялся расчёт — подними версию, и оценки пересчитаются по миниатюрам.
VERSION = 1
BLUR_THRESHOLD = 0.76
MIN_SIZE = 20
SAMPLE = 96


def ensure_schema(db):
    db.execute('''
        CREATE TABLE IF NOT EXISTS face_quality (
          face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
          blur REAL, size REAL,
          keep INTEGER NOT NULL DEFAULT 0,
          version INTEGER NOT NULL, computed_at TEXT NOT NULL)''')
    columns = {row[1] for row in db.execute('PRAGMA table_info(face_quality)')}
    for name in ('confidence', 'geometry'):
        if name not in columns:
            db.execute(f'ALTER TABLE face_quality ADD COLUMN {name} REAL')
    db.commit()


def blur_score(gray):
    """Crété-Roffet: 0 — резко, 1 — мыло. `gray` — двумерный массив яркости."""
    import cv2
    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 4:
        return None
    vertical = cv2.blur(image, (1, 9))
    horizontal = cv2.blur(image, (9, 1))
    scores = []
    for axis, blurred in ((0, vertical), (1, horizontal)):
        sharp = np.abs(np.diff(image, axis=axis))
        soft = np.abs(np.diff(blurred, axis=axis))
        lost = np.maximum(0.0, sharp - soft)[1:-1, 1:-1]
        total = float(sharp[1:-1, 1:-1].sum())
        if total <= 1e-6:
            # Ровная заливка без единого перепада — узнавать тут нечего.
            return 1.0
        scores.append((total - float(lost.sum())) / total)
    return float(max(scores))


def face_blur(image):
    """Оценка по миниатюре лица: PIL-картинка или массив BGR/серый."""
    import cv2
    if hasattr(image, 'convert'):
        gray = np.asarray(image.convert('L'))
    else:
        gray = np.asarray(image)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if min(height, width) < 4:
        return None
    # Края рамки — чаще фон и волосы, чем лицо: берём середину.
    core = gray[int(height * .1):max(int(height * .9), int(height * .1) + 2),
                int(width * .1):max(int(width * .9), int(width * .1) + 2)]
    grow = min(core.shape[:2]) < SAMPLE
    sample = cv2.resize(core, (SAMPLE, SAMPLE),
                        interpolation=cv2.INTER_CUBIC if grow else cv2.INTER_AREA)
    return blur_score(sample)


def read_thumbnail(path):
    """Миниатюра серым; `np.fromfile` — потому что путь бывает не в ASCII."""
    import cv2
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if not data.size:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def thumbnail_file(folder, thumbnail):
    """Миниатюра лица на этой машине: у ядра она может лежать только на хабе."""
    try:
        return catalogfiles.fetch(folder, thumbnail)
    except Exception:
        return Path(folder) / thumbnail


def box_size(raw_box):
    """Меньшая сторона рамки лица на оригинале, в точках."""
    try:
        left, top, right, bottom = [float(value) for value in json.loads(raw_box)[:4]]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return max(0.0, min(right - left, bottom - top))


def is_blurry(blur, size, threshold=BLUR_THRESHOLD, min_size=MIN_SIZE):
    """Мыло: оценка за порогом или лицо меньше минимального размера.

    Неизвестная оценка (миниатюры нет) мылом не считается — лучше показать
    лишнее лицо, чем молча спрятать нужное.
    """
    if blur is not None and blur >= threshold:
        return True
    return bool(min_size and size is not None and size < min_size)


def measure(db, folder, face_ids=None):
    """Досчитать оценку лицам, у которых её нет или она старой версии."""
    folder = Path(folder)
    if face_ids is None:
        rows = db.execute(
            'SELECT faces.id,faces.box,faces.thumbnail FROM faces '
            'LEFT JOIN face_quality ON face_quality.face_id=faces.id '
            'WHERE face_quality.face_id IS NULL OR face_quality.version<? '
            'ORDER BY faces.id', (VERSION,)).fetchall()
    else:
        rows = []
        wanted = sorted(set(face_ids))
        for offset in range(0, len(wanted), 900):
            batch = wanted[offset:offset + 900]
            rows += db.execute(
                f'SELECT id,box,thumbnail FROM faces WHERE id IN ({",".join("?" * len(batch))})',
                batch).fetchall()
    if not rows:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    found = []
    for face_id, raw_box, thumbnail in rows:
        blur = None
        if thumbnail:
            image = read_thumbnail(thumbnail_file(folder, thumbnail))
            if image is not None and face_crops.padded(thumbnail):
                image = face_crops.face_part(image)
            if image is not None:
                blur = face_blur(image)
        found.append((face_id, None if blur is None else round(blur, 4),
                      box_size(raw_box), VERSION, now))
    with db:
        # keep — решение человека, пересчёт оценки его не сбрасывает.
        db.executemany(
            'INSERT INTO face_quality(face_id,blur,size,version,computed_at) VALUES(?,?,?,?,?) '
            'ON CONFLICT(face_id) DO UPDATE SET blur=excluded.blur,size=excluded.size,'
            'version=excluded.version,computed_at=excluded.computed_at', found)
    return len(found)


def blurry_ids(db, threshold=BLUR_THRESHOLD, min_size=MIN_SIZE):
    """Лица, которые прячутся как мыльные. Оставленные человеком — не в счёт."""
    return {row[0] for row in db.execute(
        'SELECT face_id FROM face_quality WHERE keep=0 '
        'AND (blur>=? OR (?>0 AND size IS NOT NULL AND size<?))',
        (threshold, min_size, min_size))}


def keep(db, face_ids):
    """Человек сам назвал мыльное лицо — значит, оно ему нужно."""
    db.executemany('UPDATE face_quality SET keep=1 WHERE face_id=?',
                   [(face_id,) for face_id in face_ids])


def sharpness_weight(blur):
    """Во сколько раз резкость поднимает кадр при выборе лучших кадров трека."""
    if blur is None:
        return 1.0
    return float(np.clip((0.95 - blur) / 0.35, 0.15, 1.0))
