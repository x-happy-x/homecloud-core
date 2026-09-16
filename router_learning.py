"""Active-learning router built on already stored visual embeddings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

import numpy as np

import settings as catalog_settings


LABELS = {
    'person': ('Люди', 'a photo containing one or more people', 'a photo without people'),
    'portrait': ('Портрет', 'a portrait focused on a person', 'not a portrait'),
    'selfie': ('Селфи', 'a selfie taken by the person in the photo', 'not a selfie'),
    'group_photo': ('Групповое фото', 'a group photo with several people', 'not a group photo'),
    'indoor': ('В помещении', 'a scene indoors inside a building', 'a scene not indoors'),
    'outdoor': ('На улице', 'an outdoor scene outside', 'a scene not outdoors'),
    'landscape': ('Пейзаж', 'a landscape or nature photograph', 'not a landscape photograph'),
    'food': ('Еда', 'food or a meal is the main subject', 'food is not the main subject'),
    'pet': ('Животное', 'a pet or animal is visible', 'no pet or animal is visible'),
    'vehicle': ('Транспорт', 'a vehicle is visible', 'no vehicle is visible'),
    'architecture': ('Архитектура', 'architecture or a building is the main subject', 'not architecture'),
    'home': ('Дом', 'a home apartment or domestic interior', 'not a home interior'),
    'city': ('Город', 'a city street or urban scene', 'not a city or urban scene'),
    'beach': ('Пляж', 'a beach or seashore scene', 'not a beach'),
    'mountains': ('Горы', 'mountains are clearly visible', 'no mountains are visible'),
    'forest': ('Лес', 'a forest or woodland scene', 'not a forest'),
    'snow': ('Снег', 'snow or a snowy winter scene', 'no snow is visible'),
    'water': ('Вода', 'a sea lake river or large body of water', 'no body of water is visible'),
    'party': ('Праздник', 'a party celebration or festive gathering', 'not a party or celebration'),
    'concert': ('Концерт', 'a concert stage or live performance', 'not a concert'),
    'sports': ('Спорт', 'people doing sports or an athletic event', 'not sports'),
    'travel': ('Путешествие', 'a travel or sightseeing photograph', 'not a travel photograph'),
    'cat': ('Кошка', 'a cat is visible', 'no cat is visible'),
    'dog': ('Собака', 'a dog is visible', 'no dog is visible'),
    'bird': ('Птица', 'a bird is visible', 'no bird is visible'),
    'flowers': ('Цветы', 'flowers are clearly visible', 'no flowers are visible'),
    'product': ('Товар', 'a product or item photographed for a catalog or sale', 'not a product photograph'),
    'graphics': ('Графика', 'digital artwork illustration icon diagram or computer graphics', 'a natural camera photograph'),
    'game': ('Игра', 'a screenshot texture or asset from a video game', 'not video game content'),
    'screenshot': ('Скриншот', 'a computer or phone screenshot', 'a natural camera photograph'),
    'document': ('Документ', 'a document receipt book page or scanned paper', 'not a document'),
    'meme': ('Мем', 'an internet meme with text', 'not an internet meme'),
    'text_heavy': ('Много текста', 'an image containing a large amount of readable text', 'little or no text'),
    'receipt': ('Чек', 'a receipt invoice or payment slip', 'not a receipt or invoice'),
    'handwritten_text': ('Рукописный текст', 'handwritten notes or handwriting', 'no handwritten text'),
    'qr_code': ('QR-код', 'a visible QR code or barcode', 'no QR code or barcode'),
    'close_up': ('Крупный план', 'a close-up or macro photograph', 'not a close-up'),
    'mirror': ('Зеркало', 'a mirror reflection is an important part of the photo', 'no mirror reflection'),
    'night': ('Ночь', 'a night scene after dark', 'not a night scene'),
    'sunset': ('Закат', 'a sunset or sunrise with a colorful sky', 'not a sunset or sunrise'),
    'black_and_white': ('Чёрно-белое', 'a black and white monochrome image', 'a color image'),
    'blurred': ('Размыто', 'a blurry out of focus low quality image', 'a sharp in focus image'),
    'dark': ('Темно', 'a very dark underexposed or night image', 'a bright well exposed image'),
}

LABEL_GROUPS = {
    'Люди': {'person', 'portrait', 'selfie', 'group_photo'},
    'Место и сцена': {'indoor', 'outdoor', 'home', 'city', 'landscape', 'beach',
                      'mountains', 'forest', 'snow', 'water', 'party', 'concert',
                      'sports', 'travel'},
    'Объекты': {'food', 'pet', 'cat', 'dog', 'bird', 'flowers', 'vehicle',
                'architecture', 'product'},
    'Цифровое и текст': {'graphics', 'game', 'screenshot', 'document', 'meme',
                         'text_heavy', 'receipt', 'handwritten_text', 'qr_code'},
    'Вид кадра и качество': {'close_up', 'mirror', 'night', 'sunset',
                             'black_and_white', 'blurred', 'dark'},
}

SCHEMA = '''
CREATE TABLE IF NOT EXISTS router_predictions (
  path TEXT NOT NULL REFERENCES photos(path) ON DELETE CASCADE,
  label TEXT NOT NULL, score REAL NOT NULL, source TEXT NOT NULL,
  model_version TEXT NOT NULL, predicted_at TEXT NOT NULL,
  PRIMARY KEY(path,label,source,model_version)
);
CREATE INDEX IF NOT EXISTS router_predictions_source
  ON router_predictions(source,model_version,label,score);
CREATE TABLE IF NOT EXISTS router_training_labels (
  path TEXT NOT NULL REFERENCES photos(path) ON DELETE CASCADE,
  label TEXT NOT NULL, value INTEGER NOT NULL,
  verified_at TEXT NOT NULL, reviewer TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT 'human',
  PRIMARY KEY(path,label)
);
CREATE TABLE IF NOT EXISTS router_reviews (
  path TEXT PRIMARY KEY REFERENCES photos(path) ON DELETE CASCADE,
  verified_at TEXT NOT NULL, reviewer TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT 'human'
);
CREATE TABLE IF NOT EXISTS router_models (
  version TEXT PRIMARY KEY, embedding_model TEXT NOT NULL,
  status TEXT NOT NULL, trained_at TEXT NOT NULL, dataset_size INTEGER NOT NULL,
  labels_json TEXT NOT NULL, metrics_json TEXT NOT NULL, model_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS router_batches (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
  embedding_model TEXT NOT NULL, imported_at TEXT, source_name TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS router_batch_items (
  batch_id TEXT NOT NULL REFERENCES router_batches(id) ON DELETE CASCADE,
  item_name TEXT NOT NULL, path TEXT NOT NULL REFERENCES photos(path) ON DELETE CASCADE,
  PRIMARY KEY(batch_id,item_name), UNIQUE(batch_id,path)
);
CREATE INDEX IF NOT EXISTS router_batches_status ON router_batches(status,created_at);
CREATE TABLE IF NOT EXISTS router_skips (
  path TEXT PRIMARY KEY REFERENCES photos(path) ON DELETE CASCADE,
  skipped_at TEXT NOT NULL, reviewer TEXT NOT NULL DEFAULT ''
);
'''


def connect(catalog):
    db = sqlite3.connect(Path(catalog) / 'catalog.sqlite', timeout=60)
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript(SCHEMA)
    for table in ('router_training_labels', 'router_reviews'):
        columns = {row[1] for row in db.execute(f'PRAGMA table_info({table})')}
        if 'source' not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN source TEXT NOT NULL DEFAULT 'human'")
    db.commit()
    return db


def now():
    return datetime.now(timezone.utc).isoformat()


def write_progress(path, **value):
    if not path:
        return
    path = Path(path)
    value.update(updated_at=time.time(), pid=os.getpid())
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05 * (attempt + 1))


def label_payload():
    groups = {label: group for group, labels in LABEL_GROUPS.items() for label in labels}
    return [{'id': key, 'title': value[0], 'group': groups.get(key, 'Другое')}
            for key, value in LABELS.items()]


def batch_categories():
    """Machine-readable vocabulary shipped with every offline review batch."""
    groups = {label: group for group, labels in LABEL_GROUPS.items() for label in labels}
    return [{'id': key, 'title_ru': value[0], 'group_ru': groups.get(key, 'Другое'),
             'meaning_en': value[1]} for key, value in LABELS.items()]


def pending_batch_paths(catalog):
    """Do not put one photo into several outstanding archives at once."""
    db = connect(catalog)
    try:
        # An abandoned package stops reserving photos after seven days.
        db.execute("UPDATE router_batches SET status='expired' WHERE status='exported' "
                   "AND datetime(created_at) < datetime('now','-7 days')")
        db.commit()
        return {row[0] for row in db.execute('''SELECT i.path FROM router_batch_items i
            JOIN router_batches b ON b.id=i.batch_id WHERE b.status='exported' ''')}
    finally:
        db.close()


BATCH_SIZES = (1, 3, 5, 10, 15, 20)


def batch_size(value):
    """Размер пакета из разрешённых; неизвестное — десять, как раньше."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 10
    return value if value in BATCH_SIZES else 10


def create_batch(catalog, paths, limit=10):
    if not paths:
        raise ValueError('В очереди нет фотографий для нового пакета')
    paths = list(dict.fromkeys(str(path) for path in paths))[:batch_size(limit)]
    db = connect(catalog)
    try:
        model = catalog_settings.read(db)['visual_model']
        batch_id = uuid.uuid4().hex
        items = [{'file': f'{index:02d}.jpg', 'path': path}
                 for index, path in enumerate(paths, 1)]
        with db:
            db.execute('INSERT INTO router_batches(id,created_at,status,embedding_model) '
                       'VALUES(?,?,?,?)', (batch_id, now(), 'exported', model))
            db.executemany('INSERT INTO router_batch_items(batch_id,item_name,path) VALUES(?,?,?)',
                           [(batch_id, item['file'], item['path']) for item in items])
        return {'batch_id': batch_id, 'embedding_model': model, 'items': items}
    finally:
        db.close()


def batch_prompt(batch_id, item_names):
    files = ', '.join(item_names)
    count = len(item_names)
    return f'''Проверь все изображения из этого архива ({count} шт.): {files}.

Цель: присвоить каждому файлу все визуально подтверждённые категории из categories.json.

Правила:
1. Используй только id из categories.json. Категорий может быть несколько.
2. Описывай только то, что действительно видно. Не угадывай имена, личности, отношения, точный возраст или место.
3. Не добавляй категорию «на всякий случай». Если признака нет, не включай его в labels.
4. Каждый файл должен встретиться ровно один раз, даже если labels пуст.
5. Верни ТОЛЬКО валидный JSON: без Markdown, ``` и текста до/после.
6. Не меняй batch_id: {batch_id}

Формат ответа:
{{
  "batch_id": "{batch_id}",
  "items": [
    {{"file": "01.jpg", "labels": ["person", "portrait"]}},
    {{"file": "02.jpg", "labels": []}}
  ]
}}
'''.strip() + '\n'


def parse_answer(text):
    """Ответ нейросети как текст: снимаем ```json-обёртку и лишнее вокруг объекта."""
    text = str(text or '').strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1] if '\n' in text else ''
        text = text.rsplit('```', 1)[0]
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('В ответе не найден JSON-объект')
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f'Ответ не разобрался как JSON: {exc.msg} (строка {exc.lineno})') from exc


def import_batch(catalog, payload, reviewer=''):
    if isinstance(payload, dict) and isinstance(payload.get('text'), str):
        payload = parse_answer(payload['text'])
    if not isinstance(payload, dict):
        raise ValueError('Ответ должен быть JSON-объектом')
    batch_id = str(payload.get('batch_id', '')).strip()
    items = payload.get('items')
    if not batch_id or not isinstance(items, list):
        raise ValueError('В ответе нужны batch_id и массив items')
    db = connect(catalog)
    try:
        batch = db.execute('SELECT status FROM router_batches WHERE id=?', (batch_id,)).fetchone()
        if not batch:
            raise ValueError('Пакет не найден в HomeCloud')
        if batch[0] == 'imported':
            raise ValueError('Ответ этого пакета уже загружен')
        expected = dict(db.execute(
            'SELECT item_name,path FROM router_batch_items WHERE batch_id=? ORDER BY item_name',
            (batch_id,)))
        received = {}
        for item in items:
            if not isinstance(item, dict):
                raise ValueError('Каждый элемент items должен быть объектом')
            name = str(item.get('file', '')).strip()
            labels = item.get('labels')
            if name in received:
                raise ValueError(f'Файл {name} указан дважды')
            if name not in expected or not isinstance(labels, list):
                raise ValueError(f'Неверный файл или labels: {name}')
            # Нейросеть иногда отвечает русскими названиями вместо id.
            titles = {value[0].casefold(): key for key, value in LABELS.items()}
            labels = [titles.get(str(label).casefold(), str(label)) for label in labels]
            unknown = sorted(set(map(str, labels)) - set(LABELS))
            if unknown:
                raise ValueError(f'Неизвестные категории для {name}: {", ".join(unknown)}')
            received[name] = set(map(str, labels))
        missing = sorted(set(expected) - set(received))
        extra = sorted(set(received) - set(expected))
        if missing or extra:
            raise ValueError('Набор файлов не совпадает с пакетом. '
                             f'Пропущено: {missing}; лишние: {extra}')
        stamp = now()
        imported = 0
        skipped = []
        with db:
            for name, path in expected.items():
                prior = db.execute('SELECT source FROM router_reviews WHERE path=?',
                                   (path,)).fetchone()
                if prior and prior[0] != 'ai_batch':
                    skipped.append(name)
                    continue
                values = received[name]
                db.executemany('''INSERT INTO router_training_labels
                    (path,label,value,verified_at,reviewer,source) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(path,label) DO UPDATE SET value=excluded.value,
                    verified_at=excluded.verified_at,reviewer=excluded.reviewer,
                    source=excluded.source''',
                    [(path, key, int(key in values), stamp, reviewer[:120], 'ai_batch')
                     for key in LABELS])
                db.execute('''INSERT INTO router_reviews(path,verified_at,reviewer,source)
                    VALUES(?,?,?,?) ON CONFLICT(path) DO UPDATE SET
                    verified_at=excluded.verified_at,reviewer=excluded.reviewer,
                    source=excluded.source''', (path, stamp, reviewer[:120], 'ai_batch'))
                imported += 1
            db.execute("UPDATE router_batches SET status='imported',imported_at=?,source_name=? "
                       'WHERE id=?', (stamp, reviewer[:120], batch_id))
        return {'batch_id': batch_id, 'imported': imported, 'skipped': skipped}
    finally:
        db.close()


def prediction_source(db, embedding_model):
    active = db.execute(
        "SELECT version FROM router_models WHERE status='active' AND embedding_model=? "
        'ORDER BY trained_at DESC LIMIT 1', (embedding_model,)).fetchone()
    return ('trained', active[0]) if active else ('zero_shot', embedding_model)


def summary(catalog):
    db = connect(catalog)
    try:
        embedding_model = catalog_settings.read(db)['visual_model']
        source, version = prediction_source(db, embedding_model)
        embedded = db.execute(
            'SELECT COUNT(*) FROM photo_embeddings WHERE model=?', (embedding_model,)).fetchone()[0]
        predicted = db.execute(
            'SELECT COUNT(DISTINCT path) FROM router_predictions WHERE source=? AND model_version=?',
            (source, version)).fetchone()[0]
        reviewed = db.execute('SELECT COUNT(*) FROM router_reviews').fetchone()[0]
        human = db.execute(
            "SELECT COUNT(*) FROM router_reviews WHERE source='human'").fetchone()[0]
        propagated = db.execute(
            "SELECT COUNT(*) FROM router_reviews WHERE source='propagated'").fetchone()[0]
        ai_reviewed = db.execute(
            "SELECT COUNT(*) FROM router_reviews WHERE source='ai_batch'").fetchone()[0]
        pending_batches = db.execute(
            "SELECT COUNT(*) FROM router_batches WHERE status='exported'").fetchone()[0]
        skipped = db.execute('SELECT COUNT(*) FROM router_skips').fetchone()[0]
        models = [{
            'version': row[0], 'embedding_model': row[1], 'status': row[2],
            'trained_at': row[3], 'dataset_size': row[4],
            'labels': json.loads(row[5]), 'metrics': json.loads(row[6]),
        } for row in db.execute(
            'SELECT version,embedding_model,status,trained_at,dataset_size,labels_json,metrics_json '
            'FROM router_models ORDER BY trained_at DESC')]
        last_size = max((model['dataset_size'] for model in models
                         if model['embedding_model'] == embedding_model), default=0)
        options = catalog_settings.read(db)
        return {'embedding_model': embedding_model, 'source': source,
                'active_version': version if source == 'trained' else '',
                'embedded': embedded, 'predicted': predicted, 'reviewed': reviewed,
                'human_reviewed': human, 'propagated': propagated,
                'ai_reviewed': ai_reviewed, 'pending_batches': pending_batches,
                'skipped': skipped, 'batch_sizes': list(BATCH_SIZES),
                'pending': max(0, predicted - reviewed - skipped), 'labels': label_payload(),
                'models': models, 'auto_train': options['router_auto_train'],
                'auto_train_every': options['router_auto_train_every'],
                'new_since_training': max(0, reviewed - last_size)}
    finally:
        db.close()


def suggested_labels(scores, trained=False, limit=4):
    """Метки, которые стоит предложить человеку.

    Своя обученная версия откалибрована — берём всё, что выше половины. Общая
    zero-shot модель ошибается при любом пороге (на проверенных снимках порог
    0.6 угадывал 17% меток, а подсказывал по десятку на снимок), поэтому от
    неё — только несколько самых уверенных вариантов, и их не отмечаем заранее.
    """
    if trained:
        picked = sorted(((score, label) for label, score in scores.items() if score >= .5), reverse=True)
    else:
        picked = sorted(((score, label) for label, score in scores.items() if score >= .55), reverse=True)
    return [label for _, label in picked[:limit if not trained else 8]]


def skip(catalog, path, reviewer=''):
    db = connect(catalog)
    try:
        with db:
            db.execute('INSERT INTO router_skips(path,skipped_at,reviewer) VALUES(?,?,?) '
                       'ON CONFLICT(path) DO UPDATE SET skipped_at=excluded.skipped_at',
                       (str(path), now(), reviewer[:120]))
    finally:
        db.close()


def clear_skips(catalog):
    db = connect(catalog)
    try:
        with db:
            return db.execute('DELETE FROM router_skips').rowcount
    finally:
        db.close()


def pending_batches(catalog):
    """Выданные и ещё не загруженные пакеты: кому что отдали и какой промпт."""
    db = connect(catalog)
    try:
        db.execute("UPDATE router_batches SET status='expired' WHERE status='exported' "
                   "AND datetime(created_at) < datetime('now','-7 days')")
        db.commit()
        result = []
        for batch_id, created_at in db.execute(
                "SELECT id,created_at FROM router_batches WHERE status='exported' "
                'ORDER BY created_at DESC').fetchall():
            items = [{'file': name, 'path': path} for name, path in db.execute(
                'SELECT item_name,path FROM router_batch_items WHERE batch_id=? ORDER BY item_name',
                (batch_id,))]
            result.append({'batch_id': batch_id, 'created_at': created_at, 'items': items,
                           'prompt': batch_prompt(batch_id, [item['file'] for item in items])})
        return result
    finally:
        db.close()


def cancel_batch(catalog, batch_id):
    db = connect(catalog)
    try:
        with db:
            changed = db.execute("UPDATE router_batches SET status='cancelled' "
                                 "WHERE id=? AND status='exported'", (str(batch_id),)).rowcount
        if not changed:
            raise ValueError('Пакет не найден или уже загружен')
    finally:
        db.close()


def review_queue(catalog, limit=24, hide_adult=False):
    db = connect(catalog)
    try:
        embedding_model = catalog_settings.read(db)['visual_model']
        source, version = prediction_source(db, embedding_model)
        adult_clause = (" AND p.path NOT IN (SELECT path FROM photo_adult_analysis "
                        "WHERE status='ok' AND rating NOT IN ('safe','unknown','sensitive'))"
                        if hide_adult else '')
        rows = db.execute('''
            SELECT p.path,AVG(ABS(p.score-.5)) AS uncertainty
            FROM router_predictions p
            JOIN photos f ON f.path=p.path
            LEFT JOIN router_reviews r ON r.path=p.path
            WHERE p.source=? AND p.model_version=? AND r.path IS NULL
              AND f.status='ok' AND COALESCE(f.blocked,0)=0
              AND p.path NOT IN (SELECT path FROM hidden_photos)
              AND p.path NOT IN (SELECT path FROM router_skips)
              AND p.path NOT IN (SELECT i.path FROM router_batch_items i
                                 JOIN router_batches b ON b.id=i.batch_id
                                 WHERE b.status='exported')''' + adult_clause + '''
            GROUP BY p.path ORDER BY uncertainty ASC,p.path LIMIT ?''',
            (source, version, max(1, min(int(limit), 100)))).fetchall()
        # Оценки всех снимков — одним запросом по первичному ключу. По запросу на
        # снимок планировщик брал индекс по источнику и перебирал весь каталог:
        # пять секунд на тридцать снимков.
        by_path = {path: {} for path, _ in rows}
        if by_path:
            marks = ','.join('?' * len(by_path))
            for path, label, score in db.execute(
                    'SELECT path,label,score FROM router_predictions '
                    'INDEXED BY sqlite_autoindex_router_predictions_1 '
                    f'WHERE path IN ({marks}) AND source=? AND model_version=?',
                    [*by_path, source, version]):
                by_path[path][label] = round(float(score), 4)
        result = []
        for path, uncertainty in rows:
            scores = by_path[path]
            result.append({'path': path, 'scores': scores,
                           'suggested': suggested_labels(scores, source == 'trained'),
                           'trained': source == 'trained',
                           'uncertainty': round(float(uncertainty), 4)})
        return result
    finally:
        db.close()


def save_review(catalog, path, values, reviewer='', source='human'):
    db = connect(catalog)
    try:
        known = db.execute(
            "SELECT 1 FROM photos WHERE path=? AND status='ok'", (path,)).fetchone()
        if not known:
            raise ValueError('Фотография не найдена')
        stamp = now()
        clean = {key: int(bool(values.get(key))) for key in LABELS}
        with db:
            db.executemany('''INSERT INTO router_training_labels
                (path,label,value,verified_at,reviewer,source) VALUES(?,?,?,?,?,?)
                ON CONFLICT(path,label) DO UPDATE SET value=excluded.value,
                verified_at=excluded.verified_at,reviewer=excluded.reviewer,
                source=excluded.source''',
                [(path, key, value, stamp, reviewer[:120], source) for key, value in clean.items()])
            db.execute('''INSERT INTO router_reviews(path,verified_at,reviewer,source) VALUES(?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET verified_at=excluded.verified_at,
                reviewer=excluded.reviewer,source=excluded.source''',
                (path, stamp, reviewer[:120], source))
        return clean
    finally:
        db.close()


def similar_paths(catalog, path, threshold=.985, limit=100):
    """Unreviewed near-identical images according to the selected visual index."""
    db = connect(catalog)
    try:
        model = catalog_settings.read(db)['visual_model']
        row = db.execute('SELECT embedding,dims FROM photo_embeddings WHERE path=? AND model=?',
                         (path, model)).fetchone()
        if not row:
            return []
        base = np.frombuffer(row[0], dtype=np.float32, count=row[1]).copy()
        base /= max(float(np.linalg.norm(base)), 1e-9)
        result = []
        for raw, blob, dims in db.execute('''SELECT e.path,e.embedding,e.dims
            FROM photo_embeddings e JOIN photos p ON p.path=e.path
            LEFT JOIN router_reviews r ON r.path=e.path
            WHERE e.model=? AND e.path<>? AND r.path IS NULL
              AND p.status='ok' AND COALESCE(p.blocked,0)=0
              AND e.path NOT IN (SELECT path FROM hidden_photos)''', (model, path)):
            vector = np.frombuffer(blob, dtype=np.float32, count=dims)
            score = float(base @ vector / max(float(np.linalg.norm(vector)), 1e-9))
            if score >= threshold:
                result.append((score, raw))
        result.sort(reverse=True)
        return [raw for _, raw in result[:max(0, min(int(limit), 200))]]
    finally:
        db.close()


def save_similar(catalog, path, values, reviewer='', threshold=.985):
    paths = similar_paths(catalog, path, threshold)
    for raw in paths:
        save_review(catalog, raw, values, reviewer, 'propagated')
    return paths


def apply_trained(db, embedding_model, version, model_path, stamp=None):
    """Apply a saved router to every matching embedding, without opening photos."""
    import torch
    saved = torch.load(model_path, map_location='cpu', weights_only=False)
    if saved['embedding_model'] != embedding_model:
        raise RuntimeError('Модель роутера не соответствует визуальному индексу')
    model = torch.nn.Sequential(torch.nn.Linear(saved['input_dims'], saved['hidden']),
                                torch.nn.GELU(), torch.nn.Dropout(.15),
                                torch.nn.Linear(saved['hidden'], len(saved['labels'])))
    model.load_state_dict(saved['state_dict'])
    model.eval()
    stamp = stamp or now()
    rows = db.execute('SELECT path,embedding,dims FROM photo_embeddings WHERE model=?',
                      (embedding_model,)).fetchall()
    for offset in range(0, len(rows), 1024):
        part = rows[offset:offset + 1024]
        matrix = np.stack([np.frombuffer(row[1], dtype=np.float32, count=row[2]) for row in part])
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        with torch.inference_mode():
            scores = torch.sigmoid(model(torch.from_numpy(matrix.astype('float32')))).numpy()
        values = [(path, label, float(scores[i, j]), 'trained', version, stamp)
                  for i, (path, _, _) in enumerate(part)
                  for j, label in enumerate(saved['labels'])]
        with db:
            db.executemany('''INSERT INTO router_predictions VALUES(?,?,?,?,?,?)
                ON CONFLICT(path,label,source,model_version) DO UPDATE SET
                score=excluded.score,predicted_at=excluded.predicted_at''', values)
    return len(rows)


def bootstrap(catalog, progress=None):
    from analyze_photos import VisualEncoder
    db = connect(catalog)
    try:
        model_name = catalog_settings.read(db)['visual_model']
        rows = db.execute(
            'SELECT path,embedding,dims FROM photo_embeddings WHERE model=? ORDER BY path',
            (model_name,)).fetchall()
        if not rows:
            raise RuntimeError('Для выбранной модели ещё нет визуального индекса')
        write_progress(progress, status='running', action='bootstrap', total=len(rows), completed=0)
        encoder = VisualEncoder(model_name)
        prompts = [text for value in LABELS.values() for text in value[1:3]]
        text_vectors = encoder.texts(prompts).float().cpu().numpy().reshape(len(LABELS), 2, -1)
        stamp = now()
        batch = []
        for offset in range(0, len(rows), 512):
            part = rows[offset:offset + 512]
            matrix = np.stack([np.frombuffer(row[1], dtype=np.float32, count=row[2]) for row in part])
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
            delta = np.einsum('nd,ld->nl', matrix, text_vectors[:, 0] - text_vectors[:, 1])
            scores = 1 / (1 + np.exp(-delta * 10.0))
            for row_index, (path, _, _) in enumerate(part):
                batch.extend((path, label, float(scores[row_index, label_index]),
                              'zero_shot', model_name, stamp)
                             for label_index, label in enumerate(LABELS))
            with db:
                db.executemany('''INSERT INTO router_predictions
                    (path,label,score,source,model_version,predicted_at) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(path,label,source,model_version) DO UPDATE SET
                    score=excluded.score,predicted_at=excluded.predicted_at''', batch)
            batch.clear()
            write_progress(progress, status='running', action='bootstrap', total=len(rows),
                           completed=min(offset + len(part), len(rows)))
        active = db.execute(
            "SELECT version,model_path FROM router_models WHERE status='active' "
            'AND embedding_model=? LIMIT 1', (model_name,)).fetchone()
        if active:
            apply_trained(db, model_name, active[0], active[1], stamp)
        write_progress(progress, status='completed', action='bootstrap', total=len(rows),
                       completed=len(rows), model=model_name)
    finally:
        db.close()


def group_key(db, path):
    row = db.execute('SELECT COALESCE(dhash,sha1) FROM photo_hashes WHERE path=?',
                     (path,)).fetchone()
    return row[0] if row and row[0] else path


def train(catalog, progress=None):
    import torch
    db = connect(catalog)
    try:
        embedding_model = catalog_settings.read(db)['visual_model']
        paths = [row[0] for row in db.execute('SELECT path FROM router_reviews ORDER BY path')]
        if len(paths) < 12:
            raise RuntimeError('Для обучения нужно проверить минимум 12 фотографий')
        marks = ','.join('?' * len(paths))
        vectors = {row[0]: np.frombuffer(row[1], dtype=np.float32, count=row[2]).copy()
                   for row in db.execute(
                       f'SELECT path,embedding,dims FROM photo_embeddings WHERE model=? '
                       f'AND path IN ({marks})', [embedding_model, *paths])}
        paths = [path for path in paths if path in vectors]
        if len(paths) < 12:
            raise RuntimeError('У проверенных снимков недостаточно индексов выбранной модели')
        marks = ','.join('?' * len(paths))
        targets = {path: {} for path in paths}
        for path, label, value in db.execute(
                f'SELECT path,label,value FROM router_training_labels WHERE path IN ({marks})',
                paths):
            targets[path][label] = value
        trainable = [label for label in LABELS
                     if 1 < sum(targets[path].get(label, 0) for path in paths) < len(paths) - 1]
        if not trainable:
            raise RuntimeError('Нужны хотя бы две положительные и две отрицательные метки одного типа')
        groups = {}
        for path in paths:
            groups.setdefault(group_key(db, path), []).append(path)
        validation = {path for key, items in groups.items()
                      if int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 5 == 0
                      for path in items}
        if len(validation) < 2 or len(paths) - len(validation) < 4:
            validation = set(paths[::5])
        train_paths = [path for path in paths if path not in validation]
        val_paths = [path for path in paths if path in validation]
        def tensors(items):
            x = np.stack([vectors[path] for path in items]).astype('float32')
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
            y = np.array([[targets[path].get(label, 0) for label in trainable]
                          for path in items], dtype='float32')
            return torch.from_numpy(x), torch.from_numpy(y)
        x_train, y_train = tensors(train_paths)
        x_val, y_val = tensors(val_paths)
        hidden = min(256, max(64, x_train.shape[1] // 3))
        torch.manual_seed(41)
        model = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], hidden),
                                    torch.nn.GELU(), torch.nn.Dropout(.15),
                                    torch.nn.Linear(hidden, len(trainable)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=.01)
        write_progress(progress, status='running', action='train', total=120, completed=0)
        model.train()
        for epoch in range(120):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x_train), y_train)
            loss.backward(); optimizer.step()
            if epoch % 5 == 0:
                write_progress(progress, status='running', action='train', total=120,
                               completed=epoch + 1, loss=round(float(loss.detach()), 5))
        model.eval()
        with torch.inference_mode():
            prediction = (torch.sigmoid(model(x_val)) >= .5).int().numpy()
        truth = y_val.int().numpy()
        metrics = {}
        f1s = []
        for index, label in enumerate(trainable):
            tp = int(((prediction[:, index] == 1) & (truth[:, index] == 1)).sum())
            fp = int(((prediction[:, index] == 1) & (truth[:, index] == 0)).sum())
            fn = int(((prediction[:, index] == 0) & (truth[:, index] == 1)).sum())
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            metrics[label] = {'f1': round(f1, 4), 'support': int(truth[:, index].sum())}
            f1s.append(f1)
        metrics['macro_f1'] = round(sum(f1s) / len(f1s), 4)
        number = db.execute('SELECT COUNT(*) FROM router_models').fetchone()[0] + 1
        version = f'router-v{number}'
        folder = Path(catalog) / 'router-models'; folder.mkdir(exist_ok=True)
        target = folder / f'{version}.pt'
        torch.save({'state_dict': model.state_dict(), 'input_dims': x_train.shape[1],
                    'hidden': hidden, 'labels': trainable, 'embedding_model': embedding_model}, target)
        stamp = now()
        with db:
            db.execute('''INSERT INTO router_models
                (version,embedding_model,status,trained_at,dataset_size,labels_json,metrics_json,model_path)
                VALUES(?,?,?,?,?,?,?,?)''', (version, embedding_model, 'candidate', stamp,
                len(paths), json.dumps(trainable), json.dumps(metrics), str(target)))
        # Применяем кандидат ко всем векторам без чтения исходных фотографий.
        apply_trained(db, embedding_model, version, target, stamp)
        write_progress(progress, status='completed', action='train', total=120, completed=120,
                       version=version, metrics=metrics)
    finally:
        db.close()


def activate(catalog, version):
    db = connect(catalog)
    try:
        row = db.execute('SELECT embedding_model FROM router_models WHERE version=?',
                         (version,)).fetchone()
        if not row:
            raise ValueError('Версия модели не найдена')
        if row[0] != catalog_settings.read(db)['visual_model']:
            raise ValueError('Эта версия обучена для другой визуальной модели')
        with db:
            db.execute("UPDATE router_models SET status='candidate' WHERE status='active'")
            db.execute("UPDATE router_models SET status='active' WHERE version=?", (version,))
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('bootstrap', 'train'))
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--progress', type=Path)
    args = parser.parse_args()
    try:
        globals()[args.action](args.catalog.resolve(), args.progress)
    except Exception as exc:
        write_progress(args.progress, status='error', action=args.action, error=str(exc))
        raise


if __name__ == '__main__':
    main()
