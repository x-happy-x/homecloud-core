"""Настройки каталога: что сканировать, что пропускать, сколько кадров брать.

Живут в самом каталоге (таблица `settings`), поэтому их видят и бэкенд, и
отдельные скрипты этапов: правила пропуска нужны сканеру, а интерфейс их только
показывает и меняет.
"""
import json
import sqlite3

SCHEMA = '''
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  changed REAL
);
'''

# Значение по умолчанию задаёт и тип: строкой, числом или флагом настройка и
# останется после чтения из базы.
DEFAULTS = {
    # видео
    'video_enabled': True,
    # раз в сколько секунд искать лица: чаще — плотнее треки, но дольше счёт
    'video_track_step': 0.5,
    # если трек не удалось продолжить дольше стольких секунд — он закрывается
    'video_track_gap': 1.2,
    # сколько самых уверенных кадров трека усредняется в итоговый эмбеддинг
    'video_track_best': 8,
    'video_min_seconds': 0.0,
    'video_max_seconds': 0.0,
    # сколько кадров ролика смотрит проверка 18+: одного кадра из начала мало —
    # берётся несколько по всей длине и худший результат.
    'adult_video_frames': 3,
    # правила пропуска
    'min_side': 0,
    'min_kilobytes': 0,
    'max_ratio': 0.0,
    'ignore_patterns': '',
    # куда переносить скрытые снимки; пусто — папка `hidden` внутри каталога
    'hidden_root': '',
    # пути, которые не сканируются и не показываются, и исключения из них
    'block_paths': '',
    'allow_paths': '',
    # модель общего визуального индекса; индексы разных моделей хранятся рядом
    'visual_model': 'google/siglip2-base-patch16-224',
    # active learning: новую версию обучаем автоматически, но не активируем.
    'router_auto_train': True,
    'router_auto_train_every': 50,
    # Подсказка «эта группа похожа на такого-то»: средний вектор группы
    # сравнивается с каждым уже названным лицом. Порог подобран на реальном
    # каталоге: похожесть на чужого человека там ни разу не поднялась выше
    # 0.53, так что 0.60 оставляет запас. Ниже — больше подсказок и больше
    # шанс ошибиться, выше — подсказок меньше, зато они вернее.
    'face_suggest_enabled': True,
    'face_suggest_threshold': 0.60,
    # Разбор остатка: «шум» у кластеризации понятие относительное, человек с
    # четырьмя лицами рядом с гроздью из четырёхсот выпадает. Второй проход
    # идёт по одному остатку, и там нужен размер группы поменьше.
    'noise_cluster_size': 3,
    # Мыльные лица (face_quality.py): резкость миниатюры от 0 — резко до 1 —
    # мыло. От порога и выше лицо не группируется и прячется в «Размытые».
    # 1.0 — не прятать ничего. Лицо меньше face_min_size точек — тоже мыло.
    'face_blur_threshold': 0.76,
    'face_min_size': 20,
    # описание изображений: своя видеокарта (local) или уже запущенный LM Studio
    'caption_backend': 'local',
    'caption_model': 'Qwen/Qwen3-VL-2B-Instruct',
    'caption_lmstudio_url': 'http://127.0.0.1:1234/v1/chat/completions',
    # распознавание речи в роликах
    'speech_model': 'large-v3',
    # 'auto' — определять язык по самой записи; иначе код языка
    'speech_language': 'auto',
    # чем подменяется язык, когда определению нельзя верить
    'speech_fallback_language': 'ru',
    # автоматические подборки: снимок без пройденной проверки 18+ в них не идёт
    'highlights_require_adult_check': True,
}

LIMITS = {
    'video_track_step': (0.1, 5.0),
    'video_track_gap': (0.3, 10.0),
    'video_track_best': (1, 30),
    'adult_video_frames': (1, 20),
    'video_min_seconds': (0.0, 3600.0),
    'video_max_seconds': (0.0, 86400.0),
    'min_side': (0, 20000),
    'min_kilobytes': (0, 1024 * 1024),
    'max_ratio': (0.0, 100.0),
    'router_auto_train_every': (12, 1000),
    'face_suggest_threshold': (0.30, 0.95),
    'noise_cluster_size': (2, 8),
    'face_blur_threshold': (0.5, 1.0),
    'face_min_size': (0, 200),
}

CHOICES = {
    'caption_backend': ('local', 'lmstudio'),
    'speech_model': ('large-v3', 'medium', 'small'),
    'visual_model': (
        'google/siglip2-base-patch16-224',
        'google/siglip2-base-patch16-256',
        'jinaai/jina-clip-v2',
    ),
}

VISUAL_MODELS = (
    {'id': 'google/siglip2-base-patch16-224', 'name': 'SigLIP 2 Base 224',
     'note': 'текущая, быстрая', 'bytes': 1500800904},
    {'id': 'google/siglip2-base-patch16-256', 'name': 'SigLIP 2 Base 256',
     'note': 'точнее на деталях', 'bytes': 1500985224},
    {'id': 'jinaai/jina-clip-v2', 'name': 'Jina CLIP v2',
     'note': 'лучший русский поиск, некоммерческая лицензия', 'bytes': 1730688642},
)


def visual_models(cache=r'C:\cv-models\huggingface'):
    """Configured models and whether their PyTorch weights finished downloading."""
    from pathlib import Path
    root = Path(cache) / 'hub'
    result = []
    for item in VISUAL_MODELS:
        folder = root / ('models--' + item['id'].replace('/', '--')) / 'snapshots'
        weights = list(folder.glob('*/model.safetensors')) if folder.is_dir() else []
        installed = any(path.stat().st_size == item['bytes'] for path in weights)
        result.append({key: value for key, value in item.items() if key != 'bytes'} |
                      {'installed': installed})
    return result


def ensure_schema(db):
    db.executescript(SCHEMA)
    db.commit()


def _cast(key, value):
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return bool(value) if not isinstance(value, str) else value.strip().lower() in {
            '1', 'true', 'yes', 'on', 'да'}
    if isinstance(default, int):
        value = int(float(value))
    elif isinstance(default, float):
        value = float(value)
    else:
        value = str(value)
        choices = CHOICES.get(key)
        return value if not choices or value in choices else default
    low, high = LIMITS.get(key, (None, None))
    if low is not None:
        value = max(low, min(high, value))
    return value


def read(db):
    """Все настройки: значения по умолчанию, поверх них сохранённые."""
    ensure_schema(db)
    values = dict(DEFAULTS)
    for key, raw in db.execute('SELECT key,value FROM settings'):
        if key not in DEFAULTS:
            continue
        try:
            values[key] = _cast(key, json.loads(raw))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return values


def write(db, changes):
    """Сохраняем только знакомые ключи — мусор в настройки не попадает."""
    import time
    ensure_schema(db)
    known = {key: _cast(key, value) for key, value in (changes or {}).items()
             if key in DEFAULTS}
    if known:
        with db:
            db.executemany(
                'INSERT INTO settings(key,value,changed) VALUES(?,?,?) '
                'ON CONFLICT(key) DO UPDATE SET value=excluded.value,changed=excluded.changed',
                [(key, json.dumps(value, ensure_ascii=False), time.time())
                 for key, value in known.items()])
    return read(db)


def load(catalog):
    """Настройки прямо из папки каталога — для скриптов этапов."""
    from pathlib import Path
    folder = Path(catalog)
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / 'catalog.sqlite', timeout=30)
    try:
        return read(db)
    finally:
        db.close()


def patterns(values):
    """Маски имён из настроек: по одной в строке или через запятую."""
    raw = str(values.get('ignore_patterns') or '')
    return [item.strip().casefold() for item in raw.replace(',', '\n').splitlines()
            if item.strip()]


def rejects(values, *, size=None, width=0, height=0):
    """Почему файл пропускаем — текстом, либо None, если он подходит."""
    if size is not None and values.get('min_kilobytes') and size < values['min_kilobytes'] * 1024:
        return 'файл меньше минимального размера'
    if width and height:
        if values.get('min_side') and min(width, height) < values['min_side']:
            return 'сторона меньше минимальной'
        ratio = max(width, height) / max(1, min(width, height))
        if values.get('max_ratio') and ratio > values['max_ratio']:
            return 'слишком вытянутое изображение'
    return None
