"""Ключи снимков: «источник:путь внутри источника».

Каталог один на все источники, поэтому путь сам по себе ничего не значит:
D:\\Фото есть и на PC-X, и на PC-A. Ключ — id источника, двоеточие и путь в
родном для источника виде: `pc-x:D:\\Фото\\a.jpg`, `netcraze:/Фото/a.jpg`.
Id источника не короче двух символов, так что с буквой диска он не путается.

Каталог без ключей (старый, одного компьютера) тоже работает: строка без
префикса — обычный путь этой машины, и все функции ведут себя как раньше.

Модуль без зависимостей и без pathlib: его читают и хаб на Linux (там pathlib
не понимает обратных слэшей), и этапы обработки в любых окружениях Windows.
"""
import os
import re

SOURCE_ID = re.compile(r'^[a-z0-9][a-z0-9_-]{1,31}$')
_PREFIX = re.compile(r'^([a-z0-9][a-z0-9_-]{1,31}):(.*)$', re.S)
_DRIVE = re.compile(r'^[A-Za-z]:')


def split(key):
    """('pc-x', 'D:\\a.jpg') для ключа, ('', путь) для старого пути без источника."""
    key = str(key or '')
    match = _PREFIX.match(key)
    if not match:
        return '', key
    return match.group(1), match.group(2)


def make(source, native):
    if not SOURCE_ID.match(str(source or '')):
        raise ValueError(f'Некорректный id источника: {source!r}')
    return f'{source}:{native}'


def source_of(key):
    return split(key)[0]


def native(key):
    return split(key)[1]


def is_key(value):
    return bool(_PREFIX.match(str(value or '')))


def sep_of(value):
    """Разделитель пути: обратный слэш у путей Windows, прямой у остальных."""
    text = split(value)[1]
    if '\\' in text or _DRIVE.match(text):
        return '\\'
    if '/' in text:
        return '/'
    return os.sep


def _root_length(text):
    """Длина корня пути: `D:\\`, `\\\\host\\share\\`, `/`."""
    if _DRIVE.match(text):
        return 3 if len(text) > 2 and text[2] in '\\/' else 2
    if text.startswith(('\\\\', '//')):
        parts = re.split(r'[\\/]', text[2:], maxsplit=2)
        return min(len(text), 2 + len(parts[0]) + (1 + len(parts[1]) if len(parts) > 1 else 0) + 1)
    if text.startswith(('/', '\\')):
        return 1
    return 0


def trim(value):
    """Без хвостового разделителя, но корень остаётся корнем."""
    source, text = split(value)
    keep = _root_length(text)
    body = text[keep:].rstrip('\\/')
    root = text[:keep]
    if _DRIVE.match(root) and len(root) == 2:
        root += '\\'
    result = root + body if (root or body) else text
    return f'{source}:{result}' if source else result


def is_root(value):
    source, text = split(trim(value))
    return len(text) <= _root_length(text) or not text


def name(value):
    """Последний элемент пути; у корня — сам корень."""
    source, text = split(trim(value))
    keep = _root_length(text)
    body = text[keep:]
    if not body:
        return text
    return re.split(r'[\\/]', body)[-1]


def parent(value):
    """Папка, в которой лежит элемент. У корня родителя нет — он сам."""
    trimmed = trim(value)
    source, text = split(trimmed)
    keep = _root_length(text)
    body = text[keep:]
    if not body:
        return trimmed
    cut = max(body.rfind('\\'), body.rfind('/'))
    result = text[:keep] + (body[:cut] if cut > 0 else '')
    if not result:
        result = text[:keep] or '.'
    return f'{source}:{result}' if source else result


def join(value, *names):
    result = trim(value)
    separator = sep_of(result)
    for item in names:
        item = str(item).strip('\\/')
        if not item:
            continue
        result = result + item if result.endswith(('\\', '/')) or result.endswith(':') \
            else result + separator + item
    return result


def chain(value):
    """Цепочка папок от корня до самой папки включительно."""
    items, current = [], trim(value)
    while True:
        items.append(current)
        up = parent(current)
        if up == current:
            break
        current = up
    items.reverse()
    return items


def prefix(folder):
    """Начало путей всего, что лежит внутри папки."""
    trimmed = trim(folder)
    return trimmed if trimmed.endswith(('\\', '/')) else trimmed + sep_of(trimmed)


def bounds(folder):
    """Диапазон путей внутри папки для сравнения строк в SQL (path>=? AND path<?)."""
    start = prefix(folder)
    return start, start + '\uffff'


def inside(value, folder):
    value, folder = trim(value), trim(folder)
    return value == folder or value.startswith(prefix(folder))


def suffix(value):
    item = name(value)
    dot = item.rfind('.')
    return item[dot:].lower() if dot > 0 else ''


def stem(value):
    item = name(value)
    dot = item.rfind('.')
    return item[:dot] if dot > 0 else item


def normalize_arg(value):
    """Путь из командной строки этапа: ключ как есть, локальный путь — абсолютным."""
    value = str(value)
    if is_key(value):
        return trim(value)
    return os.path.abspath(value)


# Доля задания на этом ядре при параллельной обработке: «номер/всего». Ставит
# device_job для этапов после описи; хаб делит одно задание между ядрами.
SHARD_ENV = 'HOMECLOUD_SHARD'


def shard():
    """(номер, всего) из HOMECLOUD_SHARD, а без деления — None."""
    match = re.fullmatch(r'(\d+)/(\d+)', os.environ.get(SHARD_ENV, '').strip())
    if not match:
        return None
    index, count = int(match.group(1)), int(match.group(2))
    return (index, count) if 0 <= index < count and count > 1 else None


def shard_sql(column='photos.path'):
    """Условие «файл из доли этого ядра»: делим по rowid снимка — ровно и без хэшей в SQL."""
    part = shard()
    if part is None:
        return '', []
    return (f' AND {column} IN (SELECT path FROM photos WHERE photos.rowid % ? = ?)',
            [part[1], part[0]])


def scope_sql(roots=(), paths=(), column='photos.path'):
    """Условие «снимок внутри выбранных папок или среди выбранных файлов» (и доли ядра)."""
    where, values = _scope_sql(roots, paths, column)
    part, part_values = shard_sql(column)
    return where + part, values + part_values


def _scope_sql(roots, paths, column):
    parts, values = [], []
    for root in roots or ():
        low, high = bounds(normalize_arg(root))
        parts.append(f'({column}=? OR ({column}>=? AND {column}<?))')
        values.extend((trim(normalize_arg(root)), low, high))
    for path in paths or ():
        parts.append(f'{column}=?')
        values.append(normalize_arg(path))
    if not parts:
        return '', []
    return ' AND (' + ' OR '.join(parts) + ')', values
