"""Правила путей: что не трогать вовсе и что из этого всё-таки нужно.

Одни и те же правила читают сканер, этапы обработки и галерея, поэтому лежат
они в настройках каталога. Строка без звёздочек — это папка целиком, со
звёздочками — маска (`*` — любой кусок, `?` — один символ). Белый список
вырезает исключения из чёрного: `D:\\trash\\Apps` целиком мимо, но
`D:\\trash\\Apps\\Наши снимки` — нужен.
"""
import fnmatch

SEPARATORS = ('\\', '/')


def _lines(raw):
    return [item.strip() for item in str(raw or '').replace('\r', '').split('\n')
            if item.strip()]


def normalize(rule):
    """Правило в единый вид: нижний регистр и прямые слэши, без хвоста."""
    text = str(rule).strip().strip('"').replace('\\', '/').casefold()
    while text.endswith('/') and len(text) > 1:
        text = text[:-1]
    return text


def prepare(values):
    """Списки правил из настроек: (чёрный, белый)."""
    return ([normalize(rule) for rule in _lines((values or {}).get('block_paths'))],
            [normalize(rule) for rule in _lines((values or {}).get('allow_paths'))])


def _matches(path, rule):
    if '*' in rule or '?' in rule or '[' in rule:
        return fnmatch.fnmatchcase(path, rule)
    # Без масок правило — это сама папка и всё, что внутри неё.
    return path == rule or path.startswith(rule + '/')


def blocked(path, block, allow=()):
    """Исключён ли путь: чёрный список решает, белый — отменяет."""
    if not block:
        return False
    target = normalize(path)
    if not any(_matches(target, rule) for rule in block):
        return False
    return not any(_matches(target, rule) for rule in allow)


def filter_paths(paths, block, allow=()):
    return [path for path in paths if not blocked(path, block, allow)]


def enter(path, block=(), allow=()):
    """Стоит ли обходу заходить в эту папку.

    Заблокированная папка обычно режется целиком — но если внутри нее
    белый список открывает что-то более глубокое, спускаться всё равно надо:
    файлы и вложенные папки блокировка отфильтрует по отдельности дальше.
    """
    if not blocked(path, block, allow):
        return True
    target = normalize(path)
    return any(rule == target or rule.startswith(target + '/') for rule in allow)


def ensure_column(db):
    """Отметка «исключён» хранится в самом каталоге: так фильтр ничего не стоит."""
    if 'blocked' not in {row[1] for row in db.execute('PRAGMA table_info(photos)')}:
        db.execute('ALTER TABLE photos ADD COLUMN blocked INTEGER NOT NULL DEFAULT 0')
        db.commit()


def apply(db, values):
    """Пересчитывает отметки по правилам. SQLite не умеет ни кириллицу в LOWER,
    ни регистронезависимый GLOB, поэтому сверяем пути на стороне Python."""
    ensure_column(db)
    block, allow = prepare(values)
    marked = [(path,) for (path,) in db.execute('SELECT path FROM photos')
              if blocked(path, block, allow)]
    with db:
        db.execute('UPDATE photos SET blocked=0 WHERE blocked<>0')
        if marked:
            db.executemany('UPDATE photos SET blocked=1 WHERE path=?', marked)
    return len(marked)


def sql(column='photos'):
    """Условие «этот снимок не исключён правилами»."""
    return f' AND COALESCE({column}.blocked,0)=0'


def load(catalog):
    """Правила прямо из каталога — для отдельных скриптов этапов."""
    import settings as catalog_settings
    return prepare(catalog_settings.load(catalog))
