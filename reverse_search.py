"""Временная публичная ссылка на снимок — для поиска по картинке.

HomeCloud принципиально живёт только в домашней сети, поэтому у файла нет
адреса, который можно отдать Яндексу или Google для поиска по URL. Здесь
снимок на время заливается на Litterbox (временный раздел catbox.moe):
никакой регистрации, случайная неугадываемая ссылка, файл автоматически
стирается по истечении срока.

Из проверенных на месте вариантов основным был Litterbox. Если он временно
отваливается, используем Uguu: он тоже возвращает прямую ссылку на байты
изображения, а не HTML-страницу.

У анонимных заливок в Litterbox нет способа удалить их раньше срока — только
дождаться истечения. Самый короткий срок из тех, что принимает сервис, — час;
это и используется, хотя пользователь и просил «пару минут».
"""
import mimetypes
import json
import ssl
import uuid
from urllib.request import Request, urlopen

UPLOAD_URL = 'https://litterbox.catbox.moe/resources/internals/api.php'
UGUU_UPLOAD_URL = 'https://uguu.se/upload.php'
# Меньше сервис не принимает — это минимальный срок жизни анонимной заливки.
EXPIRES = '1h'

# На этой машине что-то подменяет TLS-сертификат внешних сайтов (антивирус
# с проверкой HTTPS или похожее) — обычная проверка виснет на рукопожатии.
# Пользователь явно попросил всё равно отключить проверку для этой функции;
# больше нигде в приложении сертификаты так не ослабляются.
_UNVERIFIED_CONTEXT = ssl.create_default_context()
_UNVERIFIED_CONTEXT.check_hostname = False
_UNVERIFIED_CONTEXT.verify_mode = ssl.CERT_NONE


def _multipart(fields, files):
    """Собирает тело multipart/form-data без внешних зависимостей."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            .encode('utf-8'))
    for name, (filename, content, content_type) in files.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode('utf-8')
            + content + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode('utf-8'))
    return b''.join(parts), f'multipart/form-data; boundary={boundary}'


def _upload_litterbox(image_bytes, filename, timeout):
    content_type = mimetypes.guess_type(filename)[0] or 'image/jpeg'
    body, content_header = _multipart(
        {'reqtype': 'fileupload', 'time': EXPIRES},
        {'fileToUpload': (filename, image_bytes, content_type)})
    request = Request(UPLOAD_URL, data=body, headers={
        'Content-Type': content_header,
        'User-Agent': 'HomeCloud/1.0 (private reverse-image-search helper)',
    })
    with urlopen(request, timeout=timeout, context=_UNVERIFIED_CONTEXT) as response:
        url = response.read().decode('utf-8').strip()
    if not url.startswith('http'):
        raise RuntimeError(f'Litterbox ответил неожиданно: {url[:200]}')
    return url


def _upload_uguu(image_bytes, filename, timeout):
    content_type = mimetypes.guess_type(filename)[0] or 'image/jpeg'
    body, content_header = _multipart(
        {},
        {'files[]': (filename, image_bytes, content_type)})
    request = Request(UGUU_UPLOAD_URL, data=body, headers={
        'Content-Type': content_header,
        'User-Agent': 'HomeCloud/1.0 (private reverse-image-search helper)',
    })
    with urlopen(request, timeout=timeout, context=_UNVERIFIED_CONTEXT) as response:
        data = response.read().decode('utf-8')
    try:
        payload = json.loads(data)
        url = payload['files'][0]['url'].replace('\\/', '/')
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f'Uguu ответил неожиданно: {data[:200]}') from exc
    if not url.startswith('http'):
        raise RuntimeError(f'Uguu ответил неожиданно: {url[:200]}')
    return url


def upload(image_bytes, filename='photo.jpg', timeout=30):
    """Заливает картинку во временное хранилище, возвращает прямую ссылку."""
    errors = []
    for name, uploader in (('Litterbox', _upload_litterbox), ('Uguu', _upload_uguu)):
        try:
            return uploader(image_bytes, filename, timeout)
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    raise RuntimeError('; '.join(errors))
