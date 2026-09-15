"""Временная публичная ссылка на снимок — для поиска по картинке.

HomeCloud принципиально живёт только в домашней сети, поэтому у файла нет
адреса, который можно отдать Яндексу или Google для поиска по URL. Здесь
снимок на время заливается на Litterbox (временный раздел catbox.moe):
никакой регистрации, случайная неугадываемая ссылка, файл автоматически
стирается по истечении срока.

Из проверенных на месте вариантов (0x0.st, file.io, tmpfiles.org) рабочим
оказался только этот: 0x0.st недоступен с этой сети (похоже, в блок-листах как
типичный анонимный дамп), у file.io публичный анонимный приём файлов сейчас
не работает, а tmpfiles.org отдаёт HTML-страницу вместо самой картинки — для
поиска по URL нужна именно прямая ссылка на байты изображения.

У анонимных заливок в Litterbox нет способа удалить их раньше срока — только
дождаться истечения. Самый короткий срок из тех, что принимает сервис, — час;
это и используется, хотя пользователь и просил «пару минут».
"""
import mimetypes
import ssl
import uuid
from urllib.request import Request, urlopen

UPLOAD_URL = 'https://litterbox.catbox.moe/resources/internals/api.php'
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


def upload(image_bytes, filename='photo.jpg', timeout=30):
    """Заливает картинку на Litterbox, возвращает прямую ссылку на файл."""
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
