"""Метаданные файла для просмотрщика: EXIF, GPS, IPTC/XMP у снимков, параметры потока у роликов.

Читается по запросу и только заголовок файла — пиксели не декодируются, поэтому
это дёшево даже для больших снимков. Ответ уже разложен по группам и
подписан по-русски: интерфейсу остаётся только нарисовать.
"""
import math
from pathlib import Path

import video as video_media

# Служебное и огромное — показывать нечего, а весит килобайты.
SKIPPED_TAGS = {
    'MakerNote', 'PrintImageMatching', 'ExifOffset', 'GPSInfo', 'InteroperabilityOffset',
    'JPEGInterchangeFormat', 'JPEGInterchangeFormatLength', 'StripOffsets', 'StripByteCounts',
    'TileOffsets', 'TileByteCounts', 'ComponentsConfiguration', 'XMLPacket', 'InterColorProfile',
    'ExifInteroperabilityOffset', 'SubsecTime', 'SubsecTimeDigitized',
}
MAX_VALUE_LENGTH = 300

# Подписи и группы для известных полей; остальные идут в «Прочее» под своим именем.
LABELS = {
    # съёмка
    'DateTimeOriginal': ('shot', 'Снято'),
    'DateTimeDigitized': ('shot', 'Оцифровано'),
    'DateTime': ('shot', 'Изменено'),
    'OffsetTimeOriginal': ('shot', 'Часовой пояс съёмки'),
    'SubsecTimeOriginal': ('shot', 'Доли секунды'),
    'ExposureTime': ('shot', 'Выдержка'),
    'FNumber': ('shot', 'Диафрагма'),
    'ISOSpeedRatings': ('shot', 'ISO'),
    'PhotographicSensitivity': ('shot', 'ISO'),
    'FocalLength': ('shot', 'Фокусное расстояние'),
    'FocalLengthIn35mmFilm': ('shot', 'Эквивалент 35 мм'),
    'ExposureBiasValue': ('shot', 'Экспокоррекция'),
    'ExposureProgram': ('shot', 'Программа экспозиции'),
    'ExposureMode': ('shot', 'Режим экспозиции'),
    'MeteringMode': ('shot', 'Замер экспозиции'),
    'Flash': ('shot', 'Вспышка'),
    'WhiteBalance': ('shot', 'Баланс белого'),
    'LightSource': ('shot', 'Источник света'),
    'SceneCaptureType': ('shot', 'Тип сцены'),
    'DigitalZoomRatio': ('shot', 'Цифровой зум'),
    'SubjectDistance': ('shot', 'Расстояние до объекта'),
    'BrightnessValue': ('shot', 'Яркость (APEX)'),
    'ShutterSpeedValue': ('shot', 'Выдержка (APEX)'),
    'ApertureValue': ('shot', 'Диафрагма (APEX)'),
    'MaxApertureValue': ('shot', 'Макс. диафрагма (APEX)'),
    # камера
    'Make': ('camera', 'Производитель'),
    'Model': ('camera', 'Модель'),
    'LensMake': ('camera', 'Производитель объектива'),
    'LensModel': ('camera', 'Объектив'),
    'LensSpecification': ('camera', 'Характеристики объектива'),
    'BodySerialNumber': ('camera', 'Серийный номер'),
    'Software': ('camera', 'Программа'),
    'HostComputer': ('camera', 'Устройство'),
    'Artist': ('camera', 'Автор'),
    'Copyright': ('camera', 'Авторские права'),
    'ImageDescription': ('camera', 'Описание'),
    'UserComment': ('camera', 'Комментарий'),
    'ImageUniqueID': ('camera', 'ID снимка'),
    # изображение
    'Orientation': ('image', 'Ориентация'),
    'ExifImageWidth': ('image', 'Ширина по EXIF'),
    'ExifImageHeight': ('image', 'Высота по EXIF'),
    'ImageWidth': ('image', 'Ширина по EXIF'),
    'ImageLength': ('image', 'Высота по EXIF'),
    'XResolution': ('image', 'Разрешение по X'),
    'YResolution': ('image', 'Разрешение по Y'),
    'ResolutionUnit': ('image', 'Единица разрешения'),
    'ColorSpace': ('image', 'Цветовое пространство'),
    'YCbCrPositioning': ('image', 'Позиция YCbCr'),
    'ExifVersion': ('image', 'Версия EXIF'),
    'FlashPixVersion': ('image', 'Версия FlashPix'),
    'SensingMethod': ('image', 'Тип сенсора'),
    'Contrast': ('image', 'Контраст'),
    'Saturation': ('image', 'Насыщенность'),
    'Sharpness': ('image', 'Резкость'),
}
GROUP_TITLES = {
    'shot': 'Съёмка', 'camera': 'Камера', 'place': 'Место', 'image': 'Изображение',
    'format': 'Формат', 'video': 'Видеопоток', 'text': 'Встроенный текст', 'other': 'Прочее',
}
GROUP_ORDER = ('shot', 'camera', 'place', 'image', 'format', 'video', 'text', 'other')

ORIENTATION = {1: 'Обычная', 2: 'Отражена по горизонтали', 3: 'Повёрнута на 180°',
               4: 'Отражена по вертикали', 5: 'Отражена и повёрнута на 90° влево',
               6: 'Повёрнута на 90° вправо', 7: 'Отражена и повёрнута на 90° вправо',
               8: 'Повёрнута на 90° влево'}
EXPOSURE_PROGRAM = {0: 'Не указана', 1: 'Ручная', 2: 'Автоматическая', 3: 'Приоритет диафрагмы',
                    4: 'Приоритет выдержки', 5: 'Творческая', 6: 'Спорт', 7: 'Портрет',
                    8: 'Пейзаж'}
METERING = {0: 'Неизвестно', 1: 'Средний', 2: 'Центровзвешенный', 3: 'Точечный',
            4: 'Многоточечный', 5: 'Матричный', 6: 'Частичный', 255: 'Другой'}
WHITE_BALANCE = {0: 'Авто', 1: 'Ручной'}
EXPOSURE_MODE = {0: 'Авто', 1: 'Ручной', 2: 'Брекетинг'}
SCENE = {0: 'Стандартная', 1: 'Пейзаж', 2: 'Портрет', 3: 'Ночная'}
COLOR_SPACE = {1: 'sRGB', 2: 'Adobe RGB', 65535: 'Не откалибровано'}
RESOLUTION_UNIT = {1: 'Без единиц', 2: 'Дюйм', 3: 'Сантиметр'}
LEVEL = {0: 'Обычный', 1: 'Слабый', 2: 'Сильный'}
LIGHT_SOURCE = {0: 'Неизвестно', 1: 'Дневной свет', 2: 'Люминесцентный', 3: 'Лампа накаливания',
                4: 'Вспышка', 9: 'Ясно', 10: 'Облачно', 11: 'Тень', 255: 'Другой'}
SENSING = {1: 'Не указан', 2: 'Однокристальный цветной', 3: 'Двухкристальный',
           4: 'Трёхкристальный', 5: 'Последовательный', 7: 'Трилинейный', 8: 'Линейный'}
ENUMS = {
    'Orientation': ORIENTATION, 'ExposureProgram': EXPOSURE_PROGRAM, 'MeteringMode': METERING,
    'WhiteBalance': WHITE_BALANCE, 'ExposureMode': EXPOSURE_MODE, 'SceneCaptureType': SCENE,
    'ColorSpace': COLOR_SPACE, 'ResolutionUnit': RESOLUTION_UNIT, 'Contrast': LEVEL,
    'Saturation': LEVEL, 'Sharpness': LEVEL, 'LightSource': LIGHT_SOURCE, 'SensingMethod': SENSING,
    'YCbCrPositioning': {1: 'По центру', 2: 'Совмещённая'},
}


def _number(value):
    """IFDRational, Fraction, пара (числитель, знаменатель) или число → float."""
    try:
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, int) for x in value):
            return value[0] / value[1] if value[1] else None
        result = float(value)
        return None if math.isnan(result) or math.isinf(result) else result
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _trim(number, digits=2):
    if digits <= 0:
        return str(round(number))
    return f'{number:.{digits}f}'.rstrip('0').rstrip('.')


DATE_TAGS = {'DateTimeOriginal', 'DateTimeDigitized', 'DateTime'}


def _date(value):
    """«2019:11:12 17:36:09» → «12.11.2019 17:36:09»."""
    text = _text(value)
    if not text or len(text) < 19 or text[4] != ':' or text[7] != ':':
        return text
    return f'{text[8:10]}.{text[5:7]}.{text[:4]} {text[11:19]}'


def _text(value):
    if isinstance(value, bytes):
        # UserComment начинается с восьми байт кодировки.
        for prefix, codec in ((b'ASCII\x00\x00\x00', 'ascii'), (b'UNICODE\x00', 'utf-16'),
                              (b'\x00' * 8, 'utf-8')):
            if value.startswith(prefix):
                value = value[8:].decode(codec, errors='replace')
                break
        else:
            if len(value) <= 8 and all(32 <= byte < 127 for byte in value):
                value = value.decode('ascii')
            elif not all(32 <= byte < 127 or byte in (9, 10, 13) for byte in value):
                return None if len(value) > 64 else value.hex(' ')
            else:
                value = value.decode('latin-1')
    text = str(value).replace('\x00', '').strip()
    if len(text) > MAX_VALUE_LENGTH:
        text = text[:MAX_VALUE_LENGTH] + '…'
    return text or None


def _flash(value):
    try:
        bits = int(value)
    except (TypeError, ValueError):
        return _text(value)
    fired = 'сработала' if bits & 1 else 'не сработала'
    extra = []
    if (bits >> 3) & 3 == 3:
        extra.append('авто')
    elif (bits >> 3) & 3 == 2:
        extra.append('выключена')
    if bits & 0x40:
        extra.append('подавление красных глаз')
    return fired + (f' ({", ".join(extra)})' if extra else '')


def format_value(name, value):
    """Значение поля EXIF в человеческом виде; None — показывать нечего."""
    if value is None:
        return None
    if name in ENUMS:
        try:
            return ENUMS[name].get(int(value), str(value))
        except (TypeError, ValueError):
            pass
    if name == 'Flash':
        return _flash(value)
    if name in DATE_TAGS:
        return _date(value)
    if name == 'SceneType':
        return 'Снято камерой напрямую' if value in (1, b'\x01', '\x01') else _text(value)
    if name == 'ExposureTime':
        seconds = _number(value)
        if not seconds:
            return None
        if seconds >= 1:
            return f'{_trim(seconds, 1)} с'
        return f'1/{round(1 / seconds)} с'
    if name == 'FNumber':
        number = _number(value)
        return f'f/{_trim(number, 1)}' if number else None
    if name in ('FocalLength', 'FocalLengthIn35mmFilm'):
        number = _number(value)
        return f'{_trim(number, 1)} мм' if number else None
    if name == 'ExposureBiasValue':
        number = _number(value)
        return None if number is None else f'{number:+.1f} EV'.replace('+0.0', '0')
    if name == 'DigitalZoomRatio':
        number = _number(value)
        return f'×{_trim(number)}' if number else None
    if name == 'SubjectDistance':
        number = _number(value)
        return f'{_trim(number)} м' if number else None
    if name in ('ExifVersion', 'FlashPixVersion') and isinstance(value, bytes):
        text = value.decode('ascii', errors='replace')
        return f'{text[:2].lstrip("0") or "0"}.{text[2:]}'
    if name == 'LensSpecification' and isinstance(value, (tuple, list)) and len(value) == 4:
        low, high, f_low, f_high = (_number(item) for item in value)
        parts = []
        if low:
            parts.append(f'{_trim(low)}–{_trim(high)} мм' if high and high != low else f'{_trim(low)} мм')
        if f_low:
            parts.append(f'f/{_trim(f_low, 1)}' + (f'–{_trim(f_high, 1)}' if f_high and f_high != f_low else ''))
        return ', '.join(parts) or None
    if isinstance(value, (tuple, list)):
        items = [format_value('', item) for item in value[:16]]
        items = [item for item in items if item is not None]
        return ', '.join(items) if items else None
    if isinstance(value, bytes):
        return _text(value)
    number = _number(value) if not isinstance(value, (str, int)) else None
    if number is not None:
        return _trim(number, 4)
    return _text(value)


def _gps_degrees(parts, ref):
    try:
        degrees, minutes, seconds = (_number(item) for item in parts)
        result = degrees + minutes / 60 + seconds / 3600
    except (TypeError, ValueError):
        return None
    return -result if str(ref or '').upper().startswith(('S', 'W')) else result


def gps_items(gps):
    """Координаты одной строкой плюс высота, направление и время по GPS."""
    items = []
    latitude = _gps_degrees(gps.get(2), gps.get(1)) if gps.get(2) else None
    longitude = _gps_degrees(gps.get(4), gps.get(3)) if gps.get(4) else None
    coords = None
    if latitude is not None and longitude is not None and not (abs(latitude) < 1e-9 and abs(longitude) < 1e-9):
        coords = {'latitude': round(latitude, 6), 'longitude': round(longitude, 6)}
        items.append(('GPSCoordinates', 'Координаты', f'{latitude:.6f}, {longitude:.6f}'))
    altitude = _number(gps.get(6)) if gps.get(6) is not None else None
    if altitude is not None:
        below = gps.get(5) in (1, b'\x01')
        items.append(('GPSAltitude', 'Высота', f'{"−" if below else ""}{_trim(altitude, 1)} м'))
    direction = _number(gps.get(17)) if gps.get(17) is not None else None
    if direction is not None:
        items.append(('GPSImgDirection', 'Направление съёмки', f'{_trim(direction, 1)}°'))
    speed = _number(gps.get(13)) if gps.get(13) is not None else None
    if speed:
        unit = {'K': 'км/ч', 'M': 'миль/ч', 'N': 'уз'}.get(str(gps.get(12) or 'K'), '')
        items.append(('GPSSpeed', 'Скорость', f'{_trim(speed, 1)} {unit}'.strip()))
    stamp, date = gps.get(7), gps.get(29)
    if date or stamp:
        clock = ''
        if isinstance(stamp, (tuple, list)) and len(stamp) == 3:
            try:
                clock = ':'.join(f'{int(_number(part) or 0):02d}' for part in stamp)
            except (TypeError, ValueError):
                clock = ''
        items.append(('GPSDateTime', 'Время по GPS (UTC)', ' '.join(filter(None, [_text(date), clock]))))
    method = gps.get(27)
    if method:
        items.append(('GPSProcessingMethod', 'Источник координат', _text(method)))
    return items, coords


def _item(key, label, value):
    return {'key': key, 'label': label, 'value': value}


def photo_metadata(path):
    """Группы метаданных снимка. Бросает OSError, если файл не открыть."""
    from PIL import ExifTags, Image
    groups = {name: [] for name in GROUP_ORDER}
    coords = None
    with Image.open(path) as image:
        groups['format'].extend(filter(None, [
            _item('Format', 'Формат', image.format_description or image.format),
            _item('Mode', 'Цветовая модель', image.mode),
            _item('Size', 'Размер в пикселях', f'{image.width}×{image.height}'),
            _item('Megapixels', 'Мегапиксели', _trim(image.width * image.height / 1e6, 1)),
            _item('Frames', 'Кадров', str(getattr(image, 'n_frames', 1)))
            if getattr(image, 'n_frames', 1) > 1 else None,
            _item('ICC', 'Цветовой профиль', _icc_name(image.info.get('icc_profile')))
            if image.info.get('icc_profile') else None,
            _item('Progressive', 'Прогрессивный JPEG', 'да') if image.info.get('progressive') else None,
            _item('DPI', 'DPI', '×'.join(_trim(_number(v) or 0, 0) for v in image.info['dpi']))
            if image.info.get('dpi') else None,
        ]))
        exif = image.getexif()
        seen = set()

        def add(tag_id, value, names=ExifTags.TAGS):
            name = names.get(tag_id, f'Tag0x{tag_id:04X}')
            if name in SKIPPED_TAGS or name in seen:
                return
            formatted = format_value(name, value)
            if formatted is None or formatted == '':
                return
            seen.add(name)
            group, label = LABELS.get(name, ('other', name))
            groups[group].append(_item(name, label, formatted))

        for tag_id, value in exif.items():
            add(tag_id, value)
        # Interop-раздел служебный (версия и индекс совместимости) — не показываем.
        try:
            for tag_id, value in exif.get_ifd(ExifTags.IFD.Exif).items():
                add(tag_id, value)
        except Exception:
            pass
        try:
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except Exception:
            gps = {}
        if gps:
            items, coords = gps_items(gps)
            groups['place'].extend(_item(*item) for item in items if item[2])
        # PNG, WebP и прочие хранят текст в info: подписи генераторов, комментарии.
        for key, value in image.info.items():
            if key in ('exif', 'icc_profile', 'dpi', 'progressive', 'progression', 'jfif',
                       'jfif_version', 'jfif_unit', 'jfif_density', 'adobe', 'adobe_transform',
                       'transparency', 'gamma', 'duration', 'loop', 'background', 'mp',
                       'photoshop', 'xmp', 'XML:com.adobe.xmp'):
                continue
            text = _text(value) if isinstance(value, (str, bytes)) else None
            if text:
                groups['text'].append(_item(f'info:{key}', str(key), text))
        xmp = image.info.get('xmp') or image.info.get('XML:com.adobe.xmp')
        if xmp:
            for key, label, value in _xmp_items(xmp):
                groups['other'].append(_item(key, label, value))
    return _result(groups, coords)


def _icc_name(profile):
    try:
        from io import BytesIO
        from PIL import ImageCms
        return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(profile))).strip() or 'встроен'
    except Exception:
        return 'встроен'


def _xmp_items(raw):
    """Несколько полезных полей XMP: рейтинг, ключевые слова, программа."""
    import re
    text = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else str(raw)
    found = []
    for key, label in (('xmp:Rating', 'Рейтинг XMP'), ('xmp:CreatorTool', 'Программа XMP'),
                       ('photoshop:DateCreated', 'Создано (XMP)'), ('dc:title', 'Заголовок XMP'),
                       ('dc:description', 'Описание XMP'), ('dc:subject', 'Ключевые слова XMP')):
        match = re.search(rf'{key}="([^"]+)"', text) or re.search(
            rf'<{key}>(.*?)</{key}>', text, re.S)
        if match:
            value = re.sub(r'<[^>]+>', ' ', match.group(1))
            value = re.sub(r'\s+', ' ', value).strip()
            if value:
                found.append((key, label, value[:MAX_VALUE_LENGTH]))
    return found


def video_metadata(path):
    """Параметры ролика, доступные OpenCV без внешних программ."""
    import cv2
    groups = {name: [] for name in GROUP_ORDER}
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError('Ролик не открылся')
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = ''.join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip('\x00 ')
        bitrate = capture.get(getattr(cv2, 'CAP_PROP_BITRATE', -1)) if hasattr(cv2, 'CAP_PROP_BITRATE') else 0
        rotation = capture.get(getattr(cv2, 'CAP_PROP_ORIENTATION_META', -1)) \
            if hasattr(cv2, 'CAP_PROP_ORIENTATION_META') else 0
    finally:
        capture.release()
    groups['format'].append(_item('Container', 'Контейнер', Path(path).suffix.lstrip('.').upper()))
    video = groups['video']
    if width and height:
        video.append(_item('Size', 'Размер кадра', f'{width}×{height}'))
    if codec and codec.isprintable():
        video.append(_item('Codec', 'Кодек', codec))
    if fps:
        video.append(_item('FPS', 'Кадров в секунду', _trim(fps, 2)))
    if frames > 0:
        video.append(_item('Frames', 'Кадров', f'{frames:,}'.replace(',', ' ')))
        if fps:
            video.append(_item('Duration', 'Длительность по потоку', f'{_trim(frames / fps, 1)} с'))
    if bitrate and bitrate > 0:
        video.append(_item('Bitrate', 'Битрейт', f'{round(bitrate)} кбит/с'))
    if rotation:
        video.append(_item('Rotation', 'Поворот', f'{_trim(rotation, 0)}°'))
    return _result(groups, None)


def _result(groups, coords):
    order = {name: index for index, name in enumerate(LABELS)}
    for name, items in groups.items():
        # Одинаковая подпись с тем же значением (ImageWidth и ExifImageWidth) — одна строка.
        unique, seen = [], set()
        for item in items:
            mark = (item['label'], item['value'])
            if mark not in seen:
                seen.add(mark)
                unique.append(item)
        if name not in ('format', 'video', 'place', 'text'):
            unique.sort(key=lambda item: order.get(item['key'], len(order)))
        groups[name] = unique
    return {
        'groups': [{'id': name, 'title': GROUP_TITLES[name], 'items': groups[name]}
                   for name in GROUP_ORDER if groups[name]],
        'coords': coords,
    }


def read(path):
    """Метаданные по виду файла."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Файл не найден на диске')
    if video_media.is_video(str(path)):
        return {'kind': 'video', **video_metadata(path)}
    return {'kind': 'photo', **photo_metadata(path)}
