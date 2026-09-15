"""Видео в фототеке: раскладываем ролик на кадры для поиска лиц.

Отдельного декодера здесь нет — берём OpenCV, он уже стоит ради InsightFace.
Кадры выбираем равномерно по всей длительности: лица ищем в каждом, а время
кадра запоминаем вместе с лицом, чтобы потом показать, откуда оно.
"""
import ctypes
import os
import sys
from pathlib import Path

SUPPORTED = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm', '.3gp', '.3g2',
             '.mpg', '.mpeg', '.mts', '.m2ts', '.wmv', '.flv'}


def is_video(path):
    return Path(path).suffix.casefold() in SUPPORTED


def _open(path):
    """VideoCapture, с запасным вариантом короткого пути Windows."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened() or os.name != 'nt':
        return capture
    capture.release()
    buffer = ctypes.create_unicode_buffer(4096)
    if ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, 4096):
        return cv2.VideoCapture(buffer.value)
    return capture


def probe(path):
    """Длительность, размер кадра и частота — без раскодирования всего ролика."""
    import cv2
    capture = _open(path)
    try:
        if not capture.isOpened():
            raise OSError('не удалось открыть видео')
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frames / fps if fps > 0 and frames > 0 else 0.0
        return {'fps': fps, 'frames': int(frames), 'width': width, 'height': height,
                'duration': duration}
    finally:
        capture.release()


def _positions(info, count, start=0.0, stop=0.0):
    """Номера кадров, равномерно разложенные по ролику."""
    total = info['frames']
    fps = info['fps'] or 25.0
    first = int(max(0.0, start) * fps)
    last = int(min(stop, info['duration']) * fps) if stop else total
    last = min(last if last > 0 else total, total) or total
    if last <= first:
        first, last = 0, total or count
    span = max(1, last - first)
    if count >= span:
        return list(range(first, first + span))
    # Края ролика обычно бесполезны (титры, размытие), поэтому берём середины
    # равных отрезков, а не сами границы.
    step = span / count
    return [int(first + step * (index + 0.5)) for index in range(count)]


def frames(path, count=12, start=0.0, stop=0.0, stop_check=None):
    """Пары «время в секундах, кадр BGR» — ровно столько, сколько попросили."""
    import cv2
    info = probe(path)
    capture = _open(path)
    try:
        if not capture.isOpened():
            raise OSError('не удалось открыть видео')
        fps = info['fps'] or 25.0
        wanted = _positions(info, max(1, int(count)), start, stop)
        seekable = info['frames'] > 0
        if seekable:
            for position in wanted:
                if stop_check and stop_check():
                    return
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                yield position / fps, frame
            return
        # Битые заголовки: перемотка не работает, читаем подряд с шагом.
        wanted = set(wanted)
        index = 0
        while True:
            if stop_check and stop_check():
                return
            ok, frame = capture.read()
            if not ok or frame is None:
                return
            if index in wanted:
                yield index / fps, frame
            index += 1
    finally:
        capture.release()


def poster(path, at=0.0):
    """Один кадр для обложки: по умолчанию из начала, но не самый первый."""
    import cv2
    info = probe(path)
    capture = _open(path)
    try:
        if not capture.isOpened():
            raise OSError('не удалось открыть видео')
        fps = info['fps'] or 25.0
        moment = at if at > 0 else min(1.0, info['duration'] / 10 if info['duration'] else 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(moment * fps))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok or frame is None:
            raise OSError('не удалось прочитать кадр')
        return frame
    finally:
        capture.release()


def to_image(frame):
    """Кадр OpenCV (BGR) → картинка PIL."""
    from PIL import Image
    return Image.fromarray(frame[:, :, ::-1])


def open_frame(path):
    """Картинка для анализа: сам файл — для фото, характерный кадр — для видео.

    Визуальный индекс, OCR, описание и проверка 18+ написаны для одной
    картинки; чтобы не дублировать их для видео, здесь просто подменяется
    источник — дальше всё работает как с обычным файлом.
    """
    from PIL import Image, ImageOps
    if is_video(path):
        return to_image(poster(path))
    with Image.open(path) as original:
        return ImageOps.exif_transpose(original).convert('RGB')


def sample_images(path, count=3):
    """Несколько кадров ролика картинками — для проверок, где важно не
    пропустить кадр в середине (18+): одного кадра из начала мало."""
    if not is_video(path):
        return [open_frame(path)]
    info = probe(path)
    duration = info['duration'] or 0
    if not duration:
        return [to_image(poster(path))]
    moments = [duration * (index + 1) / (count + 1) for index in range(count)]
    images = []
    for moment in moments:
        try:
            images.append(to_image(poster(path, moment)))
        except OSError:
            continue
    return images or [to_image(poster(path))]


def main():
    """Ручная проверка: `python video.py путь [кадров]`."""
    path = Path(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print(probe(path))
    for moment, frame in frames(path, count):
        print(f'{moment:7.2f} с  {frame.shape}')


if __name__ == '__main__':
    main()
