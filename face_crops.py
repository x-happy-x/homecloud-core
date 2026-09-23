"""Миниатюра лица с полями: квадрат вокруг рамки, а не сама рамка.

Рамка детектора режет по лбу и подбородку, и в сетке лица выходили слишком
крупными. Теперь сторона квадрата — большая сторона рамки, умноженная на
`SCALE`, и квадрат сдвигается внутрь кадра, если упёрся в край. Резкость
(`face_quality`) по-прежнему меряется по самому лицу: при детекции — по
тесной вырезке, а по готовой миниатюре — по её середине (`face_part`).

Миниатюра с полями узнаётся по имени файла (`SUFFIX`): старые остаются
тесными, пока этап «Лица» не переобрежет их из оригинала (`recut`).
"""
import hashlib
import json

SCALE = 1.6
SUFFIX = '-p.jpg'
SIZE = 160


def padded(name):
    return bool(name) and str(name).endswith(SUFFIX)


def name_for(token):
    return f'thumbnails/{token}{SUFFIX[:-4]}.jpg'


def square(box, width, height, scale=SCALE):
    """Квадрат вокруг рамки в пределах кадра: (left, top, right, bottom)."""
    left, top, right, bottom = (float(value) for value in box[:4])
    side = min(max(right - left, bottom - top) * scale, width, height)
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    x = min(max(center_x - side / 2, 0), width - side)
    y = min(max(center_y - side / 2, 0), height - side)
    return int(x), int(y), int(round(x + side)), int(round(y + side))


def cut(image, box, size=SIZE):
    """PIL-картинка → миниатюра лица с полями."""
    crop = image.crop(square(box, image.width, image.height))
    crop.thumbnail((size, size))
    return crop


def face_part(gray):
    """Середина миниатюры с полями — примерно сама рамка лица."""
    height, width = gray.shape[:2]
    margin_y = int(height * (1 - 1 / SCALE) / 2)
    margin_x = int(width * (1 - 1 / SCALE) / 2)
    return gray[margin_y:height - margin_y, margin_x:width - margin_x]


def recut(db, data, paths, open_image, publish=None, stop=None):
    """Переобрезать тесные миниатюры лиц этих файлов из оригиналов.

    `open_image(key, moment)` отдаёт PIL-картинку снимка или кадр ролика на
    секунде `moment` (None — файла нет). Возвращает число новых миниатюр.
    """
    import catalogfiles
    wanted = {}
    keys = [str(path) for path in paths]
    for offset in range(0, len(keys), 500):
        batch = keys[offset:offset + 500]
        for face_id, key, raw_box, moment, thumbnail in db.execute(
                'SELECT id,path,box,frame_time,thumbnail FROM faces '
                f'WHERE path IN ({",".join("?" * len(batch))}) AND box IS NOT NULL',
                batch):
            if not padded(thumbnail):
                wanted.setdefault(key, []).append((face_id, raw_box, moment))
    done = 0
    for number, (key, faces) in enumerate(wanted.items()):
        if stop and stop():
            break
        if publish:
            publish(key, number, len(wanted))
        frames = {}
        updates = []
        for face_id, raw_box, moment in faces:
            try:
                box = json.loads(raw_box)
                if moment not in frames:
                    frames[moment] = open_image(key, moment)
                image = frames[moment]
                if image is None or len(box) < 4:
                    continue
                token = hashlib.sha256(f'{key}:{face_id}:{raw_box}'.encode()).hexdigest()
                thumbnail = name_for(token)
                cut(image, box).save(data / thumbnail)
                catalogfiles.publish(data, thumbnail)
                updates.append((thumbnail, face_id))
            except Exception as exc:
                print(f'  {key}: миниатюра лица {face_id} не переобрезана: {exc}', flush=True)
        if updates:
            with db:
                db.executemany('UPDATE faces SET thumbnail=? WHERE id=?', updates)
            done += len(updates)
    return done
