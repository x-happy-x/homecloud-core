"""Стопки похожих кадров в карточке человека.

В карточке группы одно и то же лицо часто повторяется десятками: серия
снимков подряд, копии одного файла по разным папкам, несколько треков из
одного ролика. Смотреть это плиткой за плиткой бессмысленно, поэтому такие
кадры складываются в стопку, а наверх кладётся самый резкий.

Что считается одной стопкой:

- **ролик** — все лица человека из одного файла: у каждого своё время, и
  внутри стопки они идут по порядку, но по отдельности они только шумят;
- **копии** — снимки с почти одинаковым dHash (до `COPY_BITS` бит разницы)
  из любых папок;
- **серия** — снимки из одной папки, либо почти одинаковые по dHash (до
  `SERIES_BITS`), либо снятые в пределах `SERIES_SECONDS` друг от друга, если
  и лица на них похожи (`SERIES_SIMILARITY`): серия — это один момент, а не
  просто соседние по времени кадры.

Модуль ничего не читает сам: на вход — уже собранные описания лиц.
"""
import numpy as np

COPY_BITS = 4
SERIES_BITS = 10
SERIES_SECONDS = 3.0
SERIES_SIMILARITY = 0.6
# Попарное сравнение хешей — квадрат по числу снимков; больше — только серии.
PAIRWISE_LIMIT = 4000


class _Union:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def _dhash_bits(values):
    """Хеши строкой из 16 шестнадцатеричных знаков → матрица бит (n × 64)."""
    packed = np.zeros((len(values), 8), dtype=np.uint8)
    known = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        try:
            packed[index] = np.frombuffer(bytes.fromhex(value), dtype=np.uint8)[:8]
            known[index] = True
        except (TypeError, ValueError):
            continue
    return np.unpackbits(packed, axis=1), known


def stack_faces(faces):
    """Разложить лица по стопкам.

    `faces` — список словарей в порядке показа: `id`, `path`, `kind`
    ('photo'/'video'), `folder`, `taken` (секунды или None), `dhash` (строка
    или None), `vector` (нормированный вектор лица или None), `blur` (0 —
    резко, 1 — мыло, или None).

    Возвращает `{id лица: id верхнего лица стопки}`; у одиночного лица это
    оно само.
    """
    count = len(faces)
    if not count:
        return {}
    union = _Union(count)

    by_video = {}
    for index, face in enumerate(faces):
        if face.get('kind') == 'video':
            by_video.setdefault(face['path'], []).append(index)
    for members in by_video.values():
        for index in members[1:]:
            union.join(members[0], index)

    photos = [index for index, face in enumerate(faces) if face.get('kind') != 'video']
    if len(photos) > 1 and len(photos) <= PAIRWISE_LIMIT:
        bits, known = _dhash_bits([faces[index].get('dhash') for index in photos])
        folders = [faces[index].get('folder') for index in photos]
        for row in range(len(photos)):
            if not known[row]:
                continue
            distance = np.count_nonzero(bits[row + 1:] != bits[row], axis=1)
            for offset in np.nonzero(distance <= SERIES_BITS)[0]:
                other = row + 1 + int(offset)
                if not known[other]:
                    continue
                same_folder = folders[row] == folders[other]
                if distance[offset] <= COPY_BITS or same_folder:
                    union.join(photos[row], photos[other])

    by_folder = {}
    for index in photos:
        if faces[index].get('taken') is not None:
            by_folder.setdefault(faces[index].get('folder'), []).append(index)
    for members in by_folder.values():
        members.sort(key=lambda index: faces[index]['taken'])
        for position, index in enumerate(members):
            for other in members[position + 1:]:
                if faces[other]['taken'] - faces[index]['taken'] > SERIES_SECONDS:
                    break
                first, second = faces[index].get('vector'), faces[other].get('vector')
                if (first is not None and second is not None
                        and float(np.dot(first, second)) >= SERIES_SIMILARITY):
                    union.join(index, other)

    groups = {}
    for index in range(count):
        groups.setdefault(union.find(index), []).append(index)
    result = {}
    for members in groups.values():
        # Наверх — самое резкое лицо; без оценки — первое по порядку показа.
        top = min(members, key=lambda index: (
            faces[index].get('blur') if faces[index].get('blur') is not None else 1.0, index))
        for index in members:
            result[faces[index]['id']] = faces[top]['id']
    return result
