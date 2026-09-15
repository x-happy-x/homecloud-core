"""Возвращает имена лицам, потерянным при повторном сканировании.

Повторный проход раньше удалял лица фотографии и создавал их заново, а вместе с
ними пропадала привязка к человеку. Точной записи «какое лицо чьё» не осталось,
поэтому имена восстанавливаются по векторам: уцелевшие лица человека служат
образцами, к ним притягиваются похожие безымянные лица.

    python restore_names.py --data trash-clean-catalog            # только показать
    python restore_names.py --data trash-clean-catalog --apply    # записать
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from people_gui import CatalogStore


def vectors_of(store, face_ids):
    found = {}
    for offset in range(0, len(face_ids), 900):
        batch = face_ids[offset:offset + 900]
        places = ','.join('?' * len(batch))
        for face_id, blob in store.db.execute(
                f'SELECT id,embedding FROM faces WHERE id IN ({places})', batch):
            vector = np.frombuffer(blob, dtype='<f4').astype('<f4')
            found[face_id] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    ordered = [face_id for face_id in face_ids if face_id in found]
    return ordered, (np.stack([found[face_id] for face_id in ordered])
                     if ordered else np.zeros((0, 512), dtype='<f4'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.62,
                        help='Минимальная близость к образцам человека')
    parser.add_argument('--margin', type=float, default=0.05,
                        help='Насколько лучший человек должен опережать следующего')
    parser.add_argument('--apply', action='store_true', help='Записать в каталог')
    args = parser.parse_args()

    store = CatalogStore(args.data, thread_safe=True)
    people = {row[0]: (row[1], row[2]) for row in store.db.execute(
        'SELECT id,name,bigfam_id FROM people')}
    seeds = {}
    for person_id, face_id in store.db.execute('SELECT person_id,face_id FROM face_people'):
        seeds.setdefault(person_id, []).append(face_id)
    seeds = {person_id: ids for person_id, ids in seeds.items() if ids}
    if not seeds:
        print('Ни у кого не осталось образцов — восстанавливать не от чего.')
        return 1

    assigned = {face_id for ids in seeds.values() for face_id in ids}
    excluded = {row[0] for row in store.db.execute('SELECT face_id FROM face_exclusions')}
    candidates = [row[0] for row in store.rows
                  if row[0] not in assigned and row[0] not in excluded]
    print(f'Образцов: {len(assigned)} у {len(seeds)} человек · безымянных лиц: {len(candidates)}')
    if not candidates:
        return 0

    order = sorted(seeds)
    samples = {person_id: vectors_of(store, seeds[person_id])[1] for person_id in order}
    found, matrix = vectors_of(store, candidates)
    # Для каждого лица — близость к лучшему образцу каждого человека.
    scores = np.stack([(matrix @ samples[person_id].T).max(axis=1)
                       if samples[person_id].size else np.full(len(found), -1.0)
                       for person_id in order], axis=1)

    picked = {}
    for index, face_id in enumerate(found):
        row = scores[index]
        best = int(row.argmax())
        rest = np.delete(row, best)
        second = float(rest.max()) if rest.size else -1.0
        if float(row[best]) >= args.threshold and float(row[best]) - second >= args.margin:
            picked.setdefault(order[best], []).append((face_id, float(row[best])))

    total = 0
    for person_id, items in sorted(picked.items(), key=lambda item: -len(item[1])):
        name = people[person_id][0]
        best = max(score for _, score in items)
        worst = min(score for _, score in items)
        print(f'  {name}: +{len(items)} (близость {worst:.2f}…{best:.2f}), было {len(seeds[person_id])}')
        total += len(items)
    print(f'Итого к возврату: {total}')
    if not args.apply:
        print('Это только показ. Добавьте --apply, чтобы записать.')
        return 0
    for person_id, items in picked.items():
        name, bigfam_id = people[person_id]
        store.assign([face_id for face_id, _ in items] + seeds[person_id], name, bigfam_id)
    print('Записано. В интерфейсе «Отменить последнее действие» откатывает по одному человеку.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
