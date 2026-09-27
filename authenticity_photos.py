"""Настоящее лицо или рисунок — отсекаем мультяшных и игровых персонажей
из автогрупп раньше, чем они туда попадут.

Детектор лиц (SCRFD) иногда находит «лицо» там, где это на самом деле
нарисованный персонаж, 3D-модель или игровой скриншот — в архиве со
скачанным не-семейным видео таких попадается много, и они забивают раздел
«Люди» мусорными автогруппами. `anime_real_cls` (deepghs, через пакет
`dghs-imgutils`) отличает рисованное лицо от настоящего фото с высокой
уверенностью — проверено на реальных данных каталога: 0.95–0.99 в обе
стороны на явных случаях.

Это не решает всю проблему ложных срабатываний: маска, пейзаж, размытие —
не рисунок, а просто не лицо, такую путаницу эта модель не ловит и не
обязана. Закрывает только свою часть — мультяшных и игровых персонажей,
которых в скачанном контенте оказалось неожиданно много.

Считается на широком кропе вокруг лица (с запасом контекста — на тесной
обрезке модель путается и почти всё называет «реальным»), не на всём
снимке целиком: так работает одинаково для фото и видео, без зависимости
от файловой классификации `content_type` — та для видео ненадёжна (даже
настоящие домашние ролики иногда получают «игра»/«скриншот»).

Считаются только неназванные лица — на уже подтверждённых пользователем
именах эта проверка ничего не решает.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sqlite3
import time

from PIL import Image, ImageOps

import catalogdb
import pathkeys
import video as video_media

DEFAULT_MODEL = 'mobilenetv3_v1.4_dist'
# Порог найден по данным этого каталога: явные случаи давали 0.95-0.99 в
# обе стороны, запас в 0.85 оставляет неоднозначные лица непомеченными —
# лучше пропустить мусорную группу, чем спрятать настоящего человека.
ANIME_THRESHOLD = 0.85


def connect(catalog):
    db = catalogdb.connect(catalog, timeout=30)
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS face_authenticity (
          face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
          real_score REAL NOT NULL, anime_score REAL NOT NULL,
          model TEXT NOT NULL, analyzed_at TEXT NOT NULL
        );
    ''')
    db.commit()
    return db


def progress(path, **state):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state['pid'] = os.getpid()
    state['updated_at'] = time.time()
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')  # свой у каждого процесса
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
    # Файл прогресса читает device_job.py каждые полсекунды — os.replace изредка
    # натыкается на этот момент чтения (WinError 5), несколько попыток решают дело.
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def scope_sql(roots, paths, column='path'):
    """Условие «только выбранные папки и файлы» — как в остальных этапах."""
    return pathkeys.scope_sql(roots, paths, column)


def pending(db, args):
    """Неназванные лица без уже посчитанной подлинности."""
    where, values = scope_sql(args.root, args.path, 'faces.path')
    fresh = '' if args.force else '''
        AND NOT EXISTS (SELECT 1 FROM face_authenticity fa WHERE fa.face_id=faces.id)'''
    rows = db.execute('''
        SELECT faces.id, faces.path, faces.box, faces.frame_time FROM faces
        LEFT JOIN face_people ON face_people.face_id=faces.id
        WHERE face_people.person_id IS NULL ''' + fresh + where
        + ' ORDER BY faces.id LIMIT ?', (*values, args.limit)).fetchall()
    return video_media.only(args.kinds, rows, 1)


def wide_crop(path, box_json, frame_time, margin=0.4):
    """Кроп вокруг лица с запасом — на тесной обрезке модель путается."""
    path = Path(video_media.local(path))
    if not path.is_file():
        return None
    if video_media.is_video(path):
        image = video_media.to_image(video_media.poster(path, frame_time or 0))
    else:
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
    left, top, right, bottom = (float(value) for value in json.loads(box_json))
    side = max(right - left, bottom - top) * (1 + margin)
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    return image.crop((
        max(0, int(center_x - side / 2)), max(0, int(center_y - side / 2)),
        min(image.width, int(center_x + side / 2)),
        min(image.height, int(center_y + side / 2)))).convert('RGB')


def save(db, face_id, real_score, anime_score, model):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with db:
        db.execute(
            'INSERT INTO face_authenticity(face_id,real_score,anime_score,model,analyzed_at) '
            'VALUES(?,?,?,?,?) ON CONFLICT(face_id) DO UPDATE SET '
            'real_score=excluded.real_score,anime_score=excluded.anime_score,'
            'model=excluded.model,analyzed_at=excluded.analyzed_at',
            (face_id, real_score, anime_score, model, now))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100000)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--progress-file', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--root', action='append', type=str, default=[])
    parser.add_argument('--path', action='append', type=str, default=[])
    parser.add_argument('--kinds', choices=video_media.KINDS, default='all',
                        help='Считать снимки, ролики или всё сразу')
    parser.add_argument('--force', action='store_true',
                        help='Посчитать заново, даже если уже есть результат')
    args = parser.parse_args()

    db = connect(args.catalog.resolve())
    rows = pending(db, args)
    target = args.progress_file.resolve() if args.progress_file else None
    stop = args.stop_file.resolve() if args.stop_file else None
    state = {'status': 'preparing', 'phase': 'authenticity', 'total': len(rows), 'completed': 0,
             'anime': 0, 'real': 0, 'errors': 0, 'current': '', 'model': args.model}
    progress(target, **state)
    if not rows:
        state['status'] = 'completed'
        progress(target, **state)
        return

    from imgutils.validate import anime_real_score
    state['status'] = 'running'
    progress(target, **state)

    for face_id, path, box_json, frame_time in rows:
        if stop and stop.exists():
            state['status'] = 'stopped'
            progress(target, **state)
            return
        state['current'] = path
        try:
            crop = wide_crop(path, box_json, frame_time)
            if crop is None:
                raise OSError('файл не найден')
            scores = anime_real_score(crop, model_name=args.model)
            save(db, face_id, float(scores['real']), float(scores['anime']), args.model)
            state['anime' if scores['anime'] > scores['real'] else 'real'] += 1
        except Exception as exc:
            print(f'Подлинность: {path}: {exc}', flush=True)
            state['errors'] += 1
        state['completed'] += 1
        progress(target, **state)
        print(f"Подлинность {state['completed']}/{state['total']}", flush=True)

    state['current'] = ''
    state['status'] = 'completed'
    progress(target, **state)


if __name__ == '__main__':
    main()
