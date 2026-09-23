"""Local detailed adult-content analysis with NudeNet regions and WD tags."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from analyze_photos import connect
import pathkeys
import settings as catalog_settings
import video as video_media


WD_MODEL = 'SmilingWolf/wd-eva02-large-tagger-v3'
# Версия в имени: после правки с EXIF детектор смотрит на повёрнутый кадр,
# поэтому старые разметки считаются устаревшими и пересчитываются.
DETECTOR_MODEL = 'NudeNet-320n-exif'
RATING_NAMES = {'general': 'safe', 'sensitive': 'sensitive',
                'questionable': 'questionable', 'explicit': 'explicit'}
REGION_LABELS = {
    'FEMALE_GENITALIA_EXPOSED': 'обнажённые гениталии',
    'MALE_GENITALIA_EXPOSED': 'обнажённые гениталии',
    'ANUS_EXPOSED': 'обнажённая интимная область',
    'FEMALE_BREAST_EXPOSED': 'обнажённая грудь',
    'BUTTOCKS_EXPOSED': 'обнажённые ягодицы',
    'FEMALE_GENITALIA_COVERED': 'прикрытая интимная область',
    'FEMALE_BREAST_COVERED': 'прикрытая грудь',
    'BUTTOCKS_COVERED': 'прикрытые ягодицы',
    'ANUS_COVERED': 'прикрытая интимная область',
}
# Теги, по которым видно текст и интерфейс: на них теггер регулярно выдаёт
# высокий explicit, хотя тела на картинке нет вовсе.
TEXTUAL = {
    'english text', 'text focus', 'wall of text', 'user interface', 'fake screenshot',
    'fake phone screenshot', 'gameplay mechanics', 'chat log', 'monitor', 'screenshot',
    'no humans', 'heads-up display', 'timestamp', 'video game', 'artist name',
    'character name', 'dated', 'watermark', 'web address', 'phone screen', 'spanish text',
}
# Теги, прямо говорящие о теле: они отменяют «текстовую» поблажку.
BODY = {
    'nude', 'completely nude', 'topless', 'bottomless', 'nipples', 'nipple slip', 'penis',
    'pussy', 'vaginal', 'anal', 'anus', 'testicles', 'erection', 'fellatio', 'oral',
    'sex', 'implied sex', 'cum', 'breasts', 'ass', 'ass focus', 'cleavage', 'lingerie',
}
EXPLICIT = {'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED', 'ANUS_EXPOSED'}
NUDITY = EXPLICIT | {'FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED'}
SUGGESTIVE = {'FEMALE_GENITALIA_COVERED', 'FEMALE_BREAST_COVERED',
              'BUTTOCKS_COVERED', 'ANUS_COVERED'}
# По этой шкале сравниваются кадры видео — берётся самый небезопасный из них.
RANK = {'safe': 0, 'sensitive': 1, 'suggestive': 2, 'nudity': 3, 'explicit': 4}


def save_progress(path, **state):
    if not path:
        return
    state['pid'] = os.getpid()
    state['updated_at'] = time.time()
    temporary = path.with_suffix(path.suffix + '.tmp')
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


def load_wd(model_name):
    import torch
    import timm
    from timm.data import create_transform, resolve_data_config
    from huggingface_hub import hf_hub_download
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна для WD-tagger')
    model = timm.create_model('hf_hub:' + model_name, pretrained=True).eval().to('cuda')
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    labels_path = hf_hub_download(model_name, 'selected_tags.csv')
    with open(labels_path, encoding='utf-8') as source:
        labels = list(csv.DictReader(source))
    return torch, model, transform, labels


def wd_tags(torch, model, transform, labels, image, threshold):
    tensor = transform(image).unsqueeze(0).to('cuda')
    with torch.inference_mode():
        scores = torch.sigmoid(model(tensor)).float().cpu()[0].tolist()
    ratings, tags = {}, []
    for row, score in zip(labels, scores):
        name = row['name']
        category = int(row.get('category', 0))
        if category == 9:
            ratings[RATING_NAMES.get(name, name)] = round(score, 4)
        elif category == 0 and score >= threshold:
            tags.append({'name': name.replace('_', ' '), 'score': round(score, 4)})
    tags.sort(key=lambda item: item['score'], reverse=True)
    return ratings, tags[:40]


def summarize(ratings, tags, regions):
    significant = [item for item in regions if item['class'] in REGION_LABELS]
    parts = []
    if significant:
        grouped = {}
        for item in significant:
            label = REGION_LABELS[item['class']]
            grouped[label] = max(grouped.get(label, 0), item['score'])
        parts.append('Обнаружено: ' + ', '.join(
            f'{label} ({score:.0%})' for label, score in sorted(
                grouped.items(), key=lambda item: item[1], reverse=True)))
    useful = [item['name'] for item in tags
              if item['name'] not in {'1girl', '1boy', 'solo', 'looking at viewer'}][:16]
    if useful:
        parts.append('Теги: ' + ', '.join(useful))
    if ratings:
        best = max(ratings, key=ratings.get)
        parts.append(f'Рейтинг WD: {best} ({ratings[best]:.0%})')
    return '. '.join(parts) + ('.' if parts else 'Признаков 18+ не обнаружено.')


def large_enough(regions, width=0, height=0, share=.006):
    """Отсекает крошечные рамки: на шуме детектор любит находить мелочь."""
    if not width or not height:
        return regions
    limit = width * height * share
    kept = []
    for item in regions:
        box = item.get('box') or []
        if len(box) != 4:
            kept.append(item)
            continue
        area = abs(box[2]) * abs(box[3])   # NudeNet отдаёт [x, y, ширина, высота]
        if area >= limit:
            kept.append(item)
    return kept


def classify(ratings, regions, threshold=.55, tags=(), width=0, height=0):
    """Рейтинг по двум независимым мнениям.

    NudeNet смотрит на области тела, WD-tagger — на картинку целиком. Поодиночке
    оба ошибаются: детектор находит «грудь» на тексте и мебели, теггер выдаёт
    высокий explicit на сканах. Поэтому одиночному сигналу верим только когда он
    уверенный, а второй ему не противоречит.
    """
    names = {str(tag).casefold() for tag in tags}
    by_class = {}
    for item in large_enough(regions, width, height):
        by_class[item['class']] = max(by_class.get(item['class'], 0), item['score'])
    explicit = max((by_class.get(name, 0) for name in EXPLICIT), default=0)
    nudity = max((by_class.get(name, 0) for name in NUDITY), default=0)
    suggestive = max((by_class.get(name, 0) for name in SUGGESTIVE), default=0)

    safe_score = ratings.get('safe', 0)
    wd_explicit = ratings.get('explicit', 0)
    wd_questionable = ratings.get('questionable', 0)
    leader = max(ratings, key=ratings.get) if ratings else 'safe'
    # Теггер спокоен: высокий «safe», низкие остальные — или людей на картинке нет.
    calm = (safe_score >= .9 and max(wd_explicit, wd_questionable) < .25) or (
        'no humans' in names and safe_score >= .6)
    # Теггер поддерживает: он сам склоняется к небезопасному.
    support = (wd_explicit >= .35 or wd_questionable >= .35
               or leader in {'questionable', 'explicit'})

    strong = max(threshold, .62)
    weak = min(threshold, .45)
    score = max(explicit, nudity * .92, suggestive * .55, wd_explicit * .85,
                wd_questionable * .6)
    if calm:
        score *= .5

    # Ни одной области тела, а теги — про текст и интерфейс: теггеру тут верить не в чем.
    textual = (len(names & TEXTUAL) >= 2 or 'no humans' in names) and not (names & BODY)
    if not max(explicit, nudity, suggestive) and textual:
        return 'safe', round(score * .3, 4)

    if (explicit >= strong and not calm) or (explicit >= weak and support) or (
            wd_explicit >= .75 and leader == 'explicit'):
        rating = 'explicit'
    elif (nudity >= strong and not calm) or (nudity >= weak and support) or (
            wd_explicit >= .6 and leader in {'explicit', 'questionable'}):
        rating = 'nudity'
    elif (suggestive >= .6 and support) or (
            wd_questionable >= .65 and leader == 'questionable'):
        rating = 'suggestive'
    elif ratings.get('sensitive', 0) >= .85 and leader == 'sensitive':
        # Пометка «на грани»: галерея её не блюрит, она только для сведений.
        rating = 'sensitive'
    else:
        rating = 'safe'
    return rating, round(score, 4)


def evaluate_image(detector, torch, model, transform, labels, image, threshold, tag_threshold):
    """Полный разбор одной картинки: рамки NudeNet, теги WD, итоговый рейтинг."""
    # Внутри NudeNet обычный cv2.imread, а он на Windows возвращает None
    # для путей с кириллицей — отдаём уже открытый кадр в порядке BGR.
    detections = detector.detect(np.asarray(image)[:, :, ::-1].copy())
    regions = [{'class': item['class'], 'score': round(float(item['score']), 4),
                'box': [int(value) for value in item['box']]}
               for item in detections if float(item['score']) >= .25]
    ratings, tags = wd_tags(torch, model, transform, labels, image, tag_threshold)
    rating, score = classify(ratings, regions, threshold,
                             [item['name'] for item in tags], *image.size)
    return {'rating': rating, 'score': score, 'ratings': ratings, 'tags': tags,
            'regions': regions, 'description': summarize(ratings, tags, regions)}


def evaluate_path(detector, torch, model, transform, labels, path, threshold, tag_threshold,
                  video_frames=3):
    """Оценка файла: у видео берётся несколько кадров и худший результат —
    один кадр из начала мог случайно оказаться безобидным."""
    images = (video_media.sample_images(path, video_frames) if video_media.is_video(path)
              else [video_media.open_frame(path)])
    best = None
    for image in images:
        result = evaluate_image(detector, torch, model, transform, labels, image,
                                threshold, tag_threshold)
        if best is None or RANK[result['rating']] > RANK[best['rating']]:
            best = result
    return best


def scoped_rows(db, args):
    scope, values = pathkeys.scope_sql(args.root, args.path, 'photos.path')
    freshness = '' if args.force else ''' AND (
      photo_adult_analysis.path IS NULL OR photo_adult_analysis.size != photos.size OR
      photo_adult_analysis.modified != photos.modified OR photo_adult_analysis.status != 'ok'
      OR photo_adult_analysis.detector_model IS NOT ?)'''
    extra = () if args.force else (DETECTOR_MODEL,)
    rows = db.execute('''SELECT photos.path,photos.size,photos.modified FROM photos
      LEFT JOIN photo_adult_analysis USING(path)
      WHERE photos.status='ok'
        AND COALESCE(photos.blocked,0)=0 ''' + scope +
      freshness + ' ORDER BY photos.path LIMIT ?',
      (*values, *extra, args.limit)).fetchall()
    return video_media.only(args.kinds, rows)


def recheck(args):
    """Пересчитывает рейтинги по уже сохранённым тегам и областям, без моделей."""
    db = connect(args.catalog.resolve())
    rows = db.execute("""SELECT photo_adult_analysis.path,rating,tags_json,regions_json,
        photo_analysis.width,photo_analysis.height FROM photo_adult_analysis
        LEFT JOIN photo_analysis USING(path) WHERE photo_adult_analysis.status='ok'""").fetchall()
    changes = []
    for path, rating, tags_json, regions_json, width, height in rows:
        payload = json.loads(tags_json or '{}')
        tags = payload.get('tags', [])
        regions = json.loads(regions_json or '[]')
        fresh, score = classify(payload.get('ratings', {}), regions, args.threshold,
                                [item['name'] for item in tags], width or 0, height or 0)
        if fresh != rating:
            changes.append((fresh, score, summarize(payload.get('ratings', {}), tags, regions),
                            path, rating))
    print(f'Пересчитано {len(rows)}, поменялся рейтинг у {len(changes)}.')
    moved = {}
    for fresh, _, _, _, was in changes:
        moved[f'{was} → {fresh}'] = moved.get(f'{was} → {fresh}', 0) + 1
    for move, count in sorted(moved.items(), key=lambda item: -item[1]):
        print(f'   {move}: {count}')
    if not args.apply:
        print('Это только показ. Добавьте --apply, чтобы записать.')
        return
    with db:
        db.executemany('UPDATE photo_adult_analysis SET rating=?,adult_score=?,description=? '
                       'WHERE path=?', [item[:4] for item in changes])
    print('Записано.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--root', action='append', type=str, default=[])
    parser.add_argument('--path', action='append', type=str, default=[])
    parser.add_argument('--kinds', choices=video_media.KINDS, default='all',
                        help='Считать снимки, ролики или всё сразу')
    parser.add_argument('--progress-file', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--threshold', type=float, default=.55)
    parser.add_argument('--tag-threshold', type=float, default=.45)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--recheck', action='store_true',
                        help='Пересчитать рейтинги по сохранённым тегам, не запуская модели')
    parser.add_argument('--apply', action='store_true',
                        help='Записать результат пересчёта в каталог')
    args = parser.parse_args()
    if args.recheck:
        return recheck(args)
    db = connect(args.catalog.resolve())
    rows = scoped_rows(db, args)
    target = args.progress_file.resolve() if args.progress_file else None
    stop = args.stop_file.resolve() if args.stop_file else None
    options = catalog_settings.load(args.catalog.resolve())
    state = {'status': 'preparing', 'phase': 'adult', 'total': len(rows),
             'completed': 0, 'flagged': 0, 'errors': 0, 'current': '',
             'videos_total': sum(video_media.is_video(row[0]) for row in rows),
             'videos_done': 0, 'video_frames': options['adult_video_frames']}
    save_progress(target, **state)
    if not rows:
        state['status'] = 'completed'
        save_progress(target, **state)
        return
    from nudenet import NudeDetector
    detector = NudeDetector()
    torch, model, transform, labels = load_wd(WD_MODEL)
    state['status'] = 'running'
    save_progress(target, **state)
    for path, size, modified in rows:
        if stop and stop.exists():
            state['status'] = 'stopped'
            save_progress(target, **state)
            return
        state['current'] = path
        try:
            result = evaluate_path(detector, torch, model, transform, labels, path,
                                   args.threshold, args.tag_threshold,
                                   options['adult_video_frames'])
            now = datetime.now(timezone.utc).isoformat()
            with db:
                db.execute('''INSERT INTO photo_adult_analysis
                  (path,size,modified,rating,adult_score,tags_json,regions_json,description,
                   detector_model,tagger_model,status,error,analyzed_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?, 'ok',NULL,?) ON CONFLICT(path) DO UPDATE SET
                   size=excluded.size,modified=excluded.modified,rating=excluded.rating,
                   adult_score=excluded.adult_score,tags_json=excluded.tags_json,
                   regions_json=excluded.regions_json,description=excluded.description,
                   detector_model=excluded.detector_model,tagger_model=excluded.tagger_model,
                   status='ok',error=NULL,analyzed_at=excluded.analyzed_at''',
                  (path, size, modified, result['rating'], result['score'],
                   json.dumps({'ratings': result['ratings'], 'tags': result['tags']},
                             ensure_ascii=False),
                   json.dumps(result['regions'], ensure_ascii=False), result['description'],
                   DETECTOR_MODEL, WD_MODEL, now))
            state['flagged'] += result['rating'] != 'safe'
        except Exception as exc:
            now = datetime.now(timezone.utc).isoformat()
            with db:
                db.execute('''INSERT INTO photo_adult_analysis
                  (path,size,modified,status,error,analyzed_at) VALUES(?,?,?,'error',?,?)
                  ON CONFLICT(path) DO UPDATE SET status='error',error=excluded.error,
                  analyzed_at=excluded.analyzed_at''', (path, size, modified, str(exc), now))
            state['errors'] += 1
        state['completed'] += 1
        state['videos_done'] += int(video_media.is_video(path))
        save_progress(target, **state)
        print(f"18+ {state['completed']}/{state['total']}", flush=True)
    state['status'] = 'completed'
    save_progress(target, **state)


if __name__ == '__main__':
    main()
