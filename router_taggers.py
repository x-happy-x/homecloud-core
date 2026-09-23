"""Альтернативные разметчики для обучения: RAM++ и Qwen3-VL.

Основной путь не меняется — метки ставит визуальный индекс (zero-shot или
своя обученная версия). Эти двое смотрят на сами снимки и кладут свои оценки
рядом, в router_predictions под своим source. В очереди разметки они видны
как отдельные подсказки, а по желанию их ответ сразу сохраняется разметкой
(source ram_plus / qwen), на которой потом учится своя модель.

RAM++ (Recognize Anything Plus) знает 4585 английских тегов — наши метки
собраны из них таблицей RAM_TAGS. Живёт в work/ram-venv: ему нужны
transformers 4.25 и timm 0.4.12, которые сломали бы vision-venv (см.
setup-ram.ps1). Qwen3-VL отвечает на промпт со списком меток — локально из
кэша или через LM Studio, как описания.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import envs
import router_learning
import settings as catalog_settings


RAM_FOLDER = Path(r'C:\cv-models\ram-plus')
RAM_WEIGHTS = RAM_FOLDER / 'ram_plus_swin_large_14m.pth'
RAM_TOKENIZER = RAM_FOLDER / 'bert-base-uncased'
RAM_VERSION = 'ram_plus_swin_large_14m'

ENGINES = {
    'ram_plus': {'title': 'RAM++', 'venv': 'ram-venv'},
    'qwen': {'title': 'Qwen3-VL-2B', 'venv': 'vision-venv'},
    'lmstudio': {'title': 'LM Studio', 'venv': 'vision-venv'},
}
QWEN_MODEL = 'Qwen/Qwen3-VL-2B-Instruct'

# Метка → теги RAM++, любой из которых её подтверждает. Меток без пары
# (размыто, мем) RAM++ не оценивает вовсе — там нет честного соответствия.
RAM_TAGS = {
    'person': ['person', 'man', 'woman', 'child', 'girl', 'boy', 'business woman', 'crowd'],
    'portrait': ['portrait', 'portrait session', 'face close-up'],
    'selfie': ['selfie'],
    'group_photo': ['group photo', 'crowd'],
    'indoor': ['indoor', 'living room', 'bedroom', 'kitchen', 'home interior', 'dining room',
               'meeting room', 'hotel room'],
    'outdoor': ['outdoor', 'nature', 'city street', 'landscape', 'beach', 'mountain', 'forest'],
    'landscape': ['landscape', 'mountain landscape', 'nature'],
    'food': ['food', 'meal', 'fast food', 'street food', 'side dish', 'fruit dish'],
    'pet': ['pet', 'animal', 'cat', 'dog'],
    'vehicle': ['vehicle', 'motor vehicle', 'land vehicle', 'car', 'bus', 'train', 'motorcycle',
                'boat', 'city bus'],
    'architecture': ['architecture', 'building', 'building facade', 'office building', 'house exterior',
                     'landmark'],
    'home': ['home', 'home interior', 'living room', 'bedroom', 'kitchen', 'apartment', 'home decor'],
    'city': ['city', 'city street', 'urban', 'city skyline', 'city view', 'city square', 'street scene'],
    'beach': ['beach'],
    'mountains': ['mountain', 'mountain range', 'mountain landscape', 'snow mountain'],
    'forest': ['forest', 'pine forest', 'autumn forest', 'forest path'],
    'snow': ['snow', 'winter scene', 'snow mountain'],
    'water': ['water', 'sea', 'lake', 'river', 'sea view', 'mountain lake'],
    'party': ['party', 'birthday party', 'celebration', 'wedding party', 'dinner party'],
    'concert': ['concert', 'rock concert', 'stage'],
    'sports': ['sport', 'stadium', 'football game', 'basketball game', 'sport team'],
    'travel': ['travel', 'tourist', 'tourist attraction', 'landmark'],
    'cat': ['cat', 'persian cat'],
    'dog': ['dog', 'street dog', 'guard dog'],
    'bird': ['bird', 'water bird'],
    'flowers': ['flower', 'flower bed', 'flower field', 'hibiscus flower'],
    'product': ['product'],
    'graphics': ['illustration', 'cartoon', 'anime', 'logo', 'icon', 'app icon', 'vector icon',
                 'clip art', 'line art', 'cartoon character', 'poster', 'drawing'],
    'game': ['video game'],
    'screenshot': ['screenshot', 'website', 'text message'],
    'document': ['document', 'paper', 'letter'],
    'text_heavy': ['text', 'document', 'website', 'menu', 'poster page'],
    'receipt': ['receipt'],
    'handwritten_text': ['handwriting'],
    'qr_code': ['qr code'],
    'close_up': ['close-up', 'face close-up'],
    'mirror': ['mirror', 'bathroom mirror'],
    'night': ['night', 'night sky', 'night view', 'city nightview'],
    'sunset': ['sunset', 'sunrise'],
    'black_and_white': ['monochrome'],
    'dark': ['dark'],
}


def write_progress(path, **value):
    router_learning.write_progress(path, **value)


def ram_available():
    return RAM_WEIGHTS.is_file() and (RAM_TOKENIZER / 'vocab.txt').is_file()


def engine_version(engine, options):
    if engine == 'ram_plus':
        return RAM_VERSION
    if engine == 'qwen':
        return QWEN_MODEL
    return f"{options['caption_model']}@lmstudio"


def ram_tagger(batch_size=8):
    """Список PIL-кадров → оценки наших меток от RAM++.

    Порог у каждого из 4585 тегов свой (ram_tag_list_threshold.txt), поэтому
    вероятность переводим так, что ровно на пороге получается 0.5: дальше её
    можно сравнивать с остальными источниками по одному правилу «≥ .5».
    """
    import torch
    from ram import get_transform
    from ram.models import ram_plus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ram_plus(pretrained=str(RAM_WEIGHTS), image_size=384, vit='swin_l',
                     text_encoder_type=str(RAM_TOKENIZER)).eval().to(device)
    transform = get_transform(image_size=384)
    tags = {name: index for index, name in enumerate(model.tag_list)}
    missing = sorted({tag for values in RAM_TAGS.values() for tag in values} - set(tags))
    if missing:
        raise RuntimeError(f'В словаре RAM++ нет тегов: {", ".join(missing)}')
    thresholds = model.class_threshold.cpu().numpy()
    import numpy as np

    def logits(images):
        # Тот же путь, что generate_tag, но без отсечки: нужны сами вероятности.
        pixels = torch.stack([transform(image.convert('RGB')) for image in images]).to(device)
        with torch.inference_mode():
            image_embeds = model.image_proj(model.visual_encoder(pixels))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            cls = image_embeds[:, 0, :]
            cls = cls / cls.norm(dim=-1, keepdim=True)
            count = len(images)
            per_class = int(model.label_embed.shape[0] / model.num_class)
            weights = (model.reweight_scale.exp() * cls @ model.label_embed.t()).view(count, -1, per_class)
            weights = torch.nn.functional.softmax(weights, dim=2)
            values = model.label_embed.view(-1, per_class, 512)
            label_embed = (weights.unsqueeze(-1) * values.unsqueeze(0)).sum(dim=2)
            label_embed = torch.nn.functional.relu(model.wordvec_proj(label_embed))
            tagging = model.tagging_head(encoder_embeds=label_embed, encoder_hidden_states=image_embeds,
                                         encoder_attention_mask=image_atts, return_dict=False,
                                         mode='tagging')
            return torch.sigmoid(model.fc(tagging[0]).squeeze(-1)).float().cpu().numpy()

    def calibrated(probability, threshold):
        if probability >= threshold:
            return .5 + .5 * (probability - threshold) / max(1e-6, 1 - threshold)
        return .5 * probability / max(1e-6, threshold)

    def tag(images):
        probabilities = np.concatenate([logits(images[offset:offset + batch_size])
                                        for offset in range(0, len(images), batch_size)])
        result = []
        for row in probabilities:
            scores = {}
            for label, names in RAM_TAGS.items():
                scores[label] = round(max(calibrated(float(row[tags[name]]), float(thresholds[tags[name]]))
                                          for name in names), 4)
            top = sorted(((float(row[i]) - float(thresholds[i]), name) for name, i in tags.items()),
                         reverse=True)
            result.append((scores, [name for margin, name in top[:12] if margin > 0]))
        return result

    return tag


def qwen_prompt():
    lines = '\n'.join(f"- {key}: {value[0]} — {value[1]}"
                      for key, value in router_learning.LABELS.items())
    return f'''Посмотри на изображение и выбери все категории, которые на нём действительно видны.

Категории (id: название — смысл):
{lines}

Правила:
1. Только id из списка. Категорий может быть несколько или ни одной.
2. Отмечай только то, что видно. Не угадывай.
3. Верни ТОЛЬКО JSON без Markdown и пояснений: {{"labels": ["id1", "id2"]}}'''


def qwen_tagger(engine, options):
    """PIL-кадр → оценки меток от Qwen3-VL: 1 — назвал, 0 — не назвал.

    qwen — маленькая модель из локального кэша, lmstudio — та, что загружена в
    LM Studio по адресу из настроек описаний (обычно крупнее и точнее).
    """
    import caption_photos
    if engine == 'lmstudio':
        ask = caption_photos.lmstudio_captioner(options['caption_lmstudio_url'], options['caption_model'])
    else:
        ask = caption_photos.local_captioner(QWEN_MODEL)
    prompt = qwen_prompt()
    titles = {value[0].casefold(): key for key, value in router_learning.LABELS.items()}

    def tag(images):
        result = []
        for image in images:
            # Кадр в полном размере — тысячи визуальных токенов и полминуты на снимок;
            # для выбора меток хватает стороны в 896 точек.
            image = image.convert('RGB')
            image.thumbnail((896, 896))
            answer = router_learning.parse_answer(ask(image, prompt))
            raw = answer.get('labels') if isinstance(answer, dict) else None
            if not isinstance(raw, list):
                raise ValueError('модель не вернула labels')
            named = {titles.get(str(item).strip().casefold(), str(item).strip()) for item in raw}
            result.append(({key: float(key in named) for key in router_learning.LABELS},
                           sorted(named - set(router_learning.LABELS))))
        return result

    return tag


def candidates(db, engine, version, scope, count, hide_adult=False):
    """Кого размечать: следующие снимки очереди или уже проверенные вручную (для оценки)."""
    adult = (" AND f.path NOT IN (SELECT path FROM photo_adult_analysis "
             "WHERE status='ok' AND rating NOT IN ('safe','unknown','sensitive'))" if hide_adult else '')
    fresh = ' AND f.path NOT IN (SELECT path FROM router_predictions WHERE source=? AND model_version=?)'
    if scope == 'reviewed':
        return [row[0] for row in db.execute(
            "SELECT f.path FROM photos f JOIN router_reviews r ON r.path=f.path "
            "WHERE r.source='human' AND f.status='ok'" + fresh + ' ORDER BY f.path LIMIT ?',
            (engine, version, count))]
    # Тот же порядок, что у очереди разметки, — размеченное окажется первым в ней.
    embedding_model = catalog_settings.read(db)['visual_model']
    source, source_version = router_learning.prediction_source(db, embedding_model)
    return [row[0] for row in db.execute('''
        SELECT p.path FROM router_predictions p JOIN photos f ON f.path=p.path
        LEFT JOIN router_reviews r ON r.path=p.path
        WHERE p.source=? AND p.model_version=? AND r.path IS NULL
          AND f.status='ok' AND COALESCE(f.blocked,0)=0
          AND p.path NOT IN (SELECT path FROM hidden_photos)
          AND p.path NOT IN (SELECT path FROM router_skips)
          AND p.path NOT IN (SELECT i.path FROM router_batch_items i
                             JOIN router_batches b ON b.id=i.batch_id WHERE b.status='exported')'''
        + fresh.replace('f.path', 'p.path') + adult + '''
        GROUP BY p.path ORDER BY AVG(ABS(p.score-.5)) ASC,p.path LIMIT ?''',
        (source, source_version, engine, version, count))]


def run(catalog, engine, scope='queue', count=20, accept=False, hide_adult=False,
        progress=None, stop=None):
    import video as video_media
    if engine not in ENGINES:
        raise ValueError('Неизвестный разметчик')
    action = 'tag'
    db = router_learning.connect(catalog)
    try:
        options = catalog_settings.read(db)
        version = engine_version(engine, options)
        paths = candidates(db, engine, version, scope, max(1, min(int(count), 5000)), hide_adult)
        state = {'status': 'preparing', 'action': action, 'engine': engine, 'scope': scope,
                 'accept': accept, 'total': len(paths), 'completed': 0, 'errors': 0,
                 'accepted': 0}
        write_progress(progress, **state)
        if not paths:
            write_progress(progress, **{**state, 'status': 'completed'})
            return
        if engine == 'ram_plus':
            if not ram_available():
                raise RuntimeError(f'Веса RAM++ не найдены в {RAM_FOLDER} — запустите setup-ram.ps1')
            tag = ram_tagger()
        else:
            tag = qwen_tagger(engine, options)
        state['status'] = 'running'
        write_progress(progress, **state)
        step = 8 if engine == 'ram_plus' else 1
        for offset in range(0, len(paths), step):
            if stop and Path(stop).exists():
                write_progress(progress, **{**state, 'status': 'stopped'})
                return
            part = paths[offset:offset + step]
            images, ready = [], []
            for path in part:
                try:
                    images.append(video_media.open_frame(path))
                    ready.append(path)
                except Exception as exc:
                    state['errors'] += 1
                    print(f'{path}: {exc}', file=sys.stderr, flush=True)
            try:
                results = tag(images) if images else []
            except Exception as exc:
                state['errors'] += len(ready)
                state['error'] = str(exc)[:300]
                print(f'tagger: {exc}', file=sys.stderr, flush=True)
                results = []
                ready = []
            stamp = router_learning.now()
            with db:
                for path, (scores, extra) in zip(ready, results):
                    db.executemany('''INSERT INTO router_predictions
                        (path,label,score,source,model_version,predicted_at) VALUES(?,?,?,?,?,?)
                        ON CONFLICT(path,label,source,model_version) DO UPDATE SET
                        score=excluded.score,predicted_at=excluded.predicted_at''',
                        [(path, label, score, engine, version, stamp) for label, score in scores.items()])
            if accept and scope == 'queue':
                for path, (scores, _) in zip(ready, results):
                    # Кто-то успел разметить вручную, пока шла модель, — его ответ главнее.
                    if db.execute('SELECT 1 FROM router_reviews WHERE path=?', (path,)).fetchone():
                        continue
                    router_learning.save_review(catalog, path, {k: v >= .5 for k, v in scores.items()},
                                                ENGINES[engine]['title'], engine)
                    state['accepted'] += 1
            state['completed'] = min(len(paths), offset + len(part))
            write_progress(progress, **state)
            print(f"{engine} {state['completed']}/{state['total']}", flush=True)
        write_progress(progress, **{**state, 'status': 'completed'})
    finally:
        db.close()


def quality(db, engine, version=None):
    """Сколько из подсказок разметчика совпало с ручной разметкой.

    Считается только по меткам, которые разметчик вообще оценивает, и только по
    снимкам, проверенным человеком: чужие ответы (нейросеть, «похожие») не эталон.
    Идём от проверенных снимков по первичному ключу — у общей модели оценок
    десятки тысяч, и перебор их по источнику занимал секунды.
    """
    rows = db.execute('''SELECT p.path,p.score>=.5,l.value FROM router_reviews r
        JOIN router_predictions p INDEXED BY sqlite_autoindex_router_predictions_1 ON p.path=r.path
        JOIN router_training_labels l ON l.path=p.path AND l.label=p.label
        WHERE r.source='human' AND p.source=? AND (? IS NULL OR p.model_version=?)''',
                      (engine, version, version)).fetchall()
    photos = len({row[0] for row in rows})
    rows = [row[1:] for row in rows]
    tp = sum(1 for said, truth in rows if said and truth)
    fp = sum(1 for said, truth in rows if said and not truth)
    fn = sum(1 for said, truth in rows if not said and truth)
    return {'photos': photos,
            'precision': round(tp / (tp + fp), 4) if tp + fp else None,
            'recall': round(tp / (tp + fn), 4) if tp + fn else None}


_lmstudio_seen = {}


def lmstudio_online(url):
    """LM Studio отвечает? Чтобы не предлагать мёртвую кнопку.

    Отказ в соединении Windows отдаёт не сразу, а после повторов SYN, поэтому
    ответ помним полминуты — вкладка перезапрашивает список после каждого задания.
    """
    from urllib.parse import urlsplit
    from urllib.request import urlopen
    known = _lmstudio_seen.get(url)
    if known and time.time() - known[0] < 30:
        return known[1]
    parts = urlsplit(url)
    try:
        with urlopen(f'{parts.scheme}://{parts.netloc}/v1/models', timeout=.7) as response:
            online = response.status == 200
    except OSError:
        online = False
    _lmstudio_seen[url] = (time.time(), online)
    return online


def overview(catalog, root):
    """Состояние разметчиков для вкладки обучения."""
    db = router_learning.connect(catalog)
    try:
        options = catalog_settings.read(db)
        embedding_model = options['visual_model']
        source, version = router_learning.prediction_source(db, embedding_model)
        result = []
        for engine, meta in ENGINES.items():
            python = envs.python(meta['venv'], root)
            if engine == 'ram_plus':
                ready = python.is_file() and ram_available()
                note = 'Готовая модель распознавания: 4585 тегов, из них собраны наши метки.'
                detail = 'локально, видеокарта'
            elif engine == 'qwen':
                ready = python.is_file()
                note = 'Небольшая модель со зрением: смотрит на снимок и выбирает метки по списку.'
                detail = f'{QWEN_MODEL.split("/")[-1]} · локально, видеокарта'
            else:
                ready = python.is_file() and lmstudio_online(options['caption_lmstudio_url'])
                note = 'Модель, загруженная в LM Studio, по тому же списку меток. Нужен запущенный LM Studio.'
                detail = options['caption_model'] + ('' if ready else ' · LM Studio не отвечает')
            tagged = db.execute(
                'SELECT COUNT(*) FROM (SELECT DISTINCT path FROM router_predictions '
                'INDEXED BY router_predictions_source WHERE source=?)', (engine,)).fetchone()[0]
            accepted = db.execute('SELECT COUNT(*) FROM router_reviews WHERE source=?',
                                  (engine,)).fetchone()[0]
            result.append({'id': engine, 'title': meta['title'], 'ready': ready, 'note': note,
                           'detail': detail, 'tagged': tagged, 'accepted': accepted,
                           'labels': len(RAM_TAGS) if engine == 'ram_plus' else len(router_learning.LABELS),
                           'quality': quality(db, engine)})
        base = quality(db, source, version)
        result.insert(0, {'id': source, 'title': 'Своя модель' if source == 'trained' else 'Общая модель',
                          'base': True, 'ready': True, 'detail': version,
                          'labels': len(router_learning.LABELS), 'quality': base})
        return result
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', choices=tuple(ENGINES))
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--scope', choices=('queue', 'reviewed'), default='queue')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--accept', action='store_true',
                        help='Сразу сохранить ответ разметкой (только для очереди)')
    parser.add_argument('--hide-adult', action='store_true')
    parser.add_argument('--progress', type=Path)
    parser.add_argument('--stop', type=Path)
    args = parser.parse_args()
    started = time.time()
    try:
        run(args.catalog.resolve(), args.engine, args.scope, args.count, args.accept,
            args.hide_adult, args.progress, args.stop)
    except Exception as exc:
        write_progress(args.progress, status='error', action='tag', engine=args.engine,
                       error=str(exc))
        raise
    print(f'done in {time.time() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
