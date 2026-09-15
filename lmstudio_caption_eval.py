"""Evaluate an LM Studio vision model on local safe and NSFW folders."""
import argparse
import base64
from datetime import datetime, timezone
import io
import json
import html
from pathlib import Path
import random
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from PIL import Image, ImageOps


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
EXPECTED = {
    'caption_short_ru', 'caption_short_en', 'description_ru', 'description_en',
    'tags_ru', 'tags_en', 'people', 'scene', 'actions', 'clothing', 'objects',
    'visual_attributes', 'content', 'search_text_ru', 'search_text_en',
}
RATINGS = {'safe', 'suggestive', 'nudity', 'explicit'}


def object_schema(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


TEXT_ARRAY = {'type': 'array', 'items': {'type': 'string'}}
OUTPUT_SCHEMA = object_schema({
    'caption_short_ru': {'type': 'string'},
    'caption_short_en': {'type': 'string'},
    'description_ru': {'type': 'string'},
    'description_en': {'type': 'string'},
    'tags_ru': TEXT_ARRAY,
    'tags_en': TEXT_ARRAY,
    'people': object_schema({'count': {'type': 'integer', 'minimum': 0},
                             'presentation': TEXT_ARRAY}),
    'scene': object_schema({'ru': {'type': ['string', 'null']},
                            'en': {'type': ['string', 'null']},
                            'indoor': {'type': ['boolean', 'null']},
                            'outdoor': {'type': ['boolean', 'null']}}),
    'actions': object_schema({'ru': TEXT_ARRAY, 'en': TEXT_ARRAY}),
    'clothing': object_schema({'ru': TEXT_ARRAY, 'en': TEXT_ARRAY}),
    'objects': object_schema({'ru': TEXT_ARRAY, 'en': TEXT_ARRAY}),
    'visual_attributes': object_schema({'ru': TEXT_ARRAY, 'en': TEXT_ARRAY}),
    'content': object_schema({
        'rating': {'type': 'string', 'enum': sorted(RATINGS)},
        'nsfw': {'type': 'boolean'},
        'categories': {'type': 'array', 'items': {'type': 'string', 'enum': [
            'partial_nudity', 'nudity', 'underwear', 'lingerie', 'swimwear',
            'suggestive_pose', 'intimate_pose', 'sexual_activity']}},
    }),
    'search_text_ru': {'type': 'string'},
    'search_text_en': {'type': 'string'},
})


def choose(folder, count, seed):
    files = [path for path in folder.iterdir()
             if path.is_file() and path.suffix.casefold() in EXTENSIONS]
    files.sort(key=lambda path: path.name.casefold())
    random.Random(seed).shuffle(files)
    return files[:count]


def image_url(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    image.thumbnail((1280, 1280))
    stream = io.BytesIO()
    image.save(stream, 'JPEG', quality=86, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def request_json(url, payload, timeout=600):
    request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'LM Studio HTTP {exc.code}: {detail}') from exc


def validate(value, expected_rating):
    issues = []
    missing = EXPECTED - set(value)
    extra = set(value) - EXPECTED
    if missing:
        issues.append('missing: ' + ', '.join(sorted(missing)))
    if extra:
        issues.append('extra: ' + ', '.join(sorted(extra)))
    content = value.get('content') if isinstance(value.get('content'), dict) else {}
    rating = content.get('rating')
    nsfw = content.get('nsfw')
    if rating not in RATINGS:
        issues.append('invalid rating')
    if nsfw is not (rating != 'safe'):
        issues.append('rating/nsfw mismatch')
    if expected_rating == 'safe' and rating != 'safe':
        issues.append('possible false positive')
    if expected_rating == 'nsfw' and rating == 'safe':
        issues.append('possible false negative')
    for key in ('caption_short_ru', 'caption_short_en'):
        length = len(str(value.get(key, '')).split())
        if not 4 <= length <= 12:
            issues.append(f'{key} length={length}')
    for key in ('tags_ru', 'tags_en'):
        tags = value.get(key)
        if not isinstance(tags, list) or not 8 <= len(tags) <= 32:
            issues.append(f'{key} count={len(tags) if isinstance(tags, list) else "invalid"}')
    return issues


def samples_for(args):
    if args.files:
        if args.only == 'both':
            raise ValueError('--files requires --only safe or --only nsfw')
        return [(args.only, path) for path in args.files]
    safe = [('safe', path) for path in choose(args.safe, args.count, args.seed)]
    nsfw = [('nsfw', path) for path in choose(args.nsfw, args.count, args.seed + 1)]
    return {'safe': safe, 'nsfw': nsfw, 'both': safe + nsfw}[args.only]


def prepare_wd(samples, output):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import adult_photos
    torch, model, transform, labels = adult_photos.load_wd(adult_photos.WD_MODEL)
    result = {}
    for index, (_, path) in enumerate(samples, 1):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        ratings, tags = adult_photos.wd_tags(
            torch, model, transform, labels, image, threshold=.35)
        result[str(path.resolve())] = {'ratings': ratings, 'tags': tags[:30]}
        print(json.dumps({'wd': index, 'file': path.name, 'ratings': ratings,
                          'top_tags': [tag['name'] for tag in tags[:12]]},
                         ensure_ascii=False), flush=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def wd_prompt(context):
    if not context:
        return ''
    ratings = ', '.join(f'{name}={score:.3f}'
                        for name, score in context.get('ratings', {}).items())
    tags = ', '.join(f"{item['name']}={item['score']:.3f}"
                     for item in context.get('tags', [])[:12])
    return f'''\n\nДополнительный машинный сигнал от специализированного WD-tagger:
ratings: {ratings}
tags: {tags}

Используй эти вероятности и теги только как вспомогательные подсказки. Перепроверь
каждый признак по самому изображению: не добавляй невидимое только из-за тега и не
игнорируй визуально подтверждённую наготу, анатомию или сексуальное действие, если
соответствующий тег помогает их заметить. В итоговом JSON не упоминай WD-tagger.''' 


def write_html(report, output):
    cards = []
    for sample in report['samples']:
        result = sample.get('result') or {}
        content = result.get('content') or {}
        source_uri = Path(sample['path']).resolve().as_uri()
        raw = json.dumps(result, ensure_ascii=False, indent=2)
        tags = ', '.join(result.get('tags_ru') or [])
        wd = sample.get('wd_context') or {}
        wd_ratings = ', '.join(f'{name}: {score:.3f}'
                               for name, score in wd.get('ratings', {}).items())
        wd_tags = ', '.join(f"{item['name']} ({item['score']:.2f})"
                            for item in wd.get('tags', [])[:12])
        cards.append(f'''<article>
          <header><div><h2>{html.escape(sample['filename'])}</h2>
          <small>{sample['seconds']} с · ожидаемый набор: {html.escape(sample['set'])}</small></div>
          <b class="rating {html.escape(str(content.get('rating', 'error')))}">{html.escape(str(content.get('rating', 'error')))}</b></header>
          <h3>{html.escape(str(result.get('caption_short_ru', 'Ошибка')))}</h3>
          <p>{html.escape(str(result.get('description_ru', sample.get('error', ''))))}</p>
          <p class="tags">{html.escape(tags)}</p>
          <div class="wd"><b>WD ratings:</b> {html.escape(wd_ratings)}<br>
          <b>WD top tags:</b> {html.escape(wd_tags)}</div>
          <details><summary>Показать исходное изображение</summary><img src="{html.escape(source_uri)}" alt=""></details>
          <details><summary>Полный JSON</summary><pre>{html.escape(raw)}</pre></details>
        </article>''')
    document = f'''<!doctype html><html lang="ru"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>LM Studio + WD — тест описаний</title><style>
    :root{{color-scheme:dark}}body{{margin:0;background:#111715;color:#edf3f0;font:15px system-ui}}
    main{{max-width:1050px;margin:auto;padding:24px}}article{{background:#1a2320;border:1px solid #34433e;border-radius:18px;padding:18px;margin:16px 0}}
    header{{display:flex;justify-content:space-between;gap:15px}}h1,h2,h3{{margin:.25em 0}}small,.tags{{color:#aebdb7}}.rating{{padding:6px 10px;border-radius:999px;height:max-content;background:#77402e}}.safe{{background:#286344}}
    .wd{{background:#121a17;border-radius:10px;padding:11px;line-height:1.65}}img{{display:block;max-width:100%;max-height:75vh;margin:12px auto;border-radius:12px}}details{{margin-top:13px}}summary{{cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0d1210;padding:12px;border-radius:10px}}
    </style><main><h1>Huihui Qwen3.8 27B + WD-tagger</h1>
    <p>WD-теги добавлены в промпт как вероятностные подсказки. Изображения раскрываются только вручную.</p>{''.join(cards)}</main></html>'''
    output.write_text(document, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--safe', type=Path, required=True)
    parser.add_argument('--nsfw', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--only', choices=('both', 'safe', 'nsfw'), default='both')
    parser.add_argument('--files', type=Path, nargs='+',
                        help='Fixed files for a reproducible safe or NSFW run')
    parser.add_argument('--seed', type=int, default=3817)
    parser.add_argument('--url', default='http://127.0.0.1:1234/v1/chat/completions')
    parser.add_argument('--model', default='huihui-qwen3.8-27b-vision')
    parser.add_argument('--prompt', type=Path,
                        default=Path(__file__).with_name('lmstudio_caption_prompt.txt'))
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('lmstudio-caption-eval-results.json'))
    parser.add_argument('--html-output', type=Path)
    parser.add_argument('--wd-context', type=Path)
    parser.add_argument('--prepare-wd-context', action='store_true')
    args = parser.parse_args()
    base_prompt = args.prompt.read_text(encoding='utf-8').rstrip()
    samples = samples_for(args)
    if args.prepare_wd_context:
        if not args.wd_context:
            parser.error('--prepare-wd-context requires --wd-context')
        prepare_wd(samples, args.wd_context)
        return
    contexts = (json.loads(args.wd_context.read_text(encoding='utf-8'))
                if args.wd_context else {})
    report = {'model': args.model, 'created_at': datetime.now(timezone.utc).isoformat(),
              'prompt': str(args.prompt), 'samples': []}
    for index, (expected, path) in enumerate(samples, 1):
        started = time.perf_counter()
        entry = {'set': expected, 'path': str(path), 'filename': path.name}
        try:
            context = contexts.get(str(path.resolve()), {})
            prompt = base_prompt + wd_prompt(context) + '\n\n/no_think'
            payload = {
                'model': args.model,
                'temperature': 0.1,
                'max_tokens': 1000,
                'reasoning_effort': 'none',
                'response_format': {'type': 'text'},
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': image_url(path)}},
                ]}],
            }
            response = request_json(args.url, payload)
            raw = response['choices'][0]['message']['content'].strip()
            value = json.loads(raw)
            entry.update(ok=True, result=value, issues=validate(value, expected),
                         usage=response.get('usage', {}), wd_context=context)
        except Exception as exc:
            entry.update(ok=False, error=f'{type(exc).__name__}: {exc}', issues=['request failed'])
        entry['seconds'] = round(time.perf_counter() - started, 2)
        report['samples'].append(entry)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if args.html_output:
            write_html(report, args.html_output)
        print(json.dumps({'sample': index, 'set': expected, 'file': path.name,
                          'ok': entry['ok'], 'seconds': entry['seconds'],
                          'rating': entry.get('result', {}).get('content', {}).get('rating'),
                          'issues': entry['issues']}, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.html_output:
        write_html(report, args.html_output)
    print(json.dumps({'output': str(args.output), 'completed': len(samples)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
