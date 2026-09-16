"""Generate short searchable Russian captions — a local Qwen3-VL model or LM Studio."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from analyze_photos import connect
import video as video_media


MODEL = 'Qwen/Qwen3-VL-2B-Instruct'
LMSTUDIO_URL = 'http://127.0.0.1:1234/v1/chat/completions'
PROMPT = Path(__file__).with_name('lmstudio_caption_prompt.txt').read_text(encoding='utf-8').strip()


def prompt_with_wd(adult_rating, raw_context):
    """Добавить только уверенные WD-сигналы — подсказки не заменяют изображение."""
    try:
        context = json.loads(raw_context or '{}')
    except json.JSONDecodeError:
        context = {}
    ratings = context.get('ratings') if isinstance(context.get('ratings'), dict) else {}
    tags = context.get('tags') if isinstance(context.get('tags'), list) else []
    trusted = [item for item in tags if isinstance(item, dict)
               and float(item.get('score') or 0) >= .75][:12]
    if not ratings and not trusted and not adult_rating:
        return PROMPT
    rating_text = ', '.join(f'{name}={float(score):.3f}' for name, score in ratings.items())
    tag_text = ', '.join(f"{item.get('name')}={float(item.get('score') or 0):.3f}"
                         for item in trusted)
    return PROMPT + f'''\n\nДополнительный машинный сигнал от локального WD-tagger:
итоговый рейтинг анализатора: {adult_rating or 'unknown'}
ratings: {rating_text or 'нет'}
уверенные теги: {tag_text or 'нет'}

Используй это только как подсказку и перепроверь каждый признак по изображению.
Не добавляй невидимое из-за тега. При этом не скрывай визуально подтверждённую
наготу, анатомию или сексуальное действие. WD-tagger в ответе не упоминай.'''


def parse_result(text):
    """LM Studio иногда оборачивает JSON в markdown или служебный think-блок."""
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('модель не вернула JSON')
    value = json.loads(text[start:end + 1])
    required = {'caption_short_ru', 'description_ru', 'tags_ru', 'tags_en',
                'content', 'search_text_ru', 'search_text_en'}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError('в JSON модели не хватает обязательных полей')
    return value


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


def image_data_url(image):
    """Кадр PIL → data-URL: так его принимает /v1/chat/completions LM Studio."""
    thumbnail = image.copy()
    thumbnail.thumbnail((1280, 1280))
    stream = io.BytesIO()
    thumbnail.save(stream, 'JPEG', quality=86, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def local_captioner(model_name):
    """Готовит Qwen3-VL из локального кэша и возвращает функцию image → подпись."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.float16, low_cpu_mem_usage=True, device_map='cuda').eval()

    def caption(image, prompt):
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors='pt').to('cuda')
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        generated = generated[:, inputs.input_ids.shape[1]:]
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    return caption


def lmstudio_captioner(url, model_name, timeout=600):
    """Тот же результат через уже запущенный LM Studio — без своей видеокарты."""
    def caption(image, prompt):
        payload = {
            'model': model_name, 'temperature': 0.1, 'max_tokens': 1000,
            'reasoning_effort': 'none',
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt + '\n\n/no_think'},
                {'type': 'image_url', 'image_url': {'url': image_data_url(image)}},
            ]}],
        }
        request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=timeout) as response:
                reply = json.loads(response.read().decode('utf-8'))
        except HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'LM Studio HTTP {exc.code}: {detail}') from exc
        except OSError as exc:
            raise RuntimeError(f'LM Studio недоступен по {url}: {exc}') from exc
        return reply['choices'][0]['message']['content'].strip()

    return caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--backend', choices=('local', 'lmstudio'), default='local',
                        help='local — своя видеокарта и модель из кэша; '
                             'lmstudio — уже запущенный сервер LM Studio')
    parser.add_argument('--model', default=MODEL,
                        help='Модель Hugging Face (local) или имя модели, '
                             'загруженной в LM Studio (lmstudio)')
    parser.add_argument('--lmstudio-url', default=LMSTUDIO_URL)
    parser.add_argument('--progress-file', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--root', action='append', type=Path, default=[])
    parser.add_argument('--path', action='append', type=Path, default=[])
    parser.add_argument('--kinds', choices=video_media.KINDS, default='all',
                        help='Считать снимки, ролики или всё сразу')
    parser.add_argument('--force', action='store_true',
                        help='Описать заново, даже если описание уже есть')
    args = parser.parse_args()
    db = connect(args.catalog.resolve())
    root_sql = ''
    root_values = []
    scope = []
    if args.root:
        for root in args.root:
            value = str(root.resolve()).rstrip('\\/')
            scope.append('(photo_analysis.path=? OR photo_analysis.path LIKE ?)')
            root_values.extend((value, value + os.sep + '%'))
    for path in args.path:
        scope.append('photo_analysis.path=?')
        root_values.append(str(path.resolve()))
    if scope:
        root_sql = ' AND (' + ' OR '.join(scope) + ')'
    candidate_sql = '' if args.path else " AND photo_analysis.content_type IN ('photo','document','screenshot')"
    fresh_sql = '' if args.force else " AND (photo_analysis.caption_status IS NULL OR photo_analysis.caption_status='error')"
    rows = db.execute('''
        SELECT photo_analysis.path,photo_adult_analysis.rating,photo_adult_analysis.tags_json
        FROM photo_analysis LEFT JOIN photo_adult_analysis USING(path)
        WHERE photo_analysis.status='ok' ''' + fresh_sql
        + candidate_sql + root_sql +
        " ORDER BY CASE WHEN photo_analysis.content_type='photo' THEN 0 ELSE 1 END,photo_analysis.path LIMIT ?",
        (*root_values, args.limit)).fetchall()
    rows = video_media.only(args.kinds, rows)
    target = args.progress_file.resolve() if args.progress_file else None
    stop = args.stop_file.resolve() if args.stop_file else None
    state = {'status': 'preparing', 'phase': 'caption', 'total': len(rows),
             'completed': 0, 'captioned': 0, 'errors': 0, 'current': '',
             'videos_total': sum(video_media.is_video(row[0]) for row in rows),
             'videos_done': 0, 'video_frames': 2}
    save_progress(target, **state)
    if not rows:
        state['status'] = 'completed'
        save_progress(target, **state)
        return
    caption = (lmstudio_captioner(args.lmstudio_url, args.model) if args.backend == 'lmstudio'
              else local_captioner(args.model))
    state['status'] = 'running'
    save_progress(target, **state)
    for path, adult_rating, wd_json in rows:
        if stop and stop.exists():
            state['status'] = 'stopped'
            save_progress(target, **state)
            return
        state['current'] = path
        try:
            prompt = prompt_with_wd(adult_rating, wd_json)
            raw = caption(video_media.open_frame(path), prompt).strip()
            value = parse_result(raw)
            description = str(value.get('description_ru') or '').strip()
            short = str(value.get('caption_short_ru') or '').strip()
            search = '\n'.join(filter(None, (value.get('search_text_ru'),
                                               value.get('search_text_en'))))
            tags = {'ru': value.get('tags_ru') or [], 'en': value.get('tags_en') or []}
            with db:
                db.execute('''UPDATE photo_analysis SET caption=?,caption_short=?,
                    caption_search=?,caption_tags_json=?,caption_json=?,caption_status='ok'
                    WHERE path=?''',
                           (description, short, search,
                            json.dumps(tags, ensure_ascii=False),
                            json.dumps(value, ensure_ascii=False), path))
            state['captioned'] += bool(description or short)
        except Exception as exc:
            with db:
                db.execute("UPDATE photo_analysis SET caption_status='error',error=? WHERE path=?",
                           (f'Caption: {exc}', path))
            state['errors'] += 1
        state['completed'] += 1
        state['videos_done'] += int(video_media.is_video(path))
        save_progress(target, **state)
        print(f"Caption {state['completed']}/{state['total']}", flush=True)
    state['status'] = 'completed'
    save_progress(target, **state)


if __name__ == '__main__':
    main()
