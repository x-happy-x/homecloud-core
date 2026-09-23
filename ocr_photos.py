"""Local Russian/English OCR pass for likely screenshots and documents."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from analyze_photos import connect
import pathkeys
import video as video_media


def progress(path, **state):
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


def result_text(results):
    lines = []
    for result in results:
        payload = result.json if hasattr(result, 'json') else result
        if callable(payload):
            payload = payload()
        payload = payload.get('res', payload) if isinstance(payload, dict) else {}
        texts = payload.get('rec_texts', [])
        scores = payload.get('rec_scores', [1] * len(texts))
        lines.extend(str(text).strip() for text, score in zip(texts, scores)
                     if str(text).strip() and float(score) >= 0.45)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--progress-file', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--root', action='append', type=str, default=[])
    parser.add_argument('--path', action='append', type=str, default=[])
    parser.add_argument('--kinds', choices=video_media.KINDS, default='all',
                        help='Считать снимки, ролики или всё сразу')
    parser.add_argument('--force', action='store_true',
                        help='Распознать текст заново, даже если он уже есть')
    args = parser.parse_args()
    db = connect(args.catalog.resolve())
    # Папки и снимки задания — ключи источников (см. pathkeys.py).
    root_sql, root_values = pathkeys.scope_sql(args.root, args.path, 'path')
    candidate_sql = '' if args.path else ''' AND
          (content_type IN ('screenshot','document','meme') OR
           lower(path) LIKE '%screenshot%' OR lower(path) LIKE '%скриншот%')'''
    fresh_sql = '' if args.force else " AND (ocr_status IS NULL OR ocr_status='error')"
    rows = db.execute('''
        SELECT path FROM photo_analysis
        WHERE status='ok' ''' + fresh_sql
        + candidate_sql + root_sql + ' ORDER BY path LIMIT ?',
        (*root_values, args.limit)).fetchall()
    rows = video_media.only(args.kinds, rows)
    state = {'status': 'preparing', 'phase': 'ocr', 'total': len(rows),
             'completed': 0, 'with_text': 0, 'errors': 0, 'current': '',
             'videos_total': sum(video_media.is_video(row[0]) for row in rows),
             'videos_done': 0, 'video_frames': 2}
    target = args.progress_file.resolve() if args.progress_file else None
    stop = args.stop_file.resolve() if args.stop_file else None
    progress(target, **state)
    if not rows:
        state['status'] = 'completed'
        progress(target, **state)
        return
    from paddleocr import PaddleOCR
    pipeline = PaddleOCR(
        lang='ru', device='cpu', use_doc_orientation_classify=False,
        use_doc_unwarping=False, use_textline_orientation=False,
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='eslav_PP-OCRv5_mobile_rec',
        text_det_limit_side_len=1600, text_recognition_batch_size=4,
        enable_mkldnn=False)
    state['status'] = 'running'
    progress(target, **state)
    for (path,) in rows:
        if stop and stop.exists():
            state['status'] = 'stopped'
            progress(target, **state)
            return
        state['current'] = path
        try:
            # PaddleOCR открывает файл сам и видео не понимает — отдаём готовый
            # кадр (BGR-массив), путь оставляем только для настоящих фото.
            source = path
            if video_media.is_video(path):
                source = np.asarray(video_media.open_frame(path))[:, :, ::-1].copy()
            text = result_text(pipeline.predict(source))
            with db:
                db.execute("UPDATE photo_analysis SET ocr_text=?,ocr_status='ok' WHERE path=?",
                           (text, path))
            state['with_text'] += bool(text)
        except Exception as exc:
            with db:
                db.execute("UPDATE photo_analysis SET ocr_status='error',error=? WHERE path=?",
                           (f'OCR: {exc}', path))
            state['errors'] += 1
        state['completed'] += 1
        state['videos_done'] += int(video_media.is_video(path))
        progress(target, **state)
        print(f"OCR {state['completed']}/{state['total']}", flush=True)
    state['status'] = 'completed'
    progress(target, **state)


if __name__ == '__main__':
    main()
