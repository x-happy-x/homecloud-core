"""Fully local quality, classification and semantic indexing for the photo catalog."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

import catalogdb
import pathkeys
import video as video_media


DEFAULT_MODEL = 'google/siglip2-base-patch16-224'
JINA_MODEL_REVISION = 'e10d47f5691d0454a0fb5d13f46f2199b74cb436'
JINA_CODE_REVISION = '39e6a55ae971b59bea6e44675d237c99762e7ee2'
CONTENT_LABELS = {
    'photo': 'a natural camera photograph',
    'screenshot': 'a computer or phone screenshot',
    'document': 'a document, receipt, book page or scanned paper',
    'graphics': 'digital artwork, illustration, icon, diagram or computer graphics',
    'game': 'a screenshot from a video game',
    'meme': 'an internet meme with text',
}


def connect(catalog):
    db = catalogdb.connect(catalog, timeout=30)
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS photo_analysis (
          path TEXT PRIMARY KEY REFERENCES photos(path) ON DELETE CASCADE,
          size INTEGER NOT NULL, modified INTEGER NOT NULL,
          width INTEGER, height INTEGER, blur_score REAL,
          content_type TEXT, content_confidence REAL,
          caption TEXT NOT NULL DEFAULT '', ocr_text TEXT NOT NULL DEFAULT '',
          caption_short TEXT NOT NULL DEFAULT '', caption_search TEXT NOT NULL DEFAULT '',
          caption_tags_json TEXT NOT NULL DEFAULT '[]', caption_json TEXT NOT NULL DEFAULT '{}',
          embedding BLOB, embedding_dims INTEGER, embedding_model TEXT,
          status TEXT NOT NULL, error TEXT, analyzed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS photo_analysis_type ON photo_analysis(content_type);
        CREATE INDEX IF NOT EXISTS photo_analysis_blur ON photo_analysis(blur_score);
        CREATE TABLE IF NOT EXISTS photo_embeddings (
          path TEXT NOT NULL REFERENCES photos(path) ON DELETE CASCADE,
          model TEXT NOT NULL, size INTEGER NOT NULL, modified INTEGER NOT NULL,
          embedding BLOB NOT NULL, dims INTEGER NOT NULL,
          analyzed_at TEXT NOT NULL,
          PRIMARY KEY(path,model)
        );
        CREATE INDEX IF NOT EXISTS photo_embeddings_model ON photo_embeddings(model);
        CREATE TABLE IF NOT EXISTS photo_adult_analysis (
          path TEXT PRIMARY KEY REFERENCES photos(path) ON DELETE CASCADE,
          size INTEGER NOT NULL, modified INTEGER NOT NULL,
          rating TEXT NOT NULL DEFAULT 'unknown', adult_score REAL NOT NULL DEFAULT 0,
          tags_json TEXT NOT NULL DEFAULT '[]', regions_json TEXT NOT NULL DEFAULT '[]',
          description TEXT NOT NULL DEFAULT '',
          detector_model TEXT, tagger_model TEXT,
          status TEXT NOT NULL, error TEXT, analyzed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS photo_adult_rating ON photo_adult_analysis(rating);
        CREATE INDEX IF NOT EXISTS photo_adult_score ON photo_adult_analysis(adult_score);
    ''')
    columns = {row[1] for row in db.execute('PRAGMA table_info(photo_analysis)')}
    if 'ocr_status' not in columns:
        db.execute("ALTER TABLE photo_analysis ADD COLUMN ocr_status TEXT")
    if 'caption_status' not in columns:
        db.execute("ALTER TABLE photo_analysis ADD COLUMN caption_status TEXT")
    for name, declaration in (
            ('caption_short', "TEXT NOT NULL DEFAULT ''"),
            ('caption_search', "TEXT NOT NULL DEFAULT ''"),
            ('caption_tags_json', "TEXT NOT NULL DEFAULT '[]'"),
            ('caption_json', "TEXT NOT NULL DEFAULT '{}'") ):
        if name not in columns:
            db.execute(f'ALTER TABLE photo_analysis ADD COLUMN {name} {declaration}')
    embedding_columns = {row[1] for row in db.execute('PRAGMA table_info(photo_embeddings)')}
    for name in ('size', 'modified'):
        if name not in embedding_columns:
            db.execute(f'ALTER TABLE photo_embeddings ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0')
    db.execute('''UPDATE photo_embeddings SET
                  size=COALESCE((SELECT photos.size FROM photos WHERE photos.path=photo_embeddings.path),size),
                  modified=COALESCE((SELECT photos.modified FROM photos WHERE photos.path=photo_embeddings.path),modified)
                  WHERE size=0 OR modified=0''')
    db.commit()
    # Сохраняем уже рассчитанный прежним кодом индекс как один из вариантов.
    with db:
        db.execute('''INSERT OR IGNORE INTO photo_embeddings
                      (path,model,size,modified,embedding,dims,analyzed_at)
                      SELECT path,embedding_model,size,modified,embedding,embedding_dims,analyzed_at
                      FROM photo_analysis WHERE embedding IS NOT NULL
                      AND embedding_model IS NOT NULL AND embedding_dims IS NOT NULL''')
    return db


def write_progress(path, **payload):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload['updated_at'] = time.time()
    payload['pid'] = os.getpid()
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
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


class VisualEncoder:
    """One interface for native Transformers CLIP models and Jina CLIP v2."""
    def __init__(self, model_name):
        import torch
        from transformers import AutoModel, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in the vision environment')
        self.torch = torch
        self.name = model_name
        self.jina = model_name == 'jinaai/jina-clip-v2'
        if self.jina:
            # Jina's pinned adapter imports a training-only helper removed in
            # Transformers 5. It is not used for inference, but must exist for
            # the module to import alongside the newer Qwen-compatible stack.
            import transformers.models.clip.modeling_clip as clip_modeling
            if not hasattr(clip_modeling, 'clip_loss'):
                def contrastive_loss(logits):
                    labels = torch.arange(len(logits), device=logits.device)
                    return torch.nn.functional.cross_entropy(logits, labels)

                def clip_loss(similarity):
                    return (contrastive_loss(similarity) +
                            contrastive_loss(similarity.T)) / 2
                clip_modeling.clip_loss = clip_loss
        self.processor = None if self.jina else AutoProcessor.from_pretrained(model_name)
        extra = ({'revision': JINA_MODEL_REVISION, 'code_revision': JINA_CODE_REVISION}
                 if self.jina else {})
        self.model = AutoModel.from_pretrained(
            model_name, dtype=torch.float16, low_cpu_mem_usage=True,
            trust_remote_code=self.jina, **extra).eval().to('cuda')
        if self.jina:
            from transformers import AutoImageProcessor, AutoTokenizer
            self.model.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, fix_mistral_regex=True,
                revision=JINA_MODEL_REVISION)
            self.model.preprocess = AutoImageProcessor.from_pretrained(
                model_name, trust_remote_code=True, revision=JINA_MODEL_REVISION,
                code_revision=JINA_CODE_REVISION)

    def images(self, images):
        torch = self.torch
        with torch.inference_mode():
            if self.jina:
                value = self.model.encode_image(images, truncate_dim=512)
            else:
                inputs = self.processor(images=images, return_tensors='pt').to('cuda')
                value = feature_tensor(self.model.get_image_features(**inputs))
            value = torch.as_tensor(value, device='cuda')
            return torch.nn.functional.normalize(value, dim=-1)

    def texts(self, texts):
        torch = self.torch
        with torch.inference_mode():
            if self.jina:
                value = self.model.encode_text(texts, task='retrieval.query', truncate_dim=512)
            else:
                inputs = self.processor(
                    text=texts, padding='max_length', return_tensors='pt').to('cuda')
                value = feature_tensor(self.model.get_text_features(**inputs))
            value = torch.as_tensor(value, device='cuda')
            return torch.nn.functional.normalize(value, dim=-1)


def load_siglip(model_name):
    encoder = VisualEncoder(model_name)
    return encoder.torch, encoder, encoder.model


def quality(image):
    import cv2
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    scale = min(1.0, 768 / max(width, height))
    if scale < 1:
        gray = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def feature_tensor(value):
    """Normalize the Transformers 4.x tensor and 5.x model-output APIs."""
    return value.pooler_output if hasattr(value, 'pooler_output') else value


def classify_embeddings(torch, processor, model, images):
    prompts = list(CONTENT_LABELS.values())
    with torch.inference_mode():
        image_vectors = processor.images(images)
        text_vectors = processor.texts(prompts)
        scores = image_vectors @ text_vectors.T
        probabilities = torch.softmax(scores * 20, dim=-1)
        confidence, labels = probabilities.max(dim=-1)
    names = list(CONTENT_LABELS)
    return (image_vectors.float().cpu().numpy(),
            [names[index] for index in labels.cpu().tolist()],
            confidence.float().cpu().tolist())


def analyze(args):
    catalog = args.catalog.resolve()
    db = connect(catalog)
    # Папки и снимки задания — ключи источников (см. pathkeys.py).
    root_sql, root_values = pathkeys.analysis_scope_sql(args.root, args.path, 'photos.path')
    # Пересчитываем только новое и изменившееся: снимок версии файла — размер и mtime.
    fresh_sql = '' if args.force else '''
        AND (selected_embedding.path IS NULL OR selected_embedding.size != photos.size OR
             selected_embedding.modified != photos.modified)'''
    fresh_values = (args.model,)
    rows = db.execute('''
        SELECT photos.path,photos.size,photos.modified
        FROM photos LEFT JOIN photo_analysis USING(path)
        LEFT JOIN photo_embeddings AS selected_embedding
          ON selected_embedding.path=photos.path AND selected_embedding.model=?
        WHERE photos.status='ok'
          AND COALESCE(photos.blocked,0)=0 '''
        + fresh_sql + root_sql + ' ORDER BY photos.path LIMIT ?',
        (*fresh_values, *root_values, args.limit)).fetchall()
    rows = video_media.only(args.kinds, rows)
    progress_path = args.progress_file.resolve() if args.progress_file else None
    stop_path = args.stop_file.resolve() if args.stop_file else None
    state = dict(status='preparing', total=len(rows), completed=0, indexed=0,
                 errors=0, blurry=0, graphics=0, current='', source=str(catalog),
                 videos_total=sum(video_media.is_video(row[0]) for row in rows),
                 videos_done=0, video_frames=2)
    write_progress(progress_path, **state)
    if not rows:
        state['status'] = 'completed'
        write_progress(progress_path, **state)
        print('No new photos to analyze.')
        return
    # Текст и описание принадлежат прежней версии файла — после правки их надо считать заново.
    stale = 'photo_analysis.size!=excluded.size OR photo_analysis.modified!=excluded.modified'
    reset_sql = (f",ocr_status=CASE WHEN {stale} THEN NULL ELSE photo_analysis.ocr_status END"
                 f",caption_status=CASE WHEN {stale} THEN NULL ELSE photo_analysis.caption_status END"
                 f",ocr_text=CASE WHEN {stale} THEN '' ELSE photo_analysis.ocr_text END"
                 f",caption=CASE WHEN {stale} THEN '' ELSE photo_analysis.caption END"
                 f",caption_short=CASE WHEN {stale} THEN '' ELSE photo_analysis.caption_short END"
                 f",caption_search=CASE WHEN {stale} THEN '' ELSE photo_analysis.caption_search END"
                 f",caption_tags_json=CASE WHEN {stale} THEN '[]' ELSE photo_analysis.caption_tags_json END"
                 f",caption_json=CASE WHEN {stale} THEN '{{}}' ELSE photo_analysis.caption_json END")
    torch, processor, model = load_siglip(args.model)
    state['status'] = 'running'
    write_progress(progress_path, **state)
    for offset in range(0, len(rows), args.batch_size):
        if stop_path and stop_path.exists():
            state['status'] = 'stopped'
            write_progress(progress_path, **state)
            return
        batch_rows = rows[offset:offset + args.batch_size]
        batch_videos = sum(video_media.is_video(row[0]) for row in batch_rows)
        loaded = []
        for row in batch_rows:
            try:
                image = video_media.open_frame(row[0])
                loaded.append((row, image, quality(image)))
            except Exception as exc:
                state['errors'] += 1
                state['completed'] += 1
                with db:
                    db.execute('''INSERT INTO photo_analysis
                        (path,size,modified,status,error,analyzed_at)
                        VALUES(?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET
                        size=excluded.size,modified=excluded.modified,status='error',
                        error=excluded.error,analyzed_at=excluded.analyzed_at''',
                        (row[0], row[1], row[2], 'error', str(exc),
                         datetime.now(timezone.utc).isoformat()))
        if not loaded:
            continue
        try:
            vectors, labels, confidences = classify_embeddings(
                torch, processor, model, [item[1] for item in loaded])
            now = datetime.now(timezone.utc).isoformat()
            values = []
            for (row, image, blur), vector, label, confidence in zip(
                    loaded, vectors, labels, confidences):
                width, height = image.size
                values.append((row[0], row[1], row[2], width, height, blur,
                               label, confidence, '', '',
                               np.asarray(vector, dtype='<f4').tobytes(), len(vector),
                               args.model, 'ok', None, now))
                state['indexed'] += 1
                state['completed'] += 1
                state['blurry'] += blur < args.blur_threshold
                state['graphics'] += label in {'graphics', 'game', 'screenshot', 'meme'}
                state['current'] = row[0]
            with db:
                db.executemany('''INSERT INTO photo_analysis
                    (path,size,modified,width,height,blur_score,content_type,
                     content_confidence,caption,ocr_text,embedding,embedding_dims,
                     embedding_model,status,error,analyzed_at) VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET
                    size=excluded.size,modified=excluded.modified,width=excluded.width,
                    height=excluded.height,blur_score=excluded.blur_score,
                    content_type=excluded.content_type,
                    content_confidence=excluded.content_confidence,
                    embedding=excluded.embedding,embedding_dims=excluded.embedding_dims,
                    embedding_model=excluded.embedding_model,status='ok',error=NULL,
                    analyzed_at=excluded.analyzed_at''' + reset_sql, values)
                db.executemany('''INSERT INTO photo_embeddings
                    (path,model,size,modified,embedding,dims,analyzed_at)
                    VALUES(?,?,?,?,?,?,?) ON CONFLICT(path,model) DO UPDATE SET
                    size=excluded.size,modified=excluded.modified,
                    embedding=excluded.embedding,dims=excluded.dims,
                    analyzed_at=excluded.analyzed_at''',
                    [(value[0], args.model, value[1], value[2], value[10], value[11], now)
                     for value in values])
        except Exception as exc:
            state['errors'] += len(loaded)
            state['completed'] += len(loaded)
            print(f'Batch error: {exc}', file=sys.stderr, flush=True)
        state['videos_done'] += batch_videos
        write_progress(progress_path, **state)
        print(f"{state['completed']}/{state['total']} indexed; {state['errors']} errors", flush=True)
    state['status'] = 'completed'
    write_progress(progress_path, **state)


def query(args):
    torch, processor, model = load_siglip(args.model)
    with torch.inference_mode():
        vector = processor.texts([args.text]).float().cpu().numpy()[0]
    db = connect(args.catalog.resolve())
    rows = db.execute('SELECT path,embedding FROM photo_embeddings WHERE model=?',
                      (args.model,)).fetchall()
    scored = []
    for path, blob in rows:
        image_vector = np.frombuffer(blob, dtype='<f4')
        scored.append((float(image_vector @ vector), path))
    scored.sort(reverse=True)
    print(json.dumps([{'score': score, 'path': path} for score, path in scored[:args.top]],
                     ensure_ascii=False))


def serve(args):
    torch, processor, model = load_siglip(args.model)
    db = connect(args.catalog.resolve())
    cache = {'stamp': None, 'paths': [], 'matrix': np.zeros((0, 1), dtype='<f4')}

    def library():
        """Векторы держим в памяти: иначе каждый запрос перечитывает весь индекс."""
        stamp = db.execute(
            "SELECT COUNT(*),MAX(analyzed_at) FROM photo_embeddings WHERE model=?",
            (args.model,)).fetchone()
        if cache['stamp'] != stamp:
            rows = db.execute('SELECT path,embedding FROM photo_embeddings WHERE model=?',
                              (args.model,)).fetchall()
            cache['paths'] = [row[0] for row in rows]
            cache['matrix'] = (np.stack([np.frombuffer(row[1], dtype='<f4') for row in rows])
                               if rows else np.zeros((0, 1), dtype='<f4'))
            cache['stamp'] = stamp
        return cache['paths'], cache['matrix']

    print(json.dumps({'ready': True}), flush=True)
    library()   # прогреваем индекс сразу: первый поиск не должен ждать
    for line in sys.stdin:
        try:
            request = json.loads(line)
            text = str(request.get('text', '')).strip()
            top = min(1000, max(1, int(request.get('top', 500))))
            with torch.inference_mode():
                vector = processor.texts([text]).float().cpu().numpy()[0]
            paths, matrix = library()
            if len(paths):
                scores = matrix @ vector
                order = np.argsort(-scores)[:top]
                scored = [(float(scores[position]), paths[position]) for position in order]
            else:
                scored = []
            print(json.dumps({'results': [{'score': score, 'path': path}
                                          for score, path in scored[:top]]},
                             ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({'error': str(exc)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    analyze_parser = sub.add_parser('analyze')
    analyze_parser.add_argument('--catalog', type=Path, required=True)
    analyze_parser.add_argument('--model', default=DEFAULT_MODEL)
    analyze_parser.add_argument('--limit', type=int, default=1000)
    analyze_parser.add_argument('--batch-size', type=int, default=16)
    analyze_parser.add_argument('--blur-threshold', type=float, default=65)
    analyze_parser.add_argument('--force', action='store_true',
                                help='Пересчитать даже то, что уже посчитано')
    analyze_parser.add_argument('--progress-file', type=Path)
    analyze_parser.add_argument('--stop-file', type=Path)
    analyze_parser.add_argument('--root', action='append', type=str, default=[])
    analyze_parser.add_argument('--path', action='append', type=str, default=[])
    analyze_parser.add_argument('--kinds', choices=video_media.KINDS, default='all',
                                help='Считать снимки, ролики или всё сразу')
    query_parser = sub.add_parser('query')
    query_parser.add_argument('--catalog', type=Path, required=True)
    query_parser.add_argument('--model', default=DEFAULT_MODEL)
    query_parser.add_argument('--text', required=True)
    query_parser.add_argument('--top', type=int, default=50)
    serve_parser = sub.add_parser('serve')
    serve_parser.add_argument('--catalog', type=Path, required=True)
    serve_parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.command == 'analyze':
        analyze(args)
    elif args.command == 'query':
        query(args)
    else:
        serve(args)


if __name__ == '__main__':
    main()
