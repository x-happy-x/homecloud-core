"""Разделение голосов в роликах: кто говорил и когда, без знания кто именно.

pyannote.audio раскладывает звуковую дорожку на интервалы SPEAKER_00,
SPEAKER_01 и так далее — без заранее известного числа говорящих. Эти метки
существуют только внутри одного ролика: если во втором ролике снова
`SPEAKER_00`, это не обязательно тот же человек. Постоянная личность
появится позже, когда голосовой эмбеддинг сопоставят с сохранёнными
образцами (`voice_prints`) — здесь только эмбеддинг лежит, готовый для
сравнения.

Заодно, раз и диаризация, и расшифровка речи разбирают одну и ту же
дорожку, репликам из `video_speech_segments` сразу проставляется метка
говорящего — по тому, чей интервал перекрывает реплику дольше всего. Это не
требует видео и лиц: чистое совпадение по времени в пределах одного файла.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np

DEFAULT_MODEL = 'pyannote/speaker-diarization-community-1'

# На длинных роликах со сменой акустики (шум, музыка, разное расстояние до
# микрофона) один и тот же человек иногда расходится на два кластера — их
# эмбеддинги при этом остаются заметно ближе друг к другу, чем к чужим
# голосам. Порог подобран по разнице между «тем же человеком» (~0.69–0.70
# на проверенном ролике) и «разными людьми» (~0.12–0.31) — с запасом.
MERGE_SIMILARITY = 0.6


def connect(catalog):
    db = sqlite3.connect(Path(catalog) / 'catalog.sqlite', timeout=30)
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS video_diarization (
          path TEXT PRIMARY KEY REFERENCES photos(path) ON DELETE CASCADE,
          size INTEGER NOT NULL, modified INTEGER NOT NULL,
          speakers INTEGER NOT NULL DEFAULT 0, model TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL, error TEXT, analyzed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS video_speaker_turns (
          path TEXT NOT NULL REFERENCES photos(path) ON DELETE CASCADE,
          ord INTEGER NOT NULL,
          start REAL NOT NULL, stop REAL NOT NULL, speaker TEXT NOT NULL,
          PRIMARY KEY(path, ord)
        );
        CREATE INDEX IF NOT EXISTS video_speaker_turns_start
          ON video_speaker_turns(path, start);
        CREATE TABLE IF NOT EXISTS video_speakers (
          path TEXT NOT NULL REFERENCES photos(path) ON DELETE CASCADE,
          speaker TEXT NOT NULL,
          seconds REAL NOT NULL DEFAULT 0,
          embedding BLOB, dims INTEGER,
          PRIMARY KEY(path, speaker)
        );
    ''')
    columns = {row[1] for row in db.execute('PRAGMA table_info(video_speech_segments)')}
    if columns and 'speaker' not in columns:
        db.execute('ALTER TABLE video_speech_segments ADD COLUMN speaker TEXT')
    db.commit()
    return db


def progress(path, **state):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def scope_sql(roots, paths, column='path'):
    """Условие «только выбранные папки и файлы» — как в остальных этапах."""
    clauses, values = [], []
    for root in roots:
        value = str(Path(root).resolve()).rstrip('\\/')
        clauses.append(f'({column}=? OR {column} LIKE ?)')
        values.extend((value, value + os.sep + '%'))
    for path in paths:
        clauses.append(f'{column}=?')
        values.append(str(Path(path).resolve()))
    return (' AND (' + ' OR '.join(clauses) + ')' if clauses else ''), values


def pending(db, args):
    """Ролики с уже найденной речью, ещё не разложенные на говорящих."""
    where, values = scope_sql(args.root, args.path, 'photos.path')
    fresh = '' if args.force else '''
        AND NOT EXISTS (SELECT 1 FROM video_diarization d WHERE d.path=photos.path
                        AND d.size=photos.size AND d.modified=photos.modified
                        AND d.model=? AND d.status='ok')'''
    model = [] if args.force else [args.model]
    # Разбирать молчащий ролик незачем — там нечего разделять, а звук читать
    # и гонять через сеть всё равно придётся.
    return db.execute('''
        SELECT photos.path, photos.size, photos.modified,
               COALESCE(video_speech.speech_seconds,0)
        FROM photos JOIN video_speech ON video_speech.path=photos.path
        WHERE photos.status='ok' AND photos.kind='video'
          AND video_speech.status='ok' AND video_speech.text!='' '''
        + fresh + where + ' ORDER BY photos.path LIMIT ?',
        (*model, *values, args.limit)).fetchall()


def assign_speakers(db, path, turns):
    """Метка говорящего для уже расшифрованных реплик — по перекрытию времени."""
    segments = db.execute(
        'SELECT ord,start,stop FROM video_speech_segments WHERE path=? ORDER BY ord',
        (path,)).fetchall()
    if not segments or not turns:
        return
    updates = []
    for ord_, start, stop in segments:
        best_speaker, best_overlap = None, 0.0
        for turn_start, turn_stop, speaker in turns:
            overlap = min(stop, turn_stop) - max(start, turn_start)
            if overlap > best_overlap:
                best_speaker, best_overlap = speaker, overlap
        if best_speaker:
            updates.append((best_speaker, path, ord_))
    if updates:
        db.executemany(
            'UPDATE video_speech_segments SET speaker=? WHERE path=? AND ord=?', updates)


def save(db, path, size, modified, model, turns, speakers):
    """Реплики говорящих, голосовые эмбеддинги и метки на тексте — одной транзакцией."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with db:
        db.execute('DELETE FROM video_speaker_turns WHERE path=?', (path,))
        db.execute('DELETE FROM video_speakers WHERE path=?', (path,))
        db.executemany(
            'INSERT INTO video_speaker_turns(path,ord,start,stop,speaker) VALUES(?,?,?,?,?)',
            [(path, index, start, stop, speaker)
             for index, (start, stop, speaker) in enumerate(turns)])
        db.executemany(
            'INSERT INTO video_speakers(path,speaker,seconds,embedding,dims) VALUES(?,?,?,?,?)',
            [(path, speaker, seconds, embedding.astype(np.float32).tobytes(),
              embedding.shape[0])
             for speaker, seconds, embedding in speakers])
        assign_speakers(db, path, turns)
        db.execute(
            'INSERT INTO video_diarization(path,size,modified,speakers,model,status,error,'
            'analyzed_at) VALUES(?,?,?,?,?,?,NULL,?)'
            ' ON CONFLICT(path) DO UPDATE SET size=excluded.size,modified=excluded.modified,'
            'speakers=excluded.speakers,model=excluded.model,status=excluded.status,'
            'error=NULL,analyzed_at=excluded.analyzed_at',
            (path, size, modified, len(speakers), model,
             'ok' if turns else 'silent', now))


def fail(db, path, size, modified, model, status, message):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with db:
        db.execute(
            'INSERT INTO video_diarization(path,size,modified,model,status,error,analyzed_at)'
            ' VALUES(?,?,?,?,?,?,?)'
            ' ON CONFLICT(path) DO UPDATE SET size=excluded.size,modified=excluded.modified,'
            'model=excluded.model,status=excluded.status,error=excluded.error,'
            'analyzed_at=excluded.analyzed_at',
            (path, size, modified, model, status, message[:500], now))


def merge_close_speakers(turns, speakers, threshold=MERGE_SIMILARITY):
    """Склеивает кластеры, которые почти наверняка один и тот же голос.

    pyannote иногда разводит одного человека на два кластера, если акустика
    ролика меняется по ходу записи. Эмбеддинги такой пары остаются заметно
    ближе друг к другу, чем к любому третьему голосу, — это и есть сигнал
    для склейки, без разметки руками.
    """
    if len(speakers) < 2:
        return turns, speakers
    labels = [label for label, _, _ in speakers]
    vectors = {label: embedding / (np.linalg.norm(embedding) or 1)
               for label, _, embedding in speakers}
    parent = {label: label for label in labels}

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            if float(np.dot(vectors[a], vectors[b])) >= threshold:
                union(a, b)

    mapping = {label: find(label) for label in labels}
    if len(set(mapping.values())) == len(labels):
        return turns, speakers  # склеивать было нечего

    merged_turns = [(start, stop, mapping[speaker]) for start, stop, speaker in turns]
    seconds = {label: total for label, total, _ in speakers}
    merged_speakers = []
    for canonical in sorted(set(mapping.values())):
        members = [label for label in labels if mapping[label] == canonical]
        total = sum(seconds[label] for label in members)
        # Эмбеддинг склеенного голоса — средний по говорившей доле времени,
        # а не по кластерам: длинный кусок должен весить больше короткого.
        weighted = sum(vectors[label] * seconds[label] for label in members)
        weighted = weighted / (np.linalg.norm(weighted) or 1)
        merged_speakers.append((canonical, round(total, 2), weighted))
    return merged_turns, merged_speakers


def diarize(pipeline, audio):
    """Интервалы говорящих и их эмбеддинги — простыми объектами, без pyannote наружу."""
    import torch
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    output = pipeline({'waveform': waveform, 'sample_rate': 16000})
    turns = [(float(turn.start), float(turn.end), str(speaker))
             for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True)]
    seconds = {}
    for start, stop, speaker in turns:
        seconds[speaker] = seconds.get(speaker, 0.0) + (stop - start)
    labels = list(output.speaker_diarization.labels())
    speakers = []
    if output.speaker_embeddings is not None:
        for label, embedding in zip(labels, output.speaker_embeddings):
            speakers.append((label, round(seconds.get(label, 0.0), 2), embedding))
    return turns, speakers


def find_token(explicit):
    """Токен HuggingFace: явно передан, в окружении, либо в файле рядом со скриптом.

    Доступ к весам pyannote требует принятия условий на странице модели —
    без токена от аккаунта, который их принял, закачка не начнётся вовсе.
    """
    if explicit:
        return explicit
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN']
    token_file = Path(__file__).resolve().parent / 'hf-token.txt'
    if token_file.is_file():
        return token_file.read_text(encoding='utf-8').strip()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hf-token', default='')
    parser.add_argument('--progress-file', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--root', action='append', type=Path, default=[])
    parser.add_argument('--path', action='append', type=Path, default=[])
    parser.add_argument('--force', action='store_true',
                        help='Разложить заново, даже если уже посчитано')
    args = parser.parse_args()

    db = connect(args.catalog.resolve())
    rows = pending(db, args)
    target = args.progress_file.resolve() if args.progress_file else None
    stop = args.stop_file.resolve() if args.stop_file else None
    state = {'status': 'preparing', 'phase': 'diarize', 'total': len(rows), 'completed': 0,
             'multi_speaker': 0, 'single_speaker': 0, 'errors': 0, 'current': '',
             'seconds_total': round(sum(row[3] for row in rows), 1), 'seconds_done': 0.0,
             'model': args.model}
    progress(target, **state)
    if not rows:
        state['status'] = 'completed'
        progress(target, **state)
        return

    token = find_token(args.hf_token)
    if not token:
        state['status'] = 'error'
        state['error'] = ('Нет токена Hugging Face: положите его в hf-token.txt рядом '
                          'со скриптом, в переменную HF_TOKEN или передайте --hf-token')
        progress(target, **state)
        raise SystemExit(state['error'])

    import torch
    from pyannote.audio import Pipeline
    from faster_whisper.audio import decode_audio
    pipeline = Pipeline.from_pretrained(args.model, token=token)
    pipeline.to(torch.device(args.device))
    state['status'] = 'running'
    progress(target, **state)

    for path, size, modified, speech_seconds in rows:
        if stop and stop.exists():
            state['status'] = 'stopped'
            progress(target, **state)
            return
        state['current'] = path
        progress(target, **state)
        try:
            audio = decode_audio(path, sampling_rate=16000)
            turns, speakers = diarize(pipeline, audio)
            turns, speakers = merge_close_speakers(turns, speakers)
            save(db, path, size, modified, args.model, turns, speakers)
            state['multi_speaker' if len(speakers) > 1 else 'single_speaker'] += 1
        except Exception as exc:
            fail(db, path, size, modified, args.model, 'error', f'Диаризация: {exc}')
            state['errors'] += 1
        state['completed'] += 1
        state['seconds_done'] = round(state['seconds_done'] + (speech_seconds or 0), 1)
        progress(target, **state)
        print(f"Голоса {state['completed']}/{state['total']}", flush=True)

    state['current'] = ''
    state['status'] = 'completed'
    progress(target, **state)


if __name__ == '__main__':
    main()
