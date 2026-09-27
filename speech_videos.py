"""Расшифровка речи в роликах: субтитры и поиск по сказанному.

Звук достаётся прямо из видеофайла через PyAV, который приезжает вместе с
faster-whisper, — отдельный ffmpeg ставить не нужно. Распознаёт Whisper
large-v3 на видеокарте: с пословными таймкодами выходит примерно полчаса
записи за минуту счёта.

Результат лежит в двух таблицах: в `video_speech` — весь текст ролика одной
строкой (по нему ищут) и состояние обработки, в `video_speech_segments` —
реплики с таймкодами (из них собираются субтитры и лента ролика).
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sqlite3
import time

import catalogdb
import pathkeys
import video as video_media

DEFAULT_MODEL = 'large-v3'
DEFAULT_LANGUAGE = 'ru'

# Whisper на тишине договаривает за себя: на беззвучном ролике он выдаёт
# «Thank you.» длиной в восемь сотых секунды и полную уверенность в норвежском.
# Поэтому реплика принимается, только если модель сама считает её речью.
MIN_SEGMENT_SECONDS = 0.35
MAX_NO_SPEECH = 0.6
MIN_LOGPROB = -1.0
# Ниже этой уверенности определению языка верить нельзя — берём язык архива.
MIN_LANGUAGE_PROBABILITY = 0.5


def connect(catalog):
    db = catalogdb.connect(catalog, timeout=30)
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS video_speech (
          path TEXT PRIMARY KEY REFERENCES photos(path) ON DELETE CASCADE,
          size INTEGER NOT NULL, modified INTEGER NOT NULL,
          language TEXT, language_probability REAL NOT NULL DEFAULT 0,
          text TEXT NOT NULL DEFAULT '', segments INTEGER NOT NULL DEFAULT 0,
          speech_seconds REAL NOT NULL DEFAULT 0,
          model TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL, error TEXT, analyzed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS video_speech_status ON video_speech(status);
        CREATE TABLE IF NOT EXISTS video_speech_segments (
          path TEXT NOT NULL REFERENCES photos(path) ON DELETE CASCADE,
          ord INTEGER NOT NULL,
          start REAL NOT NULL, stop REAL NOT NULL,
          text TEXT NOT NULL DEFAULT '', words_json TEXT NOT NULL DEFAULT '[]',
          PRIMARY KEY(path, ord)
        );
        CREATE INDEX IF NOT EXISTS video_speech_segments_start
          ON video_speech_segments(path, start);
    ''')
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


def scope_sql(roots, paths):
    """Условие «только выбранные папки и файлы» — как в остальных этапах."""
    return pathkeys.analysis_scope_sql(roots, paths, 'path')


def pending(db, args):
    """Ролики, которые ещё не расшифрованы этой моделью для этой версии файла."""
    where, values = scope_sql(args.root, args.path)
    fresh = '' if args.force else '''
        AND NOT EXISTS (SELECT 1 FROM video_speech s WHERE s.path=photos.path
                        AND s.size=photos.size AND s.modified=photos.modified
                        AND s.model=? AND s.status IN ('ok','silent'))'''
    model = [] if args.force else [args.model]
    return db.execute('''
        SELECT path, size, modified, COALESCE(duration,0) FROM photos
        WHERE status='ok' AND kind='video' ''' + fresh + where
        + ' ORDER BY path LIMIT ?', (*model, *values, args.limit)).fetchall()


def save(db, path, size, modified, model, info, segments):
    """Текст ролика и его реплики — одной транзакцией, чтобы не осталось половины."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    text = ' '.join(segment['text'] for segment in segments).strip()
    spoken = sum(segment['stop'] - segment['start'] for segment in segments)
    with db:
        db.execute('DELETE FROM video_speech_segments WHERE path=?', (path,))
        db.executemany(
            'INSERT INTO video_speech_segments(path,ord,start,stop,text,words_json)'
            ' VALUES(?,?,?,?,?,?)',
            [(path, index, segment['start'], segment['stop'], segment['text'],
              json.dumps(segment['words'], ensure_ascii=False))
             for index, segment in enumerate(segments)])
        db.execute(
            'INSERT INTO video_speech(path,size,modified,language,language_probability,'
            'text,segments,speech_seconds,model,status,error,analyzed_at)'
            ' VALUES(?,?,?,?,?,?,?,?,?,?,NULL,?)'
            ' ON CONFLICT(path) DO UPDATE SET size=excluded.size,modified=excluded.modified,'
            'language=excluded.language,language_probability=excluded.language_probability,'
            'text=excluded.text,segments=excluded.segments,'
            'speech_seconds=excluded.speech_seconds,model=excluded.model,'
            'status=excluded.status,error=NULL,analyzed_at=excluded.analyzed_at',
            # У ролика без речи языка нет: записать сюда догадку модели о шуме
            # значит потом искать по нему и удивляться.
            (path, size, modified, info.get('language') if segments else None,
             info.get('probability', 0.0) if segments else 0.0,
             text, len(segments), spoken, model,
             'ok' if segments else 'silent', now))


def fail(db, path, size, modified, model, status, message):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with db:
        db.execute(
            'INSERT INTO video_speech(path,size,modified,model,status,error,analyzed_at)'
            ' VALUES(?,?,?,?,?,?,?)'
            ' ON CONFLICT(path) DO UPDATE SET size=excluded.size,modified=excluded.modified,'
            'model=excluded.model,status=excluded.status,error=excluded.error,'
            'analyzed_at=excluded.analyzed_at',
            (path, size, modified, model, status, message[:500], now))


def pick_language(model, audio, wanted, fallback):
    """Язык записи: что распознал сам Whisper, а на сомнениях — язык архива.

    Автоопределение по паре секунд бормотания уверенно называет норвежский,
    после чего распознавание уезжает вместе с ним. Проще довериться архиву.
    """
    if wanted and wanted != 'auto':
        return wanted, 1.0, False
    try:
        # Без этого язык определяется по первым тридцати секундам — то есть по
        # шуму ветра и хлопкам дверей, а не по тому, что люди говорят.
        code, probability, _ = model.detect_language(
            audio, vad_filter=True, language_detection_segments=3)
    except Exception:
        return fallback, 0.0, True
    if probability >= MIN_LANGUAGE_PROBABILITY:
        return code, float(probability), False
    return fallback, float(probability), True


def keep(piece, text):
    """Похоже ли это на настоящую речь, а не на домысел модели о тишине."""
    if len(text) < 2:
        return False
    if (piece.end - piece.start) < MIN_SEGMENT_SECONDS:
        return False
    if (piece.no_speech_prob or 0) > MAX_NO_SPEECH:
        return False
    if (piece.avg_logprob or 0) < MIN_LOGPROB:
        return False
    return True


def transcribe(model, audio, language, fallback):
    """Реплики ролика простыми словарями — без типов faster-whisper наружу."""
    code, probability, guessed = pick_language(model, audio, language, fallback)
    pieces, _ = model.transcribe(
        audio, language=code, word_timestamps=True,
        vad_filter=True, beam_size=5, condition_on_previous_text=False)
    segments, dropped = [], 0
    for piece in pieces:
        text = (piece.text or '').strip()
        if not keep(piece, text):
            dropped += 1
            continue
        segments.append({
            'start': float(piece.start), 'stop': float(piece.end), 'text': text,
            'words': [{'w': (word.word or '').strip(), 's': round(float(word.start), 2),
                       'e': round(float(word.end), 2)}
                      for word in (piece.words or []) if (word.word or '').strip()],
        })
    return segments, {'language': code, 'probability': probability,
                      'guessed': guessed, 'dropped': dropped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--compute-type', default='float16')
    parser.add_argument('--language', default='auto',
                        help='Код языка или auto — определять по самой записи')
    parser.add_argument('--fallback-language', default=DEFAULT_LANGUAGE,
                        help='Язык архива: берётся, когда определению нельзя верить')
    parser.add_argument('--progress-file', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--root', action='append', type=str, default=[])
    parser.add_argument('--path', action='append', type=str, default=[])
    parser.add_argument('--force', action='store_true',
                        help='Расшифровать заново, даже если текст уже есть')
    args = parser.parse_args()

    db = connect(args.catalog.resolve())
    rows = pending(db, args)
    target = args.progress_file.resolve() if args.progress_file else None
    stop = args.stop_file.resolve() if args.stop_file else None
    state = {'status': 'preparing', 'phase': 'speech', 'total': len(rows), 'completed': 0,
             'with_speech': 0, 'silent': 0, 'errors': 0, 'current': '',
             'seconds_total': round(sum(row[3] for row in rows), 1), 'seconds_done': 0.0,
             'dropped': 0, 'model': args.model}
    progress(target, **state)
    if not rows:
        state['status'] = 'completed'
        progress(target, **state)
        return

    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    whisper = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    state['status'] = 'running'
    progress(target, **state)

    for path, size, modified, duration in rows:
        if stop and stop.exists():
            state['status'] = 'stopped'
            progress(target, **state)
            return
        state['current'] = path
        progress(target, **state)
        try:
            audio = decode_audio(video_media.local(path), sampling_rate=16000)
        except Exception as exc:
            # У части роликов дорожки просто нет — это не поломка, а свойство файла.
            fail(db, path, size, modified, args.model, 'silent', f'Нет звука: {exc}')
            state['silent'] += 1
        else:
            try:
                segments, info = transcribe(whisper, audio, args.language,
                                            args.fallback_language)
                save(db, path, size, modified, args.model, info, segments)
                state['with_speech' if segments else 'silent'] += 1
                state['dropped'] += info['dropped']
            except Exception as exc:
                fail(db, path, size, modified, args.model, 'error', f'Речь: {exc}')
                state['errors'] += 1
        state['completed'] += 1
        state['seconds_done'] = round(state['seconds_done'] + (duration or 0), 1)
        progress(target, **state)
        print(f"Речь {state['completed']}/{state['total']}", flush=True)

    state['current'] = ''
    state['status'] = 'completed'
    progress(target, **state)


if __name__ == '__main__':
    main()
