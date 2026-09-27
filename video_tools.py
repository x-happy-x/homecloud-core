"""Обработка роликов через ffmpeg на ядре.

Операции (MediaJobs.start, поле op):

* ``replace`` — перекодировать ролик в H.264/AAC MP4 и заменить им исходник
  прямо в источнике: новый файл пишется рядом, оригинал уходит в корзину
  источника, а хаб переносит записи каталога на новый путь;
* ``trim`` — фрагмент от start до end, MP4;
* ``compress`` — облегчённая копия с высотой 720/480/360, MP4;
* ``rotate`` — поворот на 90/180/270 по часовой, MP4;
* ``frame`` — кадр на секунде time, JPEG в полном размере;
* ``audio`` — звуковая дорожка, MP3.

Всё, кроме replace, остаётся файлом на ядре (core-data/media-out) и отдаётся
на скачивание через хаб; через сутки такие файлы удаляются.

ffmpeg ставится компонентом ядра (components.py, «Видео: ffmpeg») в
tools/ffmpeg; ещё его можно указать в HOMECLOUD_FFMPEG или положить в PATH.
H.264 кодируется на видеокарте (h264_nvenc), если она есть, иначе libx264.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time

import envs
import pathkeys
import sources

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
OUTPUT_TTL = 24 * 3600
OPS = ('replace', 'trim', 'compress', 'rotate', 'frame', 'audio')
HEIGHTS = (720, 480, 360)
ROTATIONS = {90: 'transpose=1', 180: 'transpose=1,transpose=1', 270: 'transpose=2'}
# HDR (PQ и HLG, у телефонов — ещё и Dolby Vision поверх) в обычный BT.709: без
# этого H.264 из HDR-ролика выходит серым и блёклым. zscale — из libzimg сборки.
HDR_TRANSFERS = {'smpte2084', 'arib-std-b67'}
TONEMAP = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,'
           'tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p')
SDR_TAGS = ['-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709']

# Что браузеры играют без плагинов: контейнер (как его называет ffprobe) и кодеки.
BROWSER_CONTAINERS = ('mov', 'mp4', 'm4a', 'webm', 'matroska')
BROWSER_VIDEO = {'h264', 'vp8', 'vp9', 'av1'}
BROWSER_AUDIO = {'aac', 'mp3', 'opus', 'vorbis', 'flac'}
# MKV и WebM ffprobe называет одинаково (matroska,webm); WebM — это только такие кодеки.
WEBM_VIDEO = {'vp8', 'vp9', 'av1'}
WEBM_AUDIO = {'opus', 'vorbis'}


# ---------- где ffmpeg ----------

def tools_dir(root=envs.ROOT):
    return Path(root) / 'tools' / 'ffmpeg'


def find_binary(name, root=envs.ROOT):
    """ffmpeg или ffprobe: HOMECLOUD_FFMPEG (папка или сам ffmpeg), tools/ffmpeg, PATH."""
    exe = name + ('.exe' if os.name == 'nt' else '')
    configured = os.environ.get('HOMECLOUD_FFMPEG')
    if configured:
        path = Path(configured)
        candidate = path / exe if path.is_dir() else path.with_name(exe)
        if candidate.is_file():
            return candidate
    folder = tools_dir(root)
    if folder.is_dir():
        for candidate in sorted(folder.rglob(exe)):
            if candidate.is_file():
                return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def available(root=envs.ROOT):
    return find_binary('ffmpeg', root) is not None and find_binary('ffprobe', root) is not None


_encoder_cache = {}


def h264_encoder(ffmpeg):
    """h264_nvenc, если видеокарта его реально тянет, иначе libx264. Проверка — один кадр."""
    key = str(ffmpeg)
    if key not in _encoder_cache:
        try:
            result = subprocess.run(
                [key, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                 'color=c=black:s=256x256:d=0.1', '-frames:v', '1', '-c:v', 'h264_nvenc',
                 '-f', 'null', '-'], capture_output=True, timeout=60, creationflags=NO_WINDOW)
            _encoder_cache[key] = 'h264_nvenc' if result.returncode == 0 else 'libx264'
        except (OSError, subprocess.SubprocessError):
            _encoder_cache[key] = 'libx264'
    return _encoder_cache[key]


# ---------- сведения о файле ----------

def probe(path, root=envs.ROOT):
    ffprobe = find_binary('ffprobe', root)
    if ffprobe is None:
        raise RuntimeError('На ядре нет ffmpeg: установите «Видео: ffmpeg» в окружениях ядра')
    result = subprocess.run(
        [str(ffprobe), '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams',
         str(path)], capture_output=True, timeout=120, creationflags=NO_WINDOW)
    if result.returncode != 0:
        detail = result.stderr.decode('utf-8', 'replace').strip().splitlines()
        raise RuntimeError('ffprobe не смог прочитать файл' + (f': {detail[-1]}' if detail else ''))
    return summarize(json.loads(result.stdout or b'{}'))


def summarize(raw):
    """Коротко о ролике из вывода ffprobe: контейнер, дорожки, длительность."""
    streams = raw.get('streams') or []
    fmt = raw.get('format') or {}
    video = next((item for item in streams if item.get('codec_type') == 'video'
                  and not (item.get('disposition') or {}).get('attached_pic')), None)
    audio = [item for item in streams if item.get('codec_type') == 'audio']
    subtitles = [item for item in streams if item.get('codec_type') == 'subtitle']

    def number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    rotation = 0
    if video:
        for side in video.get('side_data_list') or []:
            if 'rotation' in side:
                rotation = int(number(side['rotation'])) % 360
        rotation = rotation or int(number((video.get('tags') or {}).get('rotate'))) % 360
    info = {
        'container': fmt.get('format_name', ''),
        'duration': number(fmt.get('duration')) or number((video or {}).get('duration')),
        'size': int(number(fmt.get('size'))),
        'bitrate': int(number(fmt.get('bit_rate'))),
        'video': None,
        'audio': [{'codec': item.get('codec_name', ''), 'channels': item.get('channels') or 0,
                   'language': (item.get('tags') or {}).get('language', '')} for item in audio],
        'subtitles': len(subtitles),
    }
    if video:
        info['video'] = {
            'codec': video.get('codec_name', ''), 'profile': video.get('profile', ''),
            'width': int(video.get('width') or 0), 'height': int(video.get('height') or 0),
            'pix_fmt': video.get('pix_fmt', ''), 'fps': round(_rate(video.get('avg_frame_rate')), 3),
            'rotation': rotation, 'transfer': video.get('color_transfer', ''),
            'hdr': video.get('color_transfer', '') in HDR_TRANSFERS}
    info['browser'] = browser_ok(info)
    return info


def _rate(value):
    try:
        top, _, bottom = str(value or '0/1').partition('/')
        return float(top) / float(bottom or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def browser_ok(info):
    """Сыграет ли ролик браузер как есть. 10-битный H.264 браузеры не декодируют."""
    video = info.get('video')
    if not video:
        return False
    container = info.get('container', '')
    if not any(name in container.split(',') for name in BROWSER_CONTAINERS):
        return False
    if video['codec'] not in BROWSER_VIDEO:
        return False
    if video['codec'] == 'h264' and ('10' in (video.get('pix_fmt') or '')
                                     or '4:2:2' in (video.get('profile') or '')
                                     or '4:4:4' in (video.get('profile') or '')):
        return False
    # MKV Chrome открывает, Safari — нет: такой ролик перекодируем в MP4 всегда.
    if 'matroska' in container.split(',') and (
            video['codec'] not in WEBM_VIDEO
            or any(track['codec'] not in WEBM_AUDIO for track in info.get('audio') or [])):
        return False
    return all(track['codec'] in BROWSER_AUDIO for track in info.get('audio') or [])


def copyable(info):
    """Дорожки уже подходят MP4 и браузеру — достаточно переложить без перекодирования."""
    video = info.get('video') or {}
    return (video.get('codec') == 'h264' and video.get('pix_fmt') in {'yuv420p', 'yuvj420p'}
            and all(track['codec'] in {'aac', 'mp3'} for track in info.get('audio') or []))


# ---------- команды ----------

def video_codec_args(encoder, quality):
    """quality — условное качество: 20 почти без потерь, 26 заметно легче, 30 для отправки."""
    if encoder == 'h264_nvenc':
        return ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', str(quality),
                '-b:v', '0', '-pix_fmt', 'yuv420p', '-profile:v', 'high']
    return ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(quality - 2),
            '-pix_fmt', 'yuv420p', '-profile:v', 'high']


def output_name(key, op, params):
    stem = Path(pathkeys.name(key)).stem
    if op == 'frame':
        return f'{stem} кадр {_clock(params.get("time", 0))}.jpg'
    if op == 'audio':
        return f'{stem}.mp3'
    if op == 'trim':
        return f'{stem} {_clock(params["start"])}–{_clock(params["end"])}.mp4'
    if op == 'compress':
        return f'{stem} {params["height"]}p.mp4'
    if op == 'rotate':
        return f'{stem} поворот {params["angle"]}.mp4'
    return f'{stem}.mp4'


def _clock(seconds):
    seconds = int(float(seconds or 0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f'{hours}.{minutes:02d}.{secs:02d}' if hours else f'{minutes}.{secs:02d}'


def check_params(op, params, info=None):
    """Проверить и привести параметры операции; возвращает новые params."""
    if op not in OPS:
        raise ValueError('Неизвестная операция с видео')
    params = dict(params or {})
    duration = (info or {}).get('duration') or 0
    if op == 'trim':
        start, end = float(params.get('start') or 0), float(params.get('end') or 0)
        if start < 0 or end <= start:
            raise ValueError('Конец фрагмента должен быть позже начала')
        if duration and start >= duration:
            raise ValueError('Начало фрагмента за концом ролика')
        params.update(start=round(start, 3), end=round(min(end, duration) if duration else end, 3))
    elif op == 'compress':
        height = int(params.get('height') or 720)
        if height not in HEIGHTS:
            raise ValueError('Высота сжатой копии — 720, 480 или 360')
        params['height'] = height
    elif op == 'rotate':
        angle = int(params.get('angle') or 90) % 360
        if angle not in ROTATIONS:
            raise ValueError('Поворот — на 90, 180 или 270 градусов')
        params['angle'] = angle
    elif op == 'frame':
        moment = max(0.0, float(params.get('time') or 0))
        if duration:
            moment = min(moment, max(0.0, duration - 0.05))
        params['time'] = round(moment, 3)
    return params


def video_filters(info, *extra):
    """-vf для перекодирования: у HDR сначала тонмаппинг, потом остальное."""
    chain = [TONEMAP] if (info.get('video') or {}).get('hdr') else []
    chain.extend(item for item in extra if item)
    tags = SDR_TAGS if chain and chain[0] == TONEMAP else []
    return (['-vf', ','.join(chain)] if chain else []) + tags


def build_command(ffmpeg, op, params, source, target, info, encoder):
    """Команда ffmpeg для операции. Прогресс — в stdout (-progress pipe:1)."""
    head = [str(ffmpeg), '-hide_banner', '-nostdin', '-y', '-loglevel', 'error',
            '-progress', 'pipe:1', '-nostats']
    mp4 = ['-movflags', '+faststart', '-map_metadata', '0', '-f', 'mp4', str(target)]
    audio_aac = ['-c:a', 'aac', '-b:a', '192k']
    if op == 'frame':
        return [*head, '-ss', str(params['time']), '-i', str(source), '-frames:v', '1',
                *video_filters(info), '-q:v', '2', '-f', 'image2', str(target)]
    if op == 'audio':
        if not info.get('audio'):
            raise ValueError('В ролике нет звука')
        return [*head, '-i', str(source), '-map', '0:a:0', '-vn', '-c:a', 'libmp3lame',
                '-q:a', '2', '-f', 'mp3', str(target)]
    if not info.get('video'):
        raise ValueError('В файле нет видеодорожки')
    if op == 'replace':
        maps = ['-map', '0:v:0', '-map', '0:a?']
        if copyable(info):
            return [*head, '-i', str(source), *maps, '-c', 'copy', *mp4]
        audio = (['-c:a', 'copy'] if all(track['codec'] == 'aac' for track in info['audio'])
                 else audio_aac)
        return [*head, '-i', str(source), *maps, *video_filters(info),
                *video_codec_args(encoder, 20), *audio, *mp4]
    if op == 'trim':
        return [*head, '-ss', str(params['start']), '-to', str(params['end']), '-i', str(source),
                '-map', '0:v:0', '-map', '0:a:0?', *video_filters(info),
                *video_codec_args(encoder, 20), *audio_aac, *mp4]
    if op == 'compress':
        height = params['height']
        # Не увеличиваем: у ролика ниже заданной высоты размер остаётся своим.
        scale = f"scale=-2:'min({height},ih)'"
        return [*head, '-i', str(source), '-map', '0:v:0', '-map', '0:a:0?',
                *video_filters(info, scale), *video_codec_args(encoder, 28),
                '-c:a', 'aac', '-b:a', '128k', *mp4]
    if op == 'rotate':
        return [*head, '-i', str(source), '-map', '0:v:0', '-map', '0:a?',
                *video_filters(info, ROTATIONS[params['angle']]), '-metadata:s:v:0', 'rotate=0',
                *video_codec_args(encoder, 20), '-c:a', 'copy', *mp4]
    raise ValueError('Неизвестная операция с видео')


def expected_seconds(op, params, info):
    if op == 'trim':
        return max(0.001, params['end'] - params['start'])
    if op == 'frame':
        return 0
    return info.get('duration') or 0


def parse_progress(line, total):
    """Доля готового по строке -progress: out_time_us/out_time_ms (оба в микросекундах)."""
    key, _, value = line.strip().partition('=')
    if key in {'out_time_us', 'out_time_ms'} and total:
        try:
            return max(0.0, min(1.0, int(value) / 1_000_000 / total))
        except ValueError:
            return None
    if key == 'progress' and value == 'end':
        return 1.0
    return None


def verify(info, result, op, params):
    """Готовый ролик цел: есть видео, длительность сходится с ожидаемой."""
    if not result.get('video'):
        raise RuntimeError('В результате нет видеодорожки')
    want = expected_seconds(op, params, info)
    got = result.get('duration') or 0
    if want and abs(got - want) > max(2.0, want * 0.02):
        raise RuntimeError(f'Длительность не сходится: было {want:.1f} с, вышло {got:.1f} с')


def replacement_native(driver, native):
    """Куда писать перекодированный ролик: то же имя с .mp4; занято — «имя (2).mp4».
    Если исходник сам .mp4, новый файл сначала пишется под временным именем."""
    separator = '\\' if '\\' in native and '/' not in native else '/'
    folder, _, name = native.rpartition(separator)
    stem = name.rsplit('.', 1)[0] if '.' in name else name
    final = f'{folder}{separator}{stem}.mp4'
    if final.casefold() == native.casefold():
        return final, f'{folder}{separator}.{stem}.homecloud-new.mp4'
    number = 2
    while driver.exists(final):
        final = f'{folder}{separator}{stem} ({number}).mp4'
        number += 1
    return final, final


# ---------- задания ----------

class MediaJobs:
    """Очередь операций ядра: по одной за раз, видеокарта одна.

    access — sources.Access ядра (для записи в источник), stage(key) — путь к
    файлу на этой машине, trash(record, native) — отправить оригинал в корзину
    источника (у своего диска — корзина Windows).
    """

    def __init__(self, folder, access, stage, trash, root=envs.ROOT):
        self.folder = Path(folder)
        self.access = access
        self.stage = stage
        self.trash = trash
        self.root = root
        self.lock = threading.RLock()
        self.jobs = {}
        self.queue = []
        self.wakeup = threading.Event()
        self.process = None
        threading.Thread(target=self._worker, daemon=True).start()

    def public(self, job):
        return {key: value for key, value in job.items() if not key.startswith('_')}

    def probe(self, key):
        return probe(self.stage(key), self.root)

    def start(self, key, op, params=None):
        if op not in OPS:
            raise ValueError('Неизвестная операция с видео')
        if not available(self.root):
            raise ValueError('На ядре нет ffmpeg: установите «Видео: ffmpeg» в окружениях ядра')
        params = check_params(op, params)
        job_id = secrets.token_hex(8)
        job = {'id': job_id, 'key': key, 'op': op, 'params': params, 'status': 'queued',
               'progress': 0.0, 'message': 'В очереди', 'error': '', 'result': None,
               'created_at': time.time(), 'started_at': None, 'finished_at': None,
               '_cancel': False}
        with self.lock:
            self._cleanup()
            self.jobs[job_id] = job
            self.queue.append(job_id)
        self.wakeup.set()
        return self.public(job)

    def status(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError('Задание не найдено')
            return self.public(job)

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError('Задание не найдено')
            job['_cancel'] = True
            if job['status'] == 'queued':
                self.queue.remove(job_id)
                job.update(status='cancelled', message='Отменено', finished_at=time.time())
            process = self.process if job['status'] == 'running' else None
        if process and process.poll() is None:
            process.kill()
        return self.status(job_id)

    def file(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job['status'] != 'done' or not job.get('_file'):
                raise KeyError('Файл задания не готов')
            return Path(job['_file']), job['result']['name']

    def _cleanup(self):
        now = time.time()
        for job_id, job in list(self.jobs.items()):
            if job['finished_at'] and now - job['finished_at'] > OUTPUT_TTL:
                shutil.rmtree(self.folder / job_id, ignore_errors=True)
                self.jobs.pop(job_id, None)
        # Папки от прошлых запусков службы: про них уже никто не знает.
        if self.folder.is_dir():
            for path in self.folder.iterdir():
                if path.name not in self.jobs:
                    try:
                        if now - path.stat().st_mtime > OUTPUT_TTL:
                            shutil.rmtree(path, ignore_errors=True)
                    except OSError:
                        pass

    def _worker(self):
        while True:
            self.wakeup.wait(30)
            self.wakeup.clear()
            while True:
                with self.lock:
                    if not self.queue:
                        break
                    job = self.jobs[self.queue.pop(0)]
                    job.update(status='running', started_at=time.time(), message='Готовлю файл')
                try:
                    self._run(job)
                    with self.lock:
                        job.update(status='done', progress=1.0, message='Готово',
                                   finished_at=time.time())
                except Exception as exc:
                    with self.lock:
                        cancelled = job['_cancel']
                        job.update(status='cancelled' if cancelled else 'error',
                                   message='Отменено' if cancelled else 'Ошибка',
                                   error='' if cancelled else str(exc), finished_at=time.time())
                    shutil.rmtree(self.folder / job['id'], ignore_errors=True)
                finally:
                    with self.lock:
                        self.process = None

    def _set(self, job, **values):
        with self.lock:
            job.update(values)

    def _run(self, job):
        key, op = job['key'], job['op']
        ffmpeg = find_binary('ffmpeg', self.root)
        source = self.stage(key)
        self._set(job, message='Читаю сведения о ролике')
        info = probe(source, self.root)
        params = check_params(op, job['params'], info)
        self._set(job, params=params, source_info=info)
        if op == 'replace' and info['browser'] and info['container'].startswith('mov'):
            raise ValueError('Ролик уже в MP4 и играет в браузере — перекодировать нечего')
        work = self.folder / job['id']
        work.mkdir(parents=True, exist_ok=True)
        name = output_name(key, op, params)
        target = work / ('result' + Path(name).suffix)
        encoder = h264_encoder(ffmpeg) if op not in {'frame', 'audio'} else ''
        command = build_command(ffmpeg, op, params, source, target, info, encoder)
        self._set(job, message='Перекодирую' if op != 'frame' else 'Снимаю кадр',
                  encoder=encoder)
        self._ffmpeg(job, command, expected_seconds(op, params, info))
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError('ffmpeg не создал файл')
        if op not in {'frame', 'audio'}:
            verify(info, probe(target, self.root), op, params)
        if op == 'replace':
            self._set(job, message='Записываю в источник')
            job['result'] = self._replace(key, target)
            shutil.rmtree(work, ignore_errors=True)
            return
        self._set(job, _file=str(target), result={'name': name, 'size': target.stat().st_size})

    def _ffmpeg(self, job, command, total):
        with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, creationflags=NO_WINDOW,
                              text=True, encoding='utf-8', errors='replace') as process:
            with self.lock:
                self.process = process
                if job['_cancel']:
                    process.kill()
            errors = []
            reader = threading.Thread(target=lambda: errors.extend(process.stderr), daemon=True)
            reader.start()
            for line in process.stdout:
                share = parse_progress(line, total)
                if share is not None:
                    self._set(job, progress=round(share * 0.97, 4))
            code = process.wait()
            reader.join(5)
        if job['_cancel']:
            raise RuntimeError('Отменено')
        if code:
            detail = next((line.strip() for line in reversed(errors) if line.strip()), '')
            raise RuntimeError(f'ffmpeg завершился с кодом {code}' + (f': {detail}' if detail else ''))

    def _replace(self, key, target):
        """Новый файл — рядом с оригиналом, оригинал — в корзину источника."""
        source_id, native = pathkeys.split(key)
        driver, native = self.access.resolve(key)
        record = self.access.record(source_id)
        final, temporary = replacement_native(driver, native)
        driver.put(temporary, target)
        try:
            self.trash(record, driver, native)
        except Exception:
            try:
                driver.remove(temporary)
            except Exception:
                pass
            raise
        if temporary != final:
            try:
                driver.rename(temporary, final)
            except Exception:
                # Оригинал уже в корзине: каталог пойдёт за файлом под временным именем.
                final = temporary
        info = driver.stat(final)
        return {'old': key, 'new': pathkeys.make(source_id, final), 'size': info.size,
                'modified': info.mtime_ns}


def trash_original(device_id):
    """Функция для MediaJobs.trash: у своего диска — корзина Windows, у сетевых
    источников — .homecloud-trash, как при удалении из галереи."""
    def trash(record, driver, native):
        if record and record.get('type') == 'device' and record.get('device') == device_id:
            from send2trash import send2trash
            send2trash(native)
            return
        target = sources.trash_native(record, native, datetime.now().strftime('%Y%m%d'))
        driver.rename(native, target)
    return trash
