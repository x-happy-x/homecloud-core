"""Хаб HomeCloud: каталог, превью и реестры на VM, без моделей и видеокарты.

Хаб — тот же web_server.py, запущенный с `--role hub`. Он держит у себя всё,
что посчитано: каталог SQLite (лица, описания, группы, альбомы), миниатюры
лиц, мелкие превью сетки и сведения о файлах. Поэтому уже обработанные снимки
и ролики показываются, даже когда ни одно ядро не включено. Оригиналы он
берёт прямо из источников (SMB, SFTP, FTP, WebDAV, диск устройства по SSH).

Считают ядра (core) — компьютеры с видеокартой. Хаб раздаёт им задания, а
они ходят к нему на отдельный порт связи (по умолчанию 18401):

  POST /db          — удалённое соединение с каталогом (catalogdb.Sessions)
  GET|PUT|DELETE /files/<путь>  — миниатюры лиц и модели роутера
  POST /thumbs      — превью сетки и сведения о файлах
  GET  /config      — источники с паролями и настройки ядра
  POST /hello       — ядро сообщило о себе: версия, возможности
  GET  /package     — текущий пакет ядра (для обновления)
  PUT  /import/<имя> — старый каталог устройства для переноса в общий
"""
import base64
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import sys
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen

import catalogdb
import pathkeys
import sources

THUMB_SHORT = 400
THUMB_LONG = 800
GRID_LIMIT = 640
CORE_PORT = 18311
PREVIEW_CACHE_LIMIT = 3 * 1024 ** 3

THUMBS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS photo_thumbs (
  path TEXT PRIMARY KEY,
  size INTEGER NOT NULL, modified INTEGER NOT NULL,
  width INTEGER, height INTEGER,
  thumb_width INTEGER, thumb_height INTEGER,
  bytes INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT,
  created_at REAL NOT NULL
);
'''

# Таблицы с путём снимка: по ним считается и чистится «что лежит по источнику».
PATH_TABLES = {
    'analysis': ('photo_analysis', 'photo_embeddings', 'photo_adult_analysis', 'photo_hashes',
                 'photo_curation', 'router_predictions', 'router_reviews', 'router_training_labels',
                 'router_skips', 'router_batch_items', 'video_speech', 'video_speech_segments',
                 'video_diarization', 'video_speakers', 'video_speaker_turns',
                 'video_speaker_faces'),
    'albums': ('album_photos', 'highlight_photos'),
}


def ensure_thumbs(db):
    db.executescript(THUMBS_SCHEMA)
    db.commit()


def thumb_name(key):
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()
    return f'{digest[:2]}/{digest}.jpg'


# ---------- реестр ядер ----------

def validate_core(value, current=None):
    current = current or {}
    core_id = str(value.get('id') or current.get('id') or '').strip().lower()
    if not pathkeys.SOURCE_ID.match(core_id):
        raise ValueError('Id устройства: латиница, цифры, «-» и «_», от 2 до 32 символов')
    name = str(value.get('name') or '').strip() or current.get('name') or core_id
    host = str(value.get('host') or current.get('host') or '').strip()
    if not host or re.search(r'\s', host) or len(host) > 253:
        raise ValueError('Укажите адрес устройства в домашней сети')
    port = int(value.get('port') or current.get('port') or CORE_PORT)
    if not 0 < port < 65536:
        raise ValueError('Некорректный порт ядра')
    ssh = dict(current.get('ssh') or {})
    if value.get('sshClear'):
        ssh = {}
    else:
        if value.get('sshUser') is not None and str(value.get('sshUser')).strip():
            ssh['user'] = str(value['sshUser']).strip()
        if value.get('sshPort'):
            ssh['port'] = int(value['sshPort'])
        if value.get('sshPassword'):
            ssh['password'] = str(value['sshPassword'])
        if value.get('sshHost') is not None:
            ssh['host'] = str(value.get('sshHost') or '').strip()
    if ssh and not re.match(r'^[\w.@\\-]{1,64}$', ssh.get('user') or ''):
        raise ValueError('Некорректное имя пользователя SSH')
    install_dir = str(value.get('installDir') or current.get('install_dir') or '').strip()
    if re.search(r'[\r\n"]', install_dir):
        raise ValueError('Некорректная папка установки')
    return {
        'id': core_id, 'name': name, 'host': host, 'port': port,
        'token': current.get('token') or str(value.get('token') or '') or secrets.token_urlsafe(32),
        'link_token': current.get('link_token') or secrets.token_urlsafe(32),
        'ssh': ssh or None,
        'install_dir': install_dir or 'C:\\HomeCloud\\core',
        'primary': bool(value.get('primary', current.get('primary', False))),
        'version': current.get('version', ''),
        'hello': current.get('hello', {}),
        'created_at': current.get('created_at') or time.time(),
    }


class CoreRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def load(self):
        with self.lock:
            try:
                rows = json.loads(self.path.read_text(encoding='utf-8'))
            except FileNotFoundError:
                return []
            return [row for row in rows if isinstance(row, dict) and row.get('id')]

    def save_all(self, rows):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, self.path)

    def get(self, core_id):
        return next((row for row in self.load() if row['id'] == core_id), None)

    def by_link_token(self, token):
        if not token:
            return None
        return next((row for row in self.load()
                     if secrets.compare_digest(row.get('link_token', ''), token)), None)

    def save(self, value):
        with self.lock:
            rows = self.load()
            wanted = str(value.get('id') or '').strip().lower()
            index = next((i for i, row in enumerate(rows) if row['id'] == wanted), -1)
            record = validate_core(value, rows[index] if index >= 0 else None)
            if index >= 0:
                rows[index] = record
            else:
                rows.append(record)
            if record['primary']:
                for row in rows:
                    row['primary'] = row['id'] == record['id']
            self.save_all(rows)
            return record

    def update(self, core_id, **values):
        with self.lock:
            rows = self.load()
            for row in rows:
                if row['id'] == core_id:
                    row.update(values)
                    self.save_all(rows)
                    return row
        raise KeyError('Устройство не найдено')

    def remove(self, core_id):
        with self.lock:
            rows = self.load()
            left = [row for row in rows if row['id'] != core_id]
            if len(left) == len(rows):
                raise KeyError('Устройство не найдено')
            self.save_all(left)

    @staticmethod
    def public(row):
        ssh = row.get('ssh') or None
        return {
            'id': row['id'], 'name': row['name'], 'host': row['host'], 'port': row['port'],
            'url': f"http://{row['host']}:{row['port']}", 'primary': bool(row.get('primary')),
            'installDir': row.get('install_dir', ''), 'version': row.get('version', ''),
            # Старый каталог устройства уже влит в общий — повторно не предлагать.
            'legacyMerged': bool(row.get('legacy_merged')),
            'ssh': ({'user': ssh.get('user', ''), 'port': ssh.get('port', 22),
                     'host': ssh.get('host', ''), 'hasPassword': bool(ssh.get('password'))}
                    if ssh else None),
        }


# ---------- медиа из источников ----------

class SourceMedia:
    """Оригинал снимка в источнике: то, что раньше было Path на диске бэкенда."""

    def __init__(self, hub, key):
        self.hub = hub
        self.key = key
        self.name = pathkeys.name(key)
        self._info = None

    def driver(self):
        source_id = pathkeys.source_of(self.key)
        if not self.hub.health.available(source_id):
            raise sources.SourceError(f'Источник «{source_id}» сейчас недоступен')
        return self.hub.access.resolve(self.key)

    def _watch(self, call):
        """Ошибка связи (не «нет файла») — источник недоступен до следующей проверки."""
        try:
            return call()
        except FileNotFoundError:
            raise
        except (sources.SourceError, OSError) as exc:
            if not isinstance(exc, (PermissionError, IsADirectoryError, NotADirectoryError)):
                self.hub.health.failed(pathkeys.source_of(self.key), exc)
            raise

    def info(self):
        if self._info is None:
            driver, native = self.driver()
            self._info = self._watch(lambda: driver.stat(native))
        return self._info

    def exists(self):
        try:
            return not self.info().is_dir
        except (FileNotFoundError, OSError, sources.SourceError):
            return False

    def is_file(self):
        return self.exists()

    def size(self):
        return self.info().size

    def mtime_ns(self):
        return self.info().mtime_ns

    def open(self):
        driver, native = self.driver()
        return self._watch(lambda: driver.open(native))

    def read_bytes(self, limit=None):
        with self.open() as stream:
            return stream.read(limit if limit else -1)

    def local(self):
        """Путь на хабе для библиотек, которым нужен файл (cv2, EXIF)."""
        self.driver()
        return self._watch(lambda: self.hub.stage.path(self.key))

    def resolve(self):
        return self

    def __str__(self):
        return self.key

    def __fspath__(self):
        return self.local()

# ---------- доступность источников ----------

class SourceHealth:
    """Доступен ли источник — по проверке раз в пять минут, а не на каждый кадр.

    Выключенный компьютер или уснувшее хранилище иначе стоят каждому снимку
    таймаута подключения (SSH ждёт 3–6 с), и страница из сотни плиток
    собирается минутами. Проверка идёт в фоне (start), вручную — check_all
    («Проверить сейчас» в настройках). Настоящее чтение, упавшее на связи,
    сразу помечает источник недоступным (failed). Пока источник не проверяли
    ни разу, он считается доступным.
    """
    INTERVAL = 300
    TIMEOUT = 8

    def __init__(self, hub):
        self.hub = hub
        self.lock = threading.Lock()
        self.state = {}
        self.stopped = threading.Event()

    def start(self):
        def loop():
            while not self.stopped.is_set():
                try:
                    self.check_all()
                except Exception as exc:
                    print(f'Проверка источников: {exc}', file=sys.stderr, flush=True)
                self.stopped.wait(self.INTERVAL)
        threading.Thread(target=loop, name='source-health', daemon=True).start()

    def available(self, source_id):
        with self.lock:
            item = self.state.get(source_id)
        return item is None or item['online']

    def status(self, source_id):
        with self.lock:
            item = self.state.get(source_id)
        return dict(item) if item else None

    def failed(self, source_id, error):
        with self.lock:
            self.state[source_id] = {'online': False, 'error': str(error)[:300],
                                     'checked_at': time.time(), 'passive': True}

    def check(self, record):
        """Один источник: список корней с таймаутом. Драйвер иногда висит дольше
        своего таймаута (SMB, SFTP), поэтому ждём его в отдельном потоке."""
        outcome = {}

        def probe():
            try:
                self.hub.access.driver(record['id']).test()
                outcome['ok'] = True
            except Exception as exc:
                outcome['error'] = str(exc) or type(exc).__name__

        started = time.monotonic()
        worker = threading.Thread(target=probe, daemon=True)
        worker.start()
        worker.join(self.TIMEOUT)
        online = bool(outcome.get('ok'))
        error = '' if online else outcome.get('error') or f'не ответил за {self.TIMEOUT} с'
        value = {'online': online, 'error': error[:300], 'checked_at': time.time(),
                 'ms': round((time.monotonic() - started) * 1000)}
        with self.lock:
            self.state[record['id']] = value
        return value

    def check_all(self):
        """Все источники разом, параллельно; заодно заново спрашиваются ядра."""
        with self.hub.status_lock:
            self.hub.status_cache.clear()
        records = [record for record in self.hub.sources.load() if record.get('enabled', True)]
        workers = [threading.Thread(target=self.check, args=(record,), daemon=True)
                   for record in records]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(self.TIMEOUT + 2)
        with self.lock:
            known = {record['id'] for record in self.hub.sources.load()}
            for source_id in list(self.state):
                if source_id not in known:
                    self.state.pop(source_id)
            return {key: dict(value) for key, value in self.state.items()}


# ---------- хаб ----------

class Hub:
    def __init__(self, catalog, data_dir, link_url='', ssh_key=None):
        self.catalog = Path(catalog).resolve()
        self.data = Path(data_dir).resolve()
        self.catalog.mkdir(parents=True, exist_ok=True)
        self.database = self.catalog / 'catalog.sqlite'
        self.grid = self.catalog / 'grid'
        self.link_url = link_url
        self.ssh_key = ssh_key
        self.sources = sources.Registry(self.data / 'sources.json')
        self.cores = CoreRegistry(self.data / 'cores.json')
        self.packages = self.data / 'core-packages'
        self.imports = self.data / 'imports'
        self.access = sources.Access(self.sources.load(), device_route=self.device_route,
                                     ssh_key=ssh_key)
        self.stage = sources.Stage(self.data / 'stage', self.access, 4 * 1024 ** 3)
        self.sessions = catalogdb.Sessions(self.database)
        self.status_cache = {}
        self.status_lock = threading.Lock()
        self.health = SourceHealth(self)
        self.installs = {}
        self.parallel = Parallel(self)
        db = sqlite3.connect(self.database, timeout=60)
        try:
            # WAL: чтение интерфейса и запись ядер не мешают друг другу.
            db.execute('PRAGMA journal_mode=WAL')
            ensure_thumbs(db)
        finally:
            db.close()
        self.import_backends()

    def import_backends(self):
        """Первый старт хаба: устройства прежнего реестра server.js становятся ядрами.

        Токены и SSH переезжают как были; папку ядра подсказывает прежняя
        команда запуска (…\\start-remote.ps1). Каждому ядру заводится источник
        «диск этого устройства».
        """
        legacy = self.data / 'backends.json'
        if self.cores.path.exists() or not legacy.is_file():
            return
        try:
            rows = json.loads(legacy.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return
        cores = []
        for row in rows:
            if not isinstance(row, dict) or not pathkeys.SOURCE_ID.match(str(row.get('id', ''))):
                continue
            address = urlparse(str(row.get('url') or ''))
            ssh = row.get('ssh') or None
            found = re.search(r'-File\s+"?(.+?)[\\/]start-remote\.ps1', (ssh or {}).get('command', ''))
            cores.append({
                'id': row['id'], 'name': row.get('name') or row['id'],
                'host': address.hostname or '', 'port': address.port or CORE_PORT,
                'token': row.get('token') or secrets.token_urlsafe(32),
                'link_token': secrets.token_urlsafe(32),
                'ssh': ({'user': ssh.get('user', ''), 'port': ssh.get('port', 22),
                         'host': ssh.get('host', '')} if ssh else None),
                'install_dir': found.group(1).replace('/', '\\') if found else 'C:\\HomeCloud\\core',
                'primary': bool(row.get('primary')), 'version': '', 'hello': {},
                'created_at': time.time()})
        if not cores:
            return
        self.cores.save_all(cores)
        records = self.sources.load()
        known = {record['id'] for record in records}
        for core in cores:
            if core['id'] not in known:
                records.append(sources.validate({'id': core['id'], 'name': core['name'],
                                                 'type': 'device', 'device': core['id']}))
        self.sources.save_all(records)
        self.refresh_sources()
        print(f'Ядра из backends.json: {", ".join(core["id"] for core in cores)}', flush=True)

    # ----- источники -----

    def refresh_sources(self):
        self.access.update(self.sources.load())

    def device_route(self, core_id):
        core = self.cores.get(core_id)
        if core is None:
            return None
        route = {}
        ssh = core.get('ssh')
        if ssh and ssh.get('user'):
            route['ssh'] = {'host': ssh.get('host') or core['host'], 'port': ssh.get('port', 22),
                            'user': ssh['user'], 'password': ssh.get('password', '')}
        route['agent'] = (f"http://{core['host']}:{core['port']}", core['token'])
        status = self.core_status(core_id, max_age=30)
        if status.get('online'):
            # Ядро в сети — читать через него быстрее, чем поднимать SSH.
            route.pop('ssh', None)
        return route

    def media(self, key):
        return SourceMedia(self, key)

    def source_names(self):
        return {row['id']: row['name'] for row in self.sources.load()}

    # ----- ядра -----

    def core_call(self, core, path, method='GET', body=None, timeout=10, raw=False):
        if isinstance(core, str):
            core = self.cores.get(core)
            if core is None:
                raise KeyError('Устройство не найдено')
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
        request = Request(f"http://{core['host']}:{core['port']}{path}", data=data, method=method,
                          headers={'X-Local-Token': core['token'], 'Accept': 'application/json',
                                   **({'Content-Type': 'application/json'} if data else {})})
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except Exception as exc:
            detail = ''
            if hasattr(exc, 'read'):
                try:
                    detail = json.loads(exc.read() or b'{}').get('error', '')
                except Exception:
                    detail = ''
            raise RuntimeError(detail or f"{core['name']} недоступно: {exc}") from exc
        return payload if raw else json.loads(payload or b'{}')

    def components(self, core):
        """Окружения и модели ядра; у каждой модели — на каких ещё ядрах в сети она есть."""
        result = self.core_call(core, '/api/core/components', timeout=60)
        peers = {}
        for other in self.cores.load():
            if other['id'] == core['id'] or not self.core_status(other['id'], max_age=30).get('online'):
                continue
            try:
                data = self.core_call(other, '/api/core/components', timeout=60)
            except RuntimeError:
                continue
            for item in data.get('models', []):
                if item.get('installed'):
                    peers.setdefault(item['id'], []).append({'id': other['id'],
                                                             'name': other['name']})
        for item in result.get('models', []):
            item['peers'] = peers.get(item['id'], [])
        return result

    def start_component(self, core, body):
        """Установка окружения, загрузка модели или её копия с другого ядра (from)."""
        action = str(body.get('action') or '')
        payload = {'id': str(body.get('id') or ''), 'action': action}
        if action == 'copy':
            source = self.cores.get(str(body.get('from') or ''))
            if source is None or source['id'] == core['id']:
                raise ValueError('Не выбрано ядро, с которого копировать')
            # Пропуск только на чтение моделей: свой токен ядро-источник не отдаёт.
            ticket = self.core_call(source, '/api/core/components/ticket', 'POST', {})['ticket']
            payload['peer'] = {'url': f"http://{source['host']}:{source['port']}",
                               'ticket': ticket}
        return self.core_call(core, '/api/core/components/start', 'POST', payload, timeout=30)

    # Ядро не ответило — не спрашиваем его снова пять минут: каждый запрос
    # стоил бы 4–8 с таймаута. Включённое ядро само приходит с /hello, и тогда
    # отметка снимается сразу; вручную — «Проверить сейчас» (SourceHealth.check_all).
    OFFLINE_TTL = 300

    def core_status(self, core_id, max_age=2.0):
        with self.status_lock:
            cached = self.status_cache.get(core_id)
            if cached:
                age = time.monotonic() - cached[0]
                if age < max_age or (not cached[1].get('online') and age < self.OFFLINE_TTL):
                    return cached[1]
        core = self.cores.get(core_id)
        if core is None:
            return {'online': False, 'error': 'Устройство не найдено'}
        try:
            device = self.core_call(core, '/api/device', timeout=4)
            job = self.core_call(core, '/api/device/job', timeout=4)
            value = {'online': True, 'device': device, 'job': job,
                     'legacy': device.get('role') != 'core'}
        except Exception as exc:
            value = {'online': False, 'error': str(exc)}
        with self.status_lock:
            self.status_cache[core_id] = (time.monotonic(), value)
        return value

    def cores_payload(self):
        rows = self.cores.load()
        current = self.package_info()
        result = []
        for row in rows:
            status = self.core_status(row['id'])
            version = (status.get('device') or {}).get('version') or row.get('version', '')
            result.append({**CoreRegistry.public(row), **status, 'version': version,
                           'outdated': bool(current and version and version != current['version']),
                           'install': self.install_state(row['id'])})
        return {'cores': result, 'package': current, 'publicKey': self.public_key()}

    def online_cores(self, capability=None):
        found = []
        for row in self.cores.load():
            status = self.core_status(row['id'])
            if not status.get('online') or status.get('legacy'):
                continue
            if capability and not (status.get('device') or {}).get(
                    'capabilities', {}).get(capability):
                continue
            found.append((row, status))
        found.sort(key=lambda item: (not item[0].get('primary'), item[0]['id']))
        return found

    def pick_core(self, capability=None, source_id=None, idle=False):
        """Ядро для работы: у своего диска — его хозяин, иначе основное в сети."""
        candidates = self.online_cores(capability)
        if idle:
            candidates = [item for item in candidates
                          if not (item[1].get('job') or {}).get('active')]
        if source_id:
            record = self.sources.get(source_id)
            if record and record['type'] == 'device':
                own = [item for item in candidates if item[0]['id'] == record.get('device')]
                if own:
                    return own[0][0]
        if not candidates:
            raise RuntimeError('Нет включённого ядра: запустите PC-X или PC-A')
        return candidates[0][0]

    def core_config(self, core):
        """То, что ядро получает от хаба: источники с паролями и адрес связи."""
        return {'core': core['id'], 'name': core['name'], 'sources': self.sources.load(),
                'hub': self.link_url}

    # ----- пакет ядра -----

    def package_info(self):
        try:
            return json.loads((self.packages / 'current.json').read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return None

    def package_file(self):
        info = self.package_info()
        if not info:
            return None
        path = self.packages / info['file']
        return path if path.is_file() else None

    def ensure_ssh_key(self):
        """Ключ хаба для SSH к устройствам; если его ещё нет — создаёт ed25519."""
        if not self.ssh_key:
            raise ValueError('Хабу не задан путь ключа SSH (--ssh-key)')
        path = Path(self.ssh_key)
        if path.is_file():
            return path
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        key = Ed25519PrivateKey.generate()
        private = key.private_bytes(serialization.Encoding.PEM,
                                    serialization.PrivateFormat.OpenSSH,
                                    serialization.NoEncryption())
        public = key.public_key().public_bytes(serialization.Encoding.OpenSSH,
                                               serialization.PublicFormat.OpenSSH)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(private)
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
        Path(str(path) + '.pub').write_text(public.decode('ascii') + ' homecloud-hub\n',
                                            encoding='utf-8')
        return path

    def public_key(self):
        """Открытый ключ хаба — его кладут в authorized_keys устройства."""
        if not self.ssh_key:
            return ''
        try:
            return Path(str(self.ssh_key) + '.pub').read_text(encoding='utf-8').strip()
        except OSError:
            pass
        try:
            import paramiko
        except ImportError:
            return ''
        for kind in (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey):
            try:
                key = kind.from_private_key_file(str(self.ssh_key))
                return f'{key.get_name()} {key.get_base64()} homecloud-hub'
            except (paramiko.SSHException, OSError, ValueError):
                continue
        return ''

    def install_state(self, core_id):
        job = self.installs.get(core_id)
        if not job:
            return None
        return {key: job[key] for key in ('status', 'action', 'started_at', 'finished_at',
                                          'error', 'log', 'step')}

    # ----- превью сетки -----

    def store_thumbs(self, db, items):
        now = time.time()
        rows = []
        for item in items:
            key = str(item['path'])
            if not pathkeys.is_key(key):
                raise ValueError('Превью принимаются только по ключу источника')
            data = base64.b64decode(item['data'])
            target = self.grid / thumb_name(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(data)
            os.replace(temporary, target)
            metadata = item.get('metadata')
            rows.append((key, int(item.get('size') or 0), int(item.get('modified') or 0),
                         item.get('width'), item.get('height'), item.get('thumb_width'),
                         item.get('thumb_height'), len(data),
                         json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
                         now))
        with db:
            db.executemany(
                'INSERT INTO photo_thumbs(path,size,modified,width,height,thumb_width,'
                'thumb_height,bytes,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?) '
                'ON CONFLICT(path) DO UPDATE SET size=excluded.size,modified=excluded.modified,'
                'width=excluded.width,height=excluded.height,thumb_width=excluded.thumb_width,'
                'thumb_height=excluded.thumb_height,bytes=excluded.bytes,'
                'metadata_json=COALESCE(excluded.metadata_json,photo_thumbs.metadata_json),'
                'created_at=excluded.created_at', rows)
        return len(rows)

    def thumb_file(self, key):
        path = self.grid / thumb_name(key)
        return path if path.is_file() else None

    def trim_previews(self):
        folder = self.catalog / 'previews'
        files, total = [], 0
        for path in folder.rglob('*.jpg'):
            try:
                info = path.stat()
            except OSError:
                continue
            files.append((info.st_atime, info.st_size, path))
            total += info.st_size
        if total <= PREVIEW_CACHE_LIMIT:
            return
        files.sort()
        for _stamp, size, path in files:
            if total <= PREVIEW_CACHE_LIMIT * 0.8:
                break
            path.unlink(missing_ok=True)
            total -= size

    def forget_thumbs(self, db, keys):
        keys = list(keys)
        for key in keys:
            (self.grid / thumb_name(key)).unlink(missing_ok=True)
        with db:
            for offset in range(0, len(keys), 500):
                batch = keys[offset:offset + 500]
                db.execute(f"DELETE FROM photo_thumbs WHERE path IN ({','.join('?' * len(batch))})",
                           batch)

    def rename_thumbs(self, pairs):
        for old, new in pairs:
            source = self.grid / thumb_name(old)
            if source.is_file():
                target = self.grid / thumb_name(new)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)

    # ----- файлы в источниках -----

    TRASH = sources.TRASH

    def _by_source(self, keys):
        groups = {}
        for key in keys:
            groups.setdefault(pathkeys.source_of(key), []).append(key)
        return groups

    def _device_core(self, source_id):
        record = self.sources.get(source_id)
        if record and record['type'] == 'device':
            core = self.cores.get(record.get('device'))
            if core is None:
                raise RuntimeError(f'Устройство «{record.get("device")}» не подключено')
            if not self.core_status(core['id']).get('online'):
                raise RuntimeError(f'{core["name"]} не в сети: файлы на его диске сейчас недоступны')
            return core
        return None

    def trash_key(self, key):
        """Куда в источнике уходит удалённый файл: папка .homecloud-trash в корне."""
        source, native = pathkeys.split(key)
        return pathkeys.make(source, sources.trash_native(
            self.sources.get(source), native, datetime.now().strftime('%Y%m%d')))

    def remove_originals(self, targets):
        """Удалённое уходит в корзину: у диска устройства — в корзину Windows,
        у сетевых источников — в .homecloud-trash (обход её не видит)."""
        deleted, errors = [], []
        stored = dict(targets)
        for source_id, keys in self._by_source([raw for raw, _ in targets]).items():
            try:
                core = self._device_core(source_id)
                if core is not None:
                    natives = {pathkeys.native(stored[key]): key for key in keys}
                    result = self.core_call(core, '/api/core/fileop', 'POST',
                                            {'op': 'delete', 'paths': list(natives)}, timeout=600)
                    deleted.extend(natives[item] for item in result.get('done', []))
                    errors.extend({'path': natives.get(item['path'], item['path']),
                                   'error': item['error']} for item in result.get('errors', []))
                    continue
                for key in keys:
                    try:
                        driver, native = self.access.resolve(stored[key])
                        target = pathkeys.native(self.trash_key(stored[key]))
                        driver.rename(native, target)
                        deleted.append(key)
                    except (OSError, sources.SourceError) as exc:
                        errors.append({'path': key, 'error': str(exc)})
            except Exception as exc:
                errors.extend({'path': key, 'error': str(exc)} for key in keys)
        return deleted, errors

    def remove_folder(self, folder):
        source_id, native = pathkeys.split(folder)
        core = self._device_core(source_id)
        if core is not None:
            self.core_call(core, '/api/core/fileop', 'POST', {'op': 'rmdir', 'path': native})
            return
        raise ValueError('Пустую папку в сетевом источнике удалите на самом устройстве')

    def move_originals(self, moves):
        done, errors = [], []
        for source_id, keys in self._by_source([old for old, _new in moves]).items():
            wanted = [(old, new) for old, new in moves if old in set(keys)]
            try:
                core = self._device_core(source_id)
                if core is not None:
                    pairs = {pathkeys.native(old): (old, new) for old, new in wanted}
                    result = self.core_call(core, '/api/core/fileop', 'POST', {
                        'op': 'move', 'moves': [[pathkeys.native(old), pathkeys.native(new)]
                                                for old, new in wanted]}, timeout=600)
                    done.extend(pairs[old] for old, _new in result.get('done', []))
                    errors.extend({'path': pairs.get(item['path'], (item['path'],))[0],
                                   'error': item['error']} for item in result.get('errors', []))
                    continue
                for old, new in wanted:
                    try:
                        driver, native = self.access.resolve(old)
                        if driver.exists(pathkeys.native(new)):
                            raise sources.SourceError('Файл с таким именем уже есть')
                        driver.rename(native, pathkeys.native(new))
                        done.append((old, new))
                    except (OSError, sources.SourceError) as exc:
                        errors.append({'path': old, 'error': str(exc)})
            except Exception as exc:
                errors.extend({'path': old, 'error': str(exc)} for old, _new in wanted)
        return done, errors

    # ----- данные по источникам -----

    def storage(self, db):
        """Что лежит на хабе по каждому источнику: снимки, лица, превью, анализ."""
        names = self.source_names()
        stats = {}

        def slot(source):
            return stats.setdefault(source, {
                'id': source, 'name': names.get(source, source or 'без источника'),
                'known': source in names, 'photos': 0, 'videos': 0, 'missing': 0,
                'faces': 0, 'named_faces': 0, 'thumbs': 0, 'thumb_bytes': 0, 'analysis': 0,
                'captions': 0, 'face_bytes': 0})

        def source_sql(column):
            return (f"CASE WHEN instr({column},':')>2 THEN substr({column},1,instr({column},':')-1) "
                    "ELSE '' END")

        for source, status, kind, count in db.execute(
                f"SELECT {source_sql('path')},status,kind,COUNT(*) FROM photos GROUP BY 1,2,3"):
            item = slot(source)
            if status == 'ok':
                item['videos' if kind == 'video' else 'photos'] += count
            elif status == 'missing':
                item['missing'] += count
        for source, count, named in db.execute(
                f"SELECT {source_sql('faces.path')},COUNT(*),COUNT(face_people.face_id) FROM faces "
                'LEFT JOIN face_people ON face_people.face_id=faces.id GROUP BY 1'):
            slot(source).update(faces=count, named_faces=named)
        for source, count, size in db.execute(
                f"SELECT {source_sql('path')},COUNT(*),SUM(bytes) FROM photo_thumbs GROUP BY 1"):
            slot(source).update(thumbs=count, thumb_bytes=size or 0)
        for source, count, captions in db.execute(
                f"SELECT {source_sql('path')},COUNT(*),SUM(caption_status='ok') "
                'FROM photo_analysis GROUP BY 1'):
            slot(source).update(analysis=count, captions=captions or 0)
        for item in stats.values():
            # Миниатюра лица — около 5 КБ; точный размер не стоит обхода папки.
            item['face_bytes'] = item['faces'] * 5 * 1024
        total_disk = shutil.disk_usage(self.catalog)
        return {'sources': sorted(stats.values(), key=lambda item: (not item['known'], item['name'])),
                'catalog_bytes': self.database.stat().st_size if self.database.exists() else 0,
                'disk_free': total_disk.free, 'disk_total': total_disk.total}

    def clean(self, db, source, what):
        """Удаляет с хаба обработанное по источнику. Сами файлы в источнике не трогаются."""
        if source != '' and not pathkeys.SOURCE_ID.match(source):
            raise ValueError('Неизвестный источник')
        what = set(what or ())
        unknown = what - {'thumbs', 'analysis', 'faces', 'catalog', 'previews'}
        if unknown or not what:
            raise ValueError('Выберите, что удалять: thumbs, analysis, faces, catalog')
        if source:
            low, high = source + ':', source + ':\uffff'
            where, values = '(path>=? AND path<?)', [low, high]
        else:
            where, values = "(instr(path,':')<=2)", []
        removed = {}
        if 'catalog' in what:
            what |= {'thumbs', 'analysis', 'faces'}
        if 'thumbs' in what:
            keys = [row[0] for row in db.execute(f'SELECT path FROM photo_thumbs WHERE {where}',
                                                 values)]
            for key in keys:
                (self.grid / thumb_name(key)).unlink(missing_ok=True)
            with db:
                db.execute(f'DELETE FROM photo_thumbs WHERE {where}', values)
            removed['thumbs'] = len(keys)
        if 'previews' in what or 'thumbs' in what:
            shutil.rmtree(self.catalog / 'previews', ignore_errors=True)
        if 'faces' in what:
            faces = db.execute(f'SELECT id,thumbnail FROM faces WHERE {where}', values).fetchall()
            ids = [row[0] for row in faces]
            with db:
                for offset in range(0, len(ids), 500):
                    batch = ids[offset:offset + 500]
                    marks = ','.join('?' * len(batch))
                    for table in ('face_people', 'face_exclusions', 'face_clusters',
                                  'face_authenticity', 'face_quality', 'face_track_data',
                                  'face_track_identities', 'face_track_samples', 'group_avatars',
                                  'video_speaker_faces'):
                        try:
                            db.execute(f'DELETE FROM {table} WHERE face_id IN ({marks})', batch)
                        except sqlite3.OperationalError:
                            pass
                    try:
                        db.execute(f'DELETE FROM face_identity_conflicts WHERE face_a IN ({marks}) '
                                   f'OR face_b IN ({marks})', batch * 2)
                    except sqlite3.OperationalError:
                        pass
                    db.execute(f'DELETE FROM faces WHERE id IN ({marks})', batch)
                try:
                    db.execute(f'DELETE FROM video_identities WHERE {where}', values)
                except sqlite3.OperationalError:
                    pass
                # Модель лиц переделает их при следующем скане: снимаем её отметку.
                db.execute(f"UPDATE photos SET model=NULL WHERE {where}", values)
            for _face_id, thumbnail in faces:
                if thumbnail:
                    (self.catalog / thumbnail).unlink(missing_ok=True)
            removed['faces'] = len(ids)
        if 'analysis' in what:
            count = 0
            with db:
                for table in PATH_TABLES['analysis']:
                    try:
                        count += db.execute(f'DELETE FROM {table} WHERE {where}', values).rowcount
                    except sqlite3.OperationalError:
                        pass
            removed['analysis'] = count
        if 'catalog' in what:
            with db:
                for table in (*PATH_TABLES['albums'], 'hidden_photos', 'video_people_hints',
                              'photo_dirs', 'scan_roots', 'scan_exclusions', 'photos'):
                    try:
                        db.execute(f'DELETE FROM {table} WHERE {where}', values)
                    except sqlite3.OperationalError:
                        pass
            removed['catalog'] = True
        return removed


# ---------- порт связи для ядер ----------

class LinkHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'HomeCloudHub/1.0'

    @property
    def hub(self):
        return self.server.hub

    def log_message(self, format, *args):
        if self.command == 'POST' and self.path.startswith('/db'):
            return
        print(f'link {self.address_string()} - {format % args}', flush=True)

    def core(self):
        return self.hub.cores.by_link_token(self.headers.get('X-Core-Token', ''))

    def reply(self, status, body=b'', content_type='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def body(self, limit=512 * 1024 * 1024):
        length = int(self.headers.get('Content-Length') or 0)
        if length > limit:
            raise ValueError('Слишком большой запрос')
        return self.rfile.read(length) if length else b''

    def file_target(self, raw):
        relative = unquote(raw).replace('\\', '/').lstrip('/')
        if '..' in relative.split('/') or not relative.startswith(('thumbnails/', 'router-models/')):
            raise ValueError('Недопустимый файл')
        return self.hub.catalog / relative

    def dispatch(self, method):
        core = self.core()
        if core is None:
            self.body(limit=64 * 1024 * 1024)
            return self.reply(403, {'error': 'Неизвестное ядро: нет или неверный X-Core-Token'})
        path = urlparse(self.path).path
        try:
            if method == 'POST' and path == '/db':
                payload = json.loads(self.body() or b'{}')
                return self.reply(200, self.hub.sessions.handle(payload, core['id']))
            if path.startswith('/files/'):
                target = self.file_target(path[len('/files/'):])
                if method == 'GET':
                    if not target.is_file():
                        return self.reply(404, {'error': 'Нет файла'})
                    return self.reply(200, target.read_bytes(), 'application/octet-stream')
                if method == 'PUT':
                    data = self.body()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + '.tmp')
                    temporary.write_bytes(data)
                    os.replace(temporary, target)
                    return self.reply(200, {'ok': True})
                if method == 'DELETE':
                    target.unlink(missing_ok=True)
                    return self.reply(200, {'ok': True})
            if method == 'POST' and path == '/thumbs':
                payload = json.loads(self.body() or b'{}')
                db = sqlite3.connect(self.hub.database, timeout=60)
                try:
                    stored = self.hub.store_thumbs(db, payload.get('items') or [])
                finally:
                    db.close()
                return self.reply(200, {'ok': True, 'stored': stored})
            if method == 'GET' and path == '/config':
                return self.reply(200, self.hub.core_config(core))
            if method == 'POST' and path == '/hello':
                payload = json.loads(self.body() or b'{}')
                self.hub.sessions.close_owner(core['id'])
                self.hub.cores.update(core['id'], version=str(payload.get('version') or ''),
                                      hello={**payload, 'at': time.time()})
                with self.hub.status_lock:
                    self.hub.status_cache.pop(core['id'], None)
                return self.reply(200, self.hub.core_config(core))
            if method == 'GET' and path == '/package':
                package = self.hub.package_file()
                if package is None:
                    return self.reply(404, {'error': 'Пакет ядра не опубликован'})
                return self.reply(200, package.read_bytes(), 'application/zip')
            if method == 'PUT' and path.startswith('/import/'):
                name = re.sub(r'[^\w.-]', '_', unquote(path[len('/import/'):]))[:80]
                self.hub.imports.mkdir(parents=True, exist_ok=True)
                target = self.hub.imports / f"{core['id']}-{name}"
                length = int(self.headers.get('Content-Length') or 0)
                with open(str(target) + '.part', 'wb') as output:
                    left = length
                    while left > 0:
                        chunk = self.rfile.read(min(left, 4 * 1024 * 1024))
                        if not chunk:
                            break
                        output.write(chunk)
                        left -= len(chunk)
                os.replace(str(target) + '.part', target)
                return self.reply(200, {'ok': True, 'file': target.name, 'bytes': length})
            return self.reply(404, {'error': 'Нет такого адреса'})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self.reply(400, {'error': str(exc)})
        except Exception as exc:
            print(f'link error {path}: {exc}', file=sys.stderr, flush=True)
            return self.reply(500, {'error': str(exc)})

    def do_GET(self):
        return self.dispatch('GET')

    def do_POST(self):
        return self.dispatch('POST')

    def do_PUT(self):
        return self.dispatch('PUT')

    def do_DELETE(self):
        return self.dispatch('DELETE')


def serve_link(hub, host, port):
    server = ThreadingHTTPServer((host, port), LinkHandler)
    server.daemon_threads = True
    server.hub = hub
    thread = threading.Thread(target=server.serve_forever, daemon=True, name='core-link')
    thread.start()
    return server


# ---------- вычисления на ядре вместо своих служб ----------

class Postponed(RuntimeError):
    """Ядра нет, и недавно уже проверяли: не шуметь в журнале на каждый опрос."""
    quiet = True


class RemoteTask:
    """Служба, которая раньше крутилась внутри бэкенда, а теперь — на ядре.

    Тот же вид, что у DuplicateService, ReclusterController и прочих:
    status/start/stop. Задание уходит на выбранное ядро, статус читается
    оттуда же; пока ядер нет, статус — «ядро не в сети».
    """

    def __init__(self, hub, kind, capability=None, on_finish=None, local_status=None):
        self.hub = hub
        self.kind = kind
        self.capability = capability
        self.on_finish = on_finish
        self.local_status = local_status
        self.core_id = None
        self.last = {'status': 'idle'}
        self.lock = threading.Lock()
        self.blocked_until = 0.0

    def _merge(self, value):
        if self.local_status:
            try:
                value = {**self.local_status(), **value}
            except Exception:
                pass
        return value

    def status(self):
        with self.lock:
            core_id = self.core_id
        if core_id is None:
            return self._merge(dict(self.last))
        try:
            value = self.hub.core_call(core_id, f'/api/core/task/{self.kind}', timeout=5)
        except Exception as exc:
            return self._merge({**self.last, 'core_error': str(exc)})
        value['core'] = core_id
        finished = value.get('status') in {'completed', 'stopped', 'error', 'idle'}
        previous = self.last.get('status')
        with self.lock:
            self.last = value
            if finished:
                self.core_id = None
        if finished and previous not in {'completed', 'stopped', 'error', 'idle'} and self.on_finish:
            try:
                self.on_finish(value)
            except Exception as exc:
                print(f'{self.kind} finish hook failed: {exc}', file=sys.stderr, flush=True)
        return self._merge(value)

    def start(self, *args, **kwargs):
        if time.monotonic() < self.blocked_until:
            raise Postponed('Нет включённого ядра')
        try:
            core = self.hub.pick_core(self.capability)
        except RuntimeError:
            self.blocked_until = time.monotonic() + 60
            raise
        value = self.hub.core_call(core, f'/api/core/task/{self.kind}/start', 'POST',
                                   {'args': list(args), 'kwargs': kwargs}, timeout=30)
        with self.lock:
            self.core_id = core['id']
            self.last = value.get('job') or value
        return self._merge({**self.last, 'core': core['id']})

    def stop(self):
        with self.lock:
            core_id = self.core_id
        if core_id is None:
            raise ValueError('Сейчас ничего не выполняется')
        value = self.hub.core_call(core_id, f'/api/core/task/{self.kind}/stop', 'POST', {},
                                   timeout=15)
        return self._merge(value.get('job') or value)


class RemoteSemantic:
    """Смысловой поиск: текст в вектор переводит ядро, у хаба нет моделей."""

    def __init__(self, hub):
        self.hub = hub
        self.cache = {}

    def query(self, text, top=500, vector=''):
        """С vector (base64 float32) — похожие на готовый снимок, модель ядру не нужна."""
        key = ('v:' + hashlib.sha1(vector.encode()).hexdigest()) if vector else text.casefold().strip()
        if not key:
            return []
        if key in self.cache:
            return self.cache[key]
        try:
            core = self.hub.pick_core('visual')
            payload = {'vector': vector, 'top': top} if vector else {'text': text, 'top': top}
            result = self.hub.core_call(core, '/api/core/semantic', 'POST',
                                        payload, timeout=90).get('results', [])
        except Exception as exc:
            print(f'Semantic search unavailable: {exc}', file=sys.stderr, flush=True)
            return []
        self.cache[key] = result
        if len(self.cache) > 50:
            self.cache.pop(next(iter(self.cache)))
        return result


class Parallel:
    """Одно задание на несколько ядер сразу.

    Опись и превью — один раз, на ядре-хозяине источника (диск компьютера
    читает только он). Пофайловые этапы — долями: ядро i из n берёт снимки
    с rowid % n == i (pathkeys.shard_sql). Подборки строятся по всему
    каталогу — один раз, после всех долей. Ядро берут, только если оно
    умеет все выбранные этапы.
    """
    SINGLE = ('inventory', 'highlights')

    def __init__(self, hub):
        self.hub = hub
        self.lock = threading.Lock()
        self.state = None
        self.stop_requested = False

    def status(self):
        with self.lock:
            return dict(self.state) if self.state else {'status': 'idle'}

    def eligible(self, per_file):
        """Ядра в сети, свободные и умеющие все пофайловые этапы задания."""
        result, skipped = [], []
        for core, status in self.hub.online_cores():
            if status.get('legacy'):
                continue
            capabilities = (status.get('device') or {}).get('capabilities') or {}
            missing = [name for name in per_file if not capabilities.get(name)]
            if (status.get('job') or {}).get('active'):
                skipped.append(f"{core['name']}: занято")
            elif missing:
                skipped.append(f"{core['name']}: нет {', '.join(missing)}")
            else:
                result.append(core)
        return result, skipped

    def start(self, roots, features, video_features=None, force=False, visual_model=None,
              cores=None):
        roots = [pathkeys.trim(str(root)) for root in roots or []]
        if not roots or any(not pathkeys.is_key(root) for root in roots):
            raise ValueError('Выберите папки источников')
        chosen = {name for name, on in {**(features or {}), **(video_features or {})}.items()
                  if on}
        per_file = sorted(chosen - set(self.SINGLE))
        participants, skipped = self.eligible(per_file)
        if cores:
            participants = [core for core in participants if core['id'] in set(cores)]
        if not participants:
            raise ValueError('Нет свободного ядра, умеющего все выбранные этапы'
                             + (f" ({'; '.join(skipped)})" if skipped else ''))
        sources = {pathkeys.source_of(root) for root in roots}
        owner = self.hub.pick_core(source_id=next(iter(sources)) if len(sources) == 1 else None)
        with self.lock:
            if self.state and self.state['status'] == 'running':
                raise ValueError('Параллельное задание уже идёт')
            self.stop_requested = False
            self.state = {
                'status': 'running', 'step': 'inventory', 'roots': roots,
                'features': per_file, 'cores': [core['id'] for core in participants],
                'owner': owner['id'], 'skipped': skipped, 'error': '',
                'started_at': time.time(), 'finished_at': None, 'parts': []}
        job = {'roots': roots, 'force': bool(force), 'visual_model': visual_model}
        plan = {'features': features or {}, 'video_features': video_features,
                'highlights': 'highlights' in chosen, 'per_file': per_file}
        threading.Thread(target=self.run, args=(participants, owner, job, plan),
                         daemon=True).start()
        return self.status()

    def stop(self):
        with self.lock:
            if not self.state or self.state['status'] != 'running':
                raise ValueError('Параллельного задания нет')
            self.stop_requested = True
            targets = list(self.state['cores']) + [self.state['owner']]
        for core_id in dict.fromkeys(targets):
            try:
                self.hub.core_call(core_id, '/api/device/job/stop', 'POST', {}, timeout=15)
            except RuntimeError:
                continue
        return self.status()

    def only(self, source, names):
        return {name: bool(source.get(name)) for name in names} if source else None

    def run(self, participants, owner, job, plan):
        try:
            self.step('inventory', [(owner, {'features': {'inventory': True}})], job)
            per_file = plan['per_file']
            if per_file:
                count = len(participants)
                self.step('shards', [
                    (core, {'features': self.only(plan['features'], per_file),
                            'video_features': self.only(plan['video_features'], per_file),
                            'shard': {'index': index, 'count': count}})
                    for index, core in enumerate(participants)], job)
            if plan['highlights']:
                # Доля «0 из 1»: ядро не обходит источник, фильтра по долям нет.
                self.step('highlights', [(participants[0], {
                    'features': {'highlights': True},
                    'shard': {'index': 0, 'count': 1}})], job)
            with self.lock:
                self.state.update(status='completed', step='done', finished_at=time.time())
        except Exception as exc:
            with self.lock:
                self.state.update(status='stopped' if self.stop_requested else 'error',
                                  error='' if self.stop_requested else str(exc),
                                  finished_at=time.time())

    def step(self, name, launches, job):
        """Запускает задания step на ядрах и ждёт, пока все закончат."""
        if self.stop_requested:
            raise RuntimeError('Остановлено')
        parts = []
        for core, extra in launches:
            self.hub.core_call(core, '/api/device/job/start', 'POST', {**job, **extra}, timeout=60)
            parts.append({'core': core['id'], 'name': core['name'],
                          'shard': extra.get('shard'), 'status': 'running'})
        with self.lock:
            self.state.update(step=name, parts=parts)
        while True:
            time.sleep(3)
            running = False
            for part in parts:
                status = self.hub.core_status(part['core'], max_age=0)
                current = status.get('job') or {}
                part.update(phase=current.get('phase'), completed=current.get('completed'),
                            total=current.get('total'))
                if not status.get('online'):
                    part['status'] = 'offline'
                elif current.get('active'):
                    part['status'] = 'running'
                    running = True
                else:
                    part['status'] = current.get('status') or 'completed'
            with self.lock:
                self.state['parts'] = [dict(part) for part in parts]
            if not running:
                break
        failed = [part for part in parts if part['status'] not in {'completed', 'complete'}]
        if failed:
            raise RuntimeError('; '.join(f"{part['name']}: {part['status']}" for part in failed))


class RemoteJobs:
    """Задания ядер вместо DeviceController: сводка для интерфейса и запуск."""

    def __init__(self, hub, catalog):
        self.hub = hub
        self.catalog = Path(catalog)

    def status(self):
        jobs = []
        for row in self.hub.cores.load():
            status = self.hub.core_status(row['id'])
            job = status.get('job') or {}
            # Старый бэкенд считает в свой каталог — к общему его задания не относятся.
            if job.get('active') and not status.get('legacy'):
                jobs.append({**job, 'core': row['id']})
        if jobs:
            return {**jobs[0], 'active': True, 'jobs': jobs}
        return {'active': False, 'status': 'idle', 'jobs': []}

    def info(self):
        for row, status in self.hub.online_cores():
            return status.get('device') or {}
        return {'visual_models': [], 'capabilities': {}}

    def start(self, roots, features, paths=None, force=False, visual_model=None,
              video_features=None, core_id=None, resume=False):
        keys = [*(roots or []), *(paths or [])]
        source_ids = {pathkeys.source_of(key) for key in keys}
        if '' in source_ids:
            raise ValueError('Укажите папки источников, а не пути компьютера')
        source_id = next(iter(source_ids)) if len(source_ids) == 1 else None
        core = self.hub.cores.get(core_id) if core_id else self.hub.pick_core(source_id=source_id)
        if core is None:
            raise ValueError('Устройство не найдено')
        return self.hub.core_call(core, '/api/device/job/start', 'POST', {
            'roots': list(roots or []), 'paths': list(paths or []), 'features': features,
            'video_features': video_features, 'force': bool(force),
            'visual_model': visual_model, 'resume': bool(resume)}, timeout=60).get('job')

    def stop(self, core_id=None):
        targets = [core_id] if core_id else [row['id'] for row in self.hub.cores.load()]
        result = None
        for target in targets:
            try:
                result = self.hub.core_call(target, '/api/device/job/stop', 'POST', {}, timeout=15)
            except Exception:
                continue
        if result is None:
            raise ValueError('Активного задания нет')
        return result.get('job', result)

    # История заданий лежит в каталоге — её хаб читает сам.
    def history_db(self):
        db = sqlite3.connect(self.hub.database, timeout=10)
        db.execute('''CREATE TABLE IF NOT EXISTS scan_runs (
            id INTEGER PRIMARY KEY,
            roots_json TEXT NOT NULL, paths_json TEXT NOT NULL,
            first_run_at TEXT NOT NULL, last_run_at TEXT NOT NULL,
            runs INTEGER NOT NULL DEFAULT 1,
            last_features TEXT NOT NULL, done_features TEXT NOT NULL DEFAULT '[]',
            last_status TEXT NOT NULL DEFAULT 'running')''')
        return db

    def history(self, limit=30):
        db = self.history_db()
        try:
            rows = db.execute('''SELECT id,roots_json,paths_json,first_run_at,last_run_at,
                runs,last_features,done_features,last_status FROM scan_runs
                ORDER BY last_run_at DESC LIMIT ?''', (limit,)).fetchall()
            runs = []
            for row in rows:
                roots, paths = json.loads(row[1]), json.loads(row[2])
                photos = len(paths)
                if roots:
                    where, values = pathkeys.scope_sql(roots, (), 'path')
                    photos = db.execute("SELECT COUNT(*) FROM photos WHERE status='ok'" + where,
                                        values).fetchone()[0]
                runs.append({'id': row[0], 'roots': roots, 'paths': paths,
                             'first_run_at': row[3], 'last_run_at': row[4], 'runs': row[5],
                             'features': json.loads(row[6]), 'done': json.loads(row[7]),
                             'status': row[8], 'photos': photos})
            return {'runs': runs}
        finally:
            db.close()

    def forget(self, run_id):
        db = self.history_db()
        try:
            with db:
                db.execute('DELETE FROM scan_runs WHERE id=?', (int(run_id),))
        finally:
            db.close()
        return {'ok': True}


class Idle:
    """Старые службы бэкенда (разовый скан D:\\trash), которых у хаба нет."""

    def status(self):
        return {'status': 'idle', 'active': False, 'total': 0, 'completed': 0}

    def start(self, *args, **kwargs):
        raise ValueError('Эта операция выполняется на ядре через «Сканирование»')

    stop = start


# ---------- обработка роликов на ядрах ----------

class HubMedia:
    """Операции с роликами через ffmpeg ядра (video_tools.py на ядре).

    Хаб выбирает ядро с возможностью video, пересылает запросы и хранит,
    какое задание на каком ядре. Замену в источнике (op replace) доводит сам:
    следит за заданием и, когда ядро записало новый файл, переносит записи
    каталога через on_replaced(result).
    """
    TERMINAL = {'done', 'error', 'cancelled'}

    def __init__(self, hub, on_replaced):
        self.hub = hub
        self.on_replaced = on_replaced
        self.lock = threading.RLock()
        self.jobs = {}

    def pick(self, key, op):
        source_id = pathkeys.source_of(key)
        core = self.hub.pick_core('video', source_id)
        record = self.hub.sources.get(source_id) or {}
        if op == 'replace' and record.get('type') == 'device' and core['id'] != record.get('device'):
            raise RuntimeError(f'Ролик лежит на диске «{record.get("device")}»: заменить его может '
                               'только его ядро — установите на нём «Видео: ffmpeg»')
        return core

    def probe(self, key):
        try:
            core = self.pick(key, 'probe')
        except RuntimeError as exc:
            raise RuntimeError(f'{exc}. Для обработки видео нужно ядро с ffmpeg') from exc
        info = self.hub.core_call(core, '/api/core/media/probe?key=' + quote(key, safe=''),
                                  timeout=300)
        return {**info, 'core': core['name']}

    def start(self, key, op, params):
        core = self.pick(key, op)
        job = self.hub.core_call(core, '/api/core/media/start', 'POST',
                                 {'key': key, 'op': op, 'params': params or {}}, timeout=60)
        hub_id = f"{core['id']}.{job['id']}"
        with self.lock:
            self.jobs[hub_id] = {'core': core['id'], 'job': job['id'], 'key': key, 'op': op,
                                 'finalized': False, 'final': None}
        if op == 'replace':
            threading.Thread(target=self._watch, args=(hub_id,), daemon=True).start()
        return self._public(hub_id, job)

    def _entry(self, hub_id):
        with self.lock:
            entry = self.jobs.get(hub_id)
        if entry is None:
            core_id, _, job_id = str(hub_id).partition('.')
            if not core_id or not job_id:
                raise KeyError('Задание не найдено')
            # Хаб перезапускали: скачать готовое всё ещё можно.
            entry = {'core': core_id, 'job': job_id, 'key': '', 'op': '', 'finalized': True,
                     'final': None}
        return entry

    def status(self, hub_id):
        entry = self._entry(hub_id)
        job = self.hub.core_call(entry['core'], '/api/core/media/job?id=' + quote(entry['job']),
                                 timeout=20)
        if job.get('op') == 'replace' and job.get('status') == 'done':
            job = self._finalize(hub_id, job)
        return self._public(hub_id, job)

    def cancel(self, hub_id):
        entry = self._entry(hub_id)
        job = self.hub.core_call(entry['core'], '/api/core/media/cancel', 'POST',
                                 {'id': entry['job']}, timeout=20)
        return self._public(hub_id, job)

    def open_file(self, hub_id, range_header=''):
        """Ответ ядра с готовым файлом — хаб отдаёт его дальше потоком."""
        entry = self._entry(hub_id)
        core = self.hub.cores.get(entry['core'])
        if core is None:
            raise KeyError('Устройство не найдено')
        request = Request(f"http://{core['host']}:{core['port']}/api/core/media/file?id="
                          + quote(entry['job']),
                          headers={'X-Local-Token': core['token'],
                                   **({'Range': range_header} if range_header else {})})
        try:
            return urlopen(request, timeout=60)
        except Exception as exc:
            raise RuntimeError(f"{core['name']} не отдал файл: {exc}") from exc

    def _finalize(self, hub_id, job):
        """Ядро заменило файл в источнике — каталог переходит на новый путь. Один раз."""
        with self.lock:
            entry = self.jobs.get(hub_id)
            if entry is None:
                return job
            if not entry['finalized']:
                try:
                    self.on_replaced(job['result'])
                    entry['final'] = None
                except Exception as exc:
                    entry['final'] = f'Файл заменён, но каталог не обновился: {exc}'
                    print(f'media replace {hub_id}: {exc}', file=sys.stderr, flush=True)
                entry['finalized'] = True
            if entry['final']:
                job = {**job, 'status': 'error', 'error': entry['final']}
        return job

    def _watch(self, hub_id):
        """Замена доводится и без открытого окна: опрашиваем ядро до конца."""
        failures = 0
        while failures < 200:
            time.sleep(3)
            try:
                job = self.status(hub_id)
                failures = 0
            except Exception:
                failures += 1
                continue
            if job.get('status') in self.TERMINAL:
                return

    def _public(self, hub_id, job):
        core = self.hub.cores.get(self._entry(hub_id)['core']) or {}
        return {**job, 'id': hub_id, 'core': core.get('name', ''),
                'download': f'/media/job?id={quote(hub_id)}'
                if job.get('status') == 'done' and job.get('op') != 'replace' else ''}


# ---------- API для интерфейса ----------

class HubApi:
    """Маршруты хаба для интерфейса: ядра, источники, данные по источникам.

    server.js проксирует их как обычный /api; права (viewer/editor/admin)
    проверяет он, а здесь — только токен server.js.
    """
    PREFIXES = ('/api/hub', '/api/cores', '/api/sources', '/api/storage', '/api/imports',
                '/api/scan/history', '/api/media')

    def __init__(self, app, hub):
        self.app = app
        self.hub = hub
        self.merges = {}

    def owns(self, path):
        return path.startswith(self.PREFIXES)

    def dispatch(self, handler, method, parsed, body):
        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        try:
            result = self.route(handler, method, parsed.path, query, body or {})
        except KeyError as exc:
            return handler.error_json(404, str(exc).strip("'"))
        except (ValueError, sources.SourceError, sqlite3.Error) as exc:
            return handler.error_json(400, str(exc))
        except RuntimeError as exc:
            return handler.error_json(502, str(exc))
        except Exception as exc:
            print(f'hub api {parsed.path}: {exc}', file=sys.stderr, flush=True)
            return handler.error_json(500, str(exc))
        if result is not None:
            return handler.json_response(result)

    def route(self, handler, method, path, query, body):
        hub = self.hub
        parts = [unquote(part) for part in path.split('/')[2:]]
        head = parts[0]
        if head == 'hub' and parts[1:] == ['parallel']:
            if method == 'GET':
                return hub.parallel.status()
            return hub.parallel.start(body.get('roots') or [], body.get('features') or {},
                                      body.get('video_features'), force=bool(body.get('force')),
                                      visual_model=body.get('visual_model'),
                                      cores=body.get('cores') or None)
        if head == 'hub' and parts[1:] == ['parallel', 'stop'] and method == 'POST':
            return hub.parallel.stop()
        if head == 'hub' and method == 'GET':
            return {'cores': hub.cores_payload(), 'sources': self.sources_payload()}
        if head == 'cores':
            return self.cores(method, parts[1:], body)
        if head == 'sources':
            return self.sources(method, parts[1:], query, body)
        if head == 'storage':
            db = sqlite3.connect(hub.database, timeout=60)
            try:
                if method == 'GET':
                    return hub.storage(db)
                if parts[1:] == ['clean'] and method == 'POST':
                    result = hub.clean(db, str(body.get('source') or ''), body.get('what') or [])
                    self.after_change()
                    return {'ok': True, 'removed': result, **hub.storage(db)}
            finally:
                db.close()
        if head == 'imports':
            return self.imports(method, parts[1:], body)
        if head == 'media':
            return self.media(method, parts[1:], query, body)
        if head == 'scan' and parts[1:] == ['history'] and method == 'GET':
            return self.app.device.history()
        if head == 'scan' and parts[1:] == ['history', 'forget'] and method == 'POST':
            return self.app.device.forget(body.get('id'))
        raise KeyError('Нет такого адреса')

    def media(self, method, parts, query, body):
        media = self.app.media
        if parts == ['probe'] and method == 'GET':
            return media.probe(self.app.video_key(query.get('path', '')))
        if parts == ['start'] and method == 'POST':
            op = str(body.get('op') or '')
            key = self.app.video_key(body.get('path', ''), replace=op == 'replace')
            return media.start(key, op, body.get('params') or {})
        if len(parts) == 2 and parts[0] == 'jobs' and method == 'GET':
            return media.status(parts[1])
        if len(parts) == 3 and parts[0] == 'jobs' and parts[2] == 'cancel' and method == 'POST':
            return media.cancel(parts[1])
        raise KeyError('Нет такого адреса')

    def after_change(self):
        """Каталог поменялся мимо App — сбрасываем его запомненное."""
        with self.app.lock:
            self.app.group_cache.clear()
            self.app.dup_cache.clear()
            self.app.store.reload_faces()
            self.app.folders.refresh(force=True)

    # ----- ядра -----

    def cores(self, method, parts, body):
        hub = self.hub
        if not parts and method == 'GET':
            return hub.cores_payload()
        if parts == ['save'] and method == 'POST':
            record = hub.cores.save(body)
            with hub.status_lock:
                hub.status_cache.pop(record['id'], None)
            hub.refresh_sources()
            return {'ok': True, 'core': CoreRegistry.public(record)}
        if parts == ['remove'] and method == 'POST':
            hub.cores.remove(str(body.get('id') or ''))
            return {'ok': True}
        core_id = parts[0]
        core = hub.cores.get(core_id)
        if core is None:
            raise KeyError('Устройство не найдено')
        action = '/'.join(parts[1:])
        if action == 'install' and method == 'GET':
            return hub.install_state(core_id) or {'status': 'idle'}
        if action == 'install' and method == 'POST':
            return Installer.start(hub, core, str(body.get('action') or 'update'))
        if action == 'start' and method == 'POST':
            return Installer.start(hub, core, 'start')
        if action == 'components' and method == 'GET':
            return hub.components(core)
        if action == 'components/start' and method == 'POST':
            return hub.start_component(core, body)
        if action == 'components/stop' and method == 'POST':
            return hub.core_call(core, '/api/core/components/stop', 'POST', {})
        if action == 'job/start' and method == 'POST':
            job = self.app.device.start(body.get('roots', []), body.get('features', {}),
                                        body.get('paths', []), force=bool(body.get('force')),
                                        visual_model=body.get('visual_model'),
                                        video_features=body.get('video_features'),
                                        core_id=core_id, resume=bool(body.get('resume')))
            with hub.status_lock:
                hub.status_cache.pop(core_id, None)
            return {'ok': True, 'job': job}
        if action == 'job/stop' and method == 'POST':
            return {'ok': True, 'job': self.app.device.stop(core_id)}
        if action == 'export-legacy':
            if method == 'POST':
                return hub.core_call(core, '/api/core/export-legacy', 'POST', {}, timeout=30)
            return hub.core_call(core, '/api/core/export-legacy', timeout=10)
        raise KeyError('Нет такого адреса')

    # ----- источники -----

    def sources_payload(self):
        hub = self.hub
        stats = {item['id']: item for item in self.storage_brief()}
        rows = []
        for record in hub.sources.load():
            rows.append({**sources.public(record), 'rootKey': sources.root_key(record),
                         'stats': stats.get(record['id'], {}),
                         'health': hub.health.status(record['id'])})
        return {'sources': rows, 'types': sources.TYPES,
                'devices': [{'id': row['id'], 'name': row['name']} for row in hub.cores.load()]}

    def health_payload(self):
        """Только доступность и имена источников — для плиток и просмотрщика любого
        зрителя: без адресов, логинов и статистики, и дёшево (ничего не проверяет)."""
        result = {}
        for record in self.hub.sources.load():
            status = self.hub.health.status(record['id']) or {'online': True, 'checked_at': None}
            # Текст ошибки не отдаём: в нём бывают адреса и логины.
            result[record['id']] = {'name': record.get('name') or record['id'],
                                    'online': status['online'], 'checked_at': status['checked_at']}
        return {'sources': result}

    def storage_brief(self):
        db = sqlite3.connect(self.hub.database, timeout=30)
        try:
            return self.hub.storage(db)['sources']
        finally:
            db.close()

    def sources(self, method, parts, query, body):
        hub = self.hub
        if not parts and method == 'GET':
            return self.sources_payload()
        if parts == ['health'] and method == 'GET':
            return self.health_payload()
        if parts == ['health'] and method == 'POST':
            hub.health.check_all()
            return self.sources_payload()
        if parts == ['save'] and method == 'POST':
            record = hub.sources.save(body)
            hub.refresh_sources()
            # Поменяли подключение — старый вердикт о доступности больше не про него.
            threading.Thread(target=hub.health.check, args=(record,), daemon=True).start()
            return {'ok': True, 'source': sources.public(record)}
        if parts == ['remove'] and method == 'POST':
            source_id = str(body.get('id') or '')
            if body.get('purge'):
                db = sqlite3.connect(hub.database, timeout=60)
                try:
                    hub.clean(db, source_id, ['catalog'])
                finally:
                    db.close()
                self.after_change()
            hub.sources.remove(source_id)
            hub.refresh_sources()
            return {'ok': True}
        if parts == ['test'] and method == 'POST':
            current = hub.sources.get(str(body.get('id') or '').lower())
            record = sources.validate(body, current)
            probe = sources.Access([record], device_route=hub.device_route, ssh_key=hub.ssh_key)
            try:
                driver = probe.driver(record['id'])
                driver.test()
                roots = [item.name for item in driver.roots()][:20]
            finally:
                probe.close()
            return {'ok': True, 'roots': roots}
        if parts == ['browse'] and method == 'GET':
            return self.browse(query.get('source', ''), query.get('path', ''))
        if parts == ['tree'] and method == 'GET':
            import catalog_index
            db = catalog_index.connect(hub.catalog)
            try:
                return catalog_index.tree(db, query.get('path', ''))
            finally:
                db.close()
        if parts == ['exclusions'] and method == 'POST':
            import catalog_index
            db = catalog_index.connect(hub.catalog)
            try:
                result = catalog_index.set_exclusions(db, body.get('add', []),
                                                      body.get('remove', []))
            finally:
                db.close()
            self.after_change()
            return result
        raise KeyError('Нет такого адреса')

    def browse(self, source_id, key):
        """Папки источника для выбора, что сканировать. Пусто — корни источника."""
        hub = self.hub
        record = hub.sources.get(source_id)
        if record is None:
            raise KeyError('Источник не найден')
        driver = hub.access.driver(source_id)
        if not key:
            base = record.get('path') if record['type'] not in {'smb', 'device'} else ''
            items = [sources.Entry(base, True)] if base else driver.roots()
            return {'source': source_id, 'path': '', 'parent': None, 'directories': [
                {'name': item.name, 'path': pathkeys.make(source_id, item.name)}
                for item in items]}
        native = pathkeys.native(key)
        entries = driver.listdir(native)
        directories = sorted((item for item in entries
                              if item.is_dir and not item.name.startswith(('.', '$'))),
                             key=lambda item: item.name.casefold())[:500]
        media = sum(1 for item in entries
                    if not item.is_dir and pathkeys.suffix(item.name) in MEDIA_SUFFIXES)
        parent = pathkeys.parent(key)
        if parent == key or key == sources.root_key(record) or pathkeys.is_root(key):
            parent = ''
        return {'source': source_id, 'path': key, 'media': media,
                'parent': parent,
                'directories': [{'name': item.name, 'path': pathkeys.join(key, item.name)}
                                for item in directories]}

    # ----- перенос старых каталогов -----

    def imports(self, method, parts, body):
        hub = self.hub
        if not parts and method == 'GET':
            files = sorted(path.name for path in hub.imports.glob('*')
                           if path.is_file() and not path.name.endswith('.part')) \
                if hub.imports.is_dir() else []
            return {'files': files, 'merges': self.merges}
        if parts == ['merge'] and method == 'POST':
            name = str(body.get('catalog') or '')
            source_id = str(body.get('source') or '')
            catalog = hub.imports / Path(name).name
            if not catalog.is_file():
                raise KeyError('Файл каталога не найден')
            if hub.sources.get(source_id) is None:
                raise ValueError('Сначала заведите источник, к которому относится каталог')
            thumbnails = None
            if body.get('thumbnails'):
                thumbnails = hub.imports / Path(str(body['thumbnails'])).name
            state = self.merges[name] = {'status': 'running', 'source': source_id,
                                         'started_at': time.time(), 'error': ''}

            def run():
                import migrate
                try:
                    if thumbnails is not None and thumbnails.is_file():
                        import tarfile
                        with tarfile.open(thumbnails) as tar:
                            for member in tar.getmembers():
                                if (member.isfile() and member.name.startswith('thumbnails/')
                                        and '..' not in member.name):
                                    tar.extract(member, hub.catalog)
                    with self.app.lock:
                        result = migrate.merge(hub.database, catalog, source_id)
                    state.update(status='completed', result=result, finished_at=time.time())
                    owner = re.match(r'^(.+)-\d{8}-\d{6}-catalog\.sqlite$', catalog.name)
                    if owner and hub.cores.get(owner.group(1)):
                        hub.cores.update(owner.group(1), legacy_merged=time.time())
                    self.after_change()
                except Exception as exc:
                    state.update(status='error', error=str(exc), finished_at=time.time())
            threading.Thread(target=run, daemon=True).start()
            return state
        raise KeyError('Нет такого адреса')


MEDIA_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.mp4', '.mov', '.m4v', '.avi', '.mkv',
                  '.webm', '.3gp', '.3g2', '.mpg', '.mpeg', '.mts', '.m2ts', '.wmv', '.flv'}


# ---------- установка и обновление ядер по SSH ----------

class Installer:
    """Ставит, обновляет и запускает ядро на Windows-устройстве по SSH.

    Пакет — zip исходников ядра, опубликованный на хабе (core-packages).
    На устройство уходят пакет и Install-Core.ps1; скрипт раскладывает код,
    пишет hub.json и токен, при установке ставит окружение и автозапуск, и
    перезапускает службу ядра.
    """

    def __init__(self, hub, core, action):
        self.hub = hub
        self.core = core
        self.action = action
        self.state = {'status': 'running', 'action': action, 'started_at': time.time(),
                      'finished_at': None, 'error': '', 'log': [], 'step': 'connect'}

    @classmethod
    def start(cls, hub, core, action):
        if action not in {'install', 'update', 'start', 'key'}:
            raise ValueError('Неизвестное действие')
        current = hub.installs.get(core['id'])
        if current and current['status'] == 'running':
            return current
        if not (core.get('ssh') or {}).get('user'):
            raise ValueError('Для установки и запуска укажите пользователя SSH устройства')
        if action == 'key':
            hub.ensure_ssh_key()
        elif action != 'start' and hub.package_file() is None:
            raise ValueError('На хабе нет пакета ядра: опубликуйте его '
                             '(homecloud-core\\deploy.ps1)')
        job = cls(hub, core, action)
        hub.installs[core['id']] = job.state
        threading.Thread(target=job.run, daemon=True).start()
        return job.state

    def log(self, text):
        for line in str(text).splitlines():
            line = line.rstrip()
            if line:
                self.state['log'].append(line[:500])
        del self.state['log'][:-400]

    def connect(self, key_only=False):
        import paramiko
        ssh = self.core['ssh']
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        key = self.hub.ssh_key if self.hub.ssh_key and Path(self.hub.ssh_key).is_file() else None
        password = None if key_only else ssh.get('password') or None
        client.connect(ssh.get('host') or self.core['host'], port=int(ssh.get('port') or 22),
                       username=ssh['user'], password=password,
                       key_filename=key, look_for_keys=False, allow_agent=False, timeout=15,
                       banner_timeout=20, auth_timeout=20)
        return client

    def run_command(self, client, command, timeout=7200):
        self.log('> ' + (command if 'EncodedCommand' not in command else 'powershell …'))
        channel = client.get_transport().open_session()
        channel.set_combine_stderr(True)
        channel.settimeout(timeout)
        channel.exec_command(command)
        buffer = b''
        while True:
            data = channel.recv(65536)
            if not data:
                break
            buffer += data
            *lines, buffer = buffer.split(b'\n')
            for line in lines:
                self.log(decode_console(line))
        if buffer:
            self.log(decode_console(buffer))
        return channel.recv_exit_status()

    def run(self):
        client = None
        try:
            client = self.connect()
            install_dir = self.core['install_dir'].rstrip('\\/')
            if self.action == 'start':
                self.state['step'] = 'start'
                code = self.run_command(client, powershell(
                    f'& "{install_dir}\\start-remote.ps1"'), timeout=120)
                if code:
                    raise RuntimeError(f'Запуск завершился с кодом {code}')
            elif self.action == 'key':
                self.trust_key(client)
            else:
                self.deploy(client, install_dir)
            self.state.update(status='completed', step='done', finished_at=time.time())
        except Exception as exc:
            self.log(f'Ошибка: {exc}')
            self.state.update(status='error', error=str(exc), finished_at=time.time())
        finally:
            if client is not None:
                client.close()

    def trust_key(self, client):
        """Кладёт ключ хаба на устройство, проверяет вход по нему и забывает пароль."""
        public = self.hub.public_key()
        if not public:
            raise RuntimeError('Не удалось прочитать открытый ключ хаба')
        self.state['step'] = 'key'
        code = self.run_command(client, powershell(trust_key_script(public)), timeout=60)
        if code:
            raise RuntimeError(f'Не удалось записать ключ (код {code})')
        self.state['step'] = 'verify'
        try:
            check = self.connect(key_only=True)
        except Exception as exc:
            raise RuntimeError(f'Ключ записан, но вход по нему не удался: {exc}. '
                               'Пароль оставлен') from exc
        check.close()
        self.log('Вход по ключу хаба работает')
        if (self.core.get('ssh') or {}).get('password'):
            ssh = {k: v for k, v in self.core['ssh'].items() if k != 'password'}
            self.hub.cores.update(self.core['id'], ssh=ssh)
            self.core['ssh'] = ssh
            self.log('Пароль удалён из настроек хаба')

    def deploy(self, client, install_dir):
        import zipfile
        package = self.hub.package_file()
        info = self.hub.package_info()
        self.state['step'] = 'upload'
        # Не в work: на рабочих машинах это бывает ссылка, а SFTP через неё не пишет.
        work = f'{install_dir}\\core-data\\update'
        code = self.run_command(client, powershell(
            f'New-Item -ItemType Directory -Force -Path "{work}" | Out-Null'), timeout=60)
        if code:
            raise RuntimeError('Не удалось создать папку установки')
        sftp = client.open_sftp()
        try:
            sftp.put(str(package), sftp_path(f'{work}\\{package.name}'))
            with zipfile.ZipFile(package) as archive:
                script = archive.read('Install-Core.ps1')
            with sftp.open(sftp_path(f'{work}\\Install-Core.ps1'), 'wb') as output:
                output.write(script)
        finally:
            sftp.close()
        self.log(f'Пакет {info["version"]} загружен')
        self.state['step'] = 'install'
        arguments = (
            f'-Package "{work}\\{package.name}" -Repository "{install_dir}" '
            f'-Mode {self.action} -HubUrl "{self.hub.link_url}" '
            f'-CoreId "{self.core["id"]}" -CoreName "{self.core["name"]}" '
            f'-LinkToken "{self.core["link_token"]}" -CoreToken "{self.core["token"]}"')
        code = self.run_command(client, powershell(
            f'& "{work}\\Install-Core.ps1" {arguments}'), timeout=4 * 3600)
        if code:
            raise RuntimeError(f'Установка завершилась с кодом {code}')
        self.state['step'] = 'wait'
        deadline = time.time() + 180
        while time.time() < deadline:
            with self.hub.status_lock:
                self.hub.status_cache.pop(self.core['id'], None)
            status = self.hub.core_status(self.core['id'])
            version = (status.get('device') or {}).get('version')
            if status.get('online') and not status.get('legacy'):
                self.log(f'Ядро отвечает, версия {version}')
                return
            time.sleep(3)
        raise RuntimeError('Ядро не ответило за три минуты после установки')


def powershell(script):
    """Команда PowerShell для OpenSSH на Windows: скрипт в base64, без кавычечной каши."""
    encoded = base64.b64encode(("$ProgressPreference='SilentlyContinue';"
                                "[Console]::OutputEncoding=[Text.Encoding]::UTF8;" + script)
                               .encode('utf-16-le')).decode('ascii')
    return f'powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}'


TRUST_KEY_SCRIPT = r"""$ErrorActionPreference = 'Stop'
$k = '__KEY__'
$id = ($k -split ' ')[0..1] -join ' '
function Add-Key($file) {
  $dir = Split-Path $file
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  $text = if (Test-Path $file) { [IO.File]::ReadAllText($file) } else { '' }
  if ($text.Contains($id)) { "Ключ уже есть: $file"; return }
  if ($text.Length -and -not $text.EndsWith("`n")) { $text += "`r`n" }
  [IO.File]::WriteAllText($file, $text + $k + "`r`n", (New-Object Text.UTF8Encoding $false))
  "Ключ добавлен: $file"
}
Add-Key (Join-Path $env:USERPROFILE '.ssh\authorized_keys')
$me = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if ($me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  # Для администраторов sshd читает только этот файл, и только с такими правами.
  $file = Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'
  Add-Key $file
  icacls $file /inheritance:r /grant '*S-1-5-32-544:F' /grant '*S-1-5-18:F' | Out-Null
  if ($LASTEXITCODE) { throw "icacls: код $LASTEXITCODE" }
}
"""


def trust_key_script(public):
    """PowerShell: добавить открытый ключ хаба в authorized_keys пользователя SSH."""
    if not re.match(r'^[\w-]+ [A-Za-z0-9+/=]+( [\w.@-]+)?$', public):
        raise ValueError('Странный открытый ключ хаба')
    return TRUST_KEY_SCRIPT.replace('__KEY__', public)


def sftp_path(windows):
    return '/' + windows.replace('\\', '/')


def decode_console(raw):
    for encoding in ('utf-8', 'cp866', 'cp1251'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', 'replace')
