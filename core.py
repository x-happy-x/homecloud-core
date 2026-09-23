"""Служба ядра: web_server.py с `--role core` на компьютере с видеокартой.

Ядро ничего не хранит у себя: каталог — на хабе (соединение catalogdb по
hub.json), превью и миниатюры лиц уходят туда же. Здесь только работа:

* задания сканирования по источникам (device_job.py с ключами источников);
* пересборка групп лиц, подборки, дубликаты, обучение роутера — то, что
  хаб раньше считал сам, а теперь присылает сюда (/api/core/task/…);
* смысловой поиск: текст в вектор (/api/core/semantic);
* диск этого компьютера как источник: список, сведения и чтение файлов для
  хаба (/api/core/list|stat|read), удаление и перенос (/api/core/fileop);
* временные копии файлов с источников, которые нельзя открыть как путь
  (/api/core/stage) — для этапов в других окружениях;
* перенос старого каталога этого устройства в общий (/api/core/export-legacy).

Все запросы — с X-Local-Token ядра: его знает только хаб.
"""
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import catalogdb
import components
import hublink
import job_features
import pathkeys
import settings as catalog_settings
import sources

ROOT = Path(__file__).resolve().parent


def version():
    """Версия кода: VERSION.json из пакета, у рабочей копии — коммит git."""
    try:
        return json.loads((ROOT / 'VERSION.json').read_text(encoding='utf-8'))['version']
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    try:
        result = subprocess.run(['git', 'describe', '--always', '--dirty'], cwd=ROOT,
                                capture_output=True, text=True, timeout=10,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode == 0 and result.stdout.strip():
            return 'dev-' + result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return 'dev'


class CoreApp:
    def __init__(self, folder, token, device_id, device_name, port, legacy=None):
        from web_server import (DeviceController, DuplicateService, HighlightService,
                                ReclusterController, RouterController, SemanticService)
        self.folder = Path(folder).resolve()
        self.token = token
        self.port = port
        self.device_id = device_id
        self.device_name = device_name
        self.legacy = Path(legacy).resolve() if legacy else None
        self.link = hublink.link_for(self.folder)
        if self.link is None:
            raise SystemExit(f'Нет {self.folder / hublink.CONFIG_NAME}: ядро не знает хаба')
        self.version = version()
        self.config = {'sources': [], 'core': device_id}
        self.access = sources.Access([], device_id=device_id)
        self.stage = sources.Stage(self.folder / 'stage', self.access)
        sources.set_resolver(self.resolve)
        self.lock = threading.RLock()
        self.hello()
        self.device = CoreDevice(self, DeviceController)
        shim = SimpleNamespace(store=SimpleNamespace(folder=self.folder, min_cluster_size=8),
                               lock=threading.RLock(), identity_reload_pending=False)
        self.tasks = {
            'recluster': Task(ReclusterController(shim)),
            'highlights': Task(HighlightService(self.folder)),
            'duplicates': Task(DuplicateService(self.folder)),
            'router': Task(RouterController(self.folder)),
        }
        self.semantic = SemanticService(self.folder)
        self.components = components.Components(log_folder=self.folder)
        self.exports = {}

    # ----- связь с хабом -----

    def hello(self):
        info = {'version': self.version, 'hostname': socket.gethostname(),
                'port': self.port, 'legacy': self.legacy_info()}
        try:
            config = self.link.json('POST', '/hello', info, timeout=30)
        except hublink.HubError as exc:
            print(f'Хаб недоступен: {exc}', file=sys.stderr, flush=True)
            return False
        self.apply_config(config)
        return True

    def refresh(self):
        try:
            self.apply_config(self.link.json('GET', '/config', timeout=30))
        except hublink.HubError as exc:
            if not self.config.get('sources'):
                raise ValueError(f'Хаб недоступен: {exc}') from exc

    def apply_config(self, config):
        with self.lock:
            self.config = config
            self.access.update(config.get('sources', []))
            unc = {}
            for record in config.get('sources', []):
                if record['type'] == 'smb' and os.name == 'nt' and record.get('share'):
                    try:
                        unc[record['id']] = bool(self.access.driver(record['id']).local_path(
                            '/' + record['share']))
                    except Exception:
                        unc[record['id']] = False
            payload = {'core': self.device_id, 'sources': config.get('sources', []),
                       'agent': {'url': f'http://127.0.0.1:{self.port}', 'token': self.token},
                       'unc': unc}
            target = self.folder / 'sources.json'
            temporary = target.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            os.replace(temporary, target)

    def resolve(self, key):
        """Локальный путь к файлу источника для этапа обработки."""
        return self.stage.path(key)

    # ----- диск этого компьютера -----

    def local_path(self, raw):
        path = Path(str(raw or ''))
        if not path.is_absolute():
            raise ValueError('Нужен полный путь')
        return path

    def list_dir(self, raw):
        path = self.local_path(raw)
        return {'path': str(path), 'entries': [
            {'name': item.name, 'dir': item.is_dir, 'size': item.size, 'mtime_ns': item.mtime_ns}
            for item in sources.LocalDriver().listdir(str(path))]}

    def stat_path(self, raw):
        path = self.local_path(raw)
        info = path.stat()
        return {'name': path.name or str(path), 'dir': path.is_dir(), 'size': info.st_size,
                'mtime_ns': info.st_mtime_ns}

    def file_operation(self, body):
        """Удаление и перенос файлов на своём диске — по просьбе хаба."""
        op = body.get('op')
        if op == 'delete':
            from send2trash import send2trash
            done, errors = [], []
            for raw in body.get('paths') or []:
                try:
                    send2trash(str(self.local_path(raw)))
                    done.append(raw)
                except OSError as exc:
                    errors.append({'path': raw, 'error': str(exc)})
            return {'done': done, 'errors': errors}
        if op == 'move':
            done, errors = [], []
            for old, new in body.get('moves') or []:
                try:
                    source, target = self.local_path(old), self.local_path(new)
                    if target.exists():
                        raise OSError('Файл с таким именем уже есть')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target))
                    done.append([old, new])
                except OSError as exc:
                    errors.append({'path': old, 'error': str(exc)})
            return {'done': done, 'errors': errors}
        if op == 'rmdir':
            self.local_path(body.get('path')).rmdir()
            return {'done': [body.get('path')], 'errors': []}
        if op == 'isdir':
            return {'dir': self.local_path(body.get('path')).is_dir()}
        raise ValueError('Неизвестная операция с файлами')

    # ----- старый каталог -----

    def legacy_info(self):
        if not self.legacy:
            return None
        database = self.legacy / 'catalog.sqlite'
        if not database.is_file():
            return None
        try:
            db = sqlite3.connect(f'file:{database.as_posix()}?mode=ro', uri=True, timeout=5)
            try:
                photos = db.execute("SELECT COUNT(*) FROM photos WHERE status='ok'").fetchone()[0]
                faces = db.execute('SELECT COUNT(*) FROM faces').fetchone()[0]
                people = db.execute('SELECT COUNT(*) FROM people').fetchone()[0]
            finally:
                db.close()
        except sqlite3.Error:
            return {'path': str(database), 'bytes': database.stat().st_size}
        return {'path': str(database), 'bytes': database.stat().st_size, 'photos': photos,
                'faces': faces, 'people': people}

    def export_legacy(self):
        """Снимок старого каталога и миниатюр лиц — на хаб, для переноса в общий."""
        if not self.legacy or not (self.legacy / 'catalog.sqlite').is_file():
            raise ValueError('На этом устройстве нет старого каталога')
        state = self.exports.get('legacy')
        if state and state.get('status') == 'running':
            return state
        state = self.exports['legacy'] = {'status': 'running', 'step': 'snapshot',
                                          'started_at': time.time(), 'error': ''}

        def run():
            work = Path(tempfile.mkdtemp(prefix='homecloud-export-', dir=self.folder))
            try:
                snapshot = work / 'catalog.sqlite'
                source = sqlite3.connect(self.legacy / 'catalog.sqlite', timeout=60)
                target = sqlite3.connect(snapshot)
                try:
                    source.backup(target)
                finally:
                    target.close()
                    source.close()
                state['step'] = 'thumbnails'
                archive = work / 'thumbnails.tar'
                with tarfile.open(archive, 'w') as tar:
                    folder = self.legacy / 'thumbnails'
                    if folder.is_dir():
                        tar.add(folder, arcname='thumbnails')
                stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                for step, path, name in (('upload-catalog', snapshot, f'{stamp}-catalog.sqlite'),
                                         ('upload-thumbnails', archive, f'{stamp}-thumbnails.tar')):
                    state['step'] = step
                    upload(self.link, name, path)
                state.update(status='completed', step='done', finished_at=time.time(),
                             catalog=f'{self.device_id}-{stamp}-catalog.sqlite',
                             thumbnails=f'{self.device_id}-{stamp}-thumbnails.tar')
            except Exception as exc:
                state.update(status='error', error=str(exc), finished_at=time.time())
            finally:
                shutil.rmtree(work, ignore_errors=True)
        threading.Thread(target=run, daemon=True).start()
        return state


def upload(link, name, path):
    """Большой файл на хаб одним PUT: поток кусками, без чтения целиком в память."""
    import http.client
    from urllib.parse import quote
    size = path.stat().st_size
    connection = http.client.HTTPConnection(link.host, link.port, timeout=600)
    try:
        connection.putrequest('PUT', '/import/' + quote(name))
        connection.putheader('X-Core-Token', link.token)
        connection.putheader('X-Core-Id', link.core)
        connection.putheader('Content-Length', str(size))
        connection.putheader('Content-Type', 'application/octet-stream')
        connection.endheaders()
        with open(path, 'rb') as stream:
            while True:
                chunk = stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        response = connection.getresponse()
        body = response.read()
        if response.status >= 400:
            raise RuntimeError(body[:300].decode('utf-8', 'replace'))
    finally:
        connection.close()


class Task:
    """Обёртка вычислительной службы: единый вид status/start/stop для хаба."""

    def __init__(self, controller):
        self.controller = controller

    def status(self):
        value = self.controller.status()
        if 'active' in value and 'status' not in value:
            value['status'] = 'running' if value['active'] else 'idle'
        if value.get('active') and value.get('status') not in {'running', 'preparing'}:
            value['status'] = 'running'
        return value

    def start(self, args, kwargs):
        return self.controller.start(*args, **kwargs)

    def stop(self):
        return self.controller.stop()


class CoreDevice:
    """DeviceController ядра: корни и снимки — ключи источников, а не пути диска."""

    def __init__(self, app, base):
        self.app = app
        self.controller = base(app.folder, app.device_id, app.device_name)

    def info(self):
        value = self.controller.info()
        value.update(role='core', version=self.app.version,
                     legacy=self.app.legacy_info(),
                     sources=[record['id'] for record in self.app.config.get('sources', [])])
        return value

    def status(self):
        return self.controller.status()

    def browse(self, raw_path=''):
        return self.controller.browse(raw_path)

    def stop(self):
        return self.controller.stop()

    def start(self, roots, features, paths=None, force=False, visual_model=None,
              video_features=None):
        controller = self.controller
        self.app.refresh()
        with controller.lock:
            if controller.status()['active']:
                raise ValueError('На ядре уже выполняется задание')
            known = {record['id']: record for record in self.app.config.get('sources', [])}
            chosen = []
            for raw in [*(roots or []), *(paths or [])]:
                source = pathkeys.source_of(raw)
                if source not in known:
                    raise ValueError(f'Неизвестный источник: {raw}')
            for raw in roots or []:
                key = pathkeys.trim(str(raw))
                if key not in chosen:
                    chosen.append(key)
            selected_paths = list(dict.fromkeys(str(path) for path in (paths or [])))
            if len(selected_paths) > 500:
                raise ValueError('За один запуск можно выбрать не более 500 файлов')
            if not chosen and not selected_paths:
                raise ValueError('Выберите хотя бы одну папку или снимок')
            supported = controller.info()['capabilities']
            selected, kinds = job_features.resolve(features, video_features, supported)
            # Превью сетки и сведения о файлах обновляются при каждом обходе источника.
            if chosen:
                selected['inventory'] = True
            if not any(selected.values()):
                raise ValueError('Выберите хотя бы одну возможность')
            unavailable = [name for name, enabled in selected.items()
                           if enabled and not supported.get(name)]
            if unavailable:
                raise ValueError('Недоступно на ядре: ' + ', '.join(unavailable))
            if selected.get('visual') and visual_model is not None:
                available = {item['id']: item for item in catalog_settings.visual_models(
                    controller.model_cache)}
                if str(visual_model) not in available:
                    raise ValueError('Неизвестная модель визуального индекса')
                if not available[str(visual_model)]['installed']:
                    raise ValueError('Модель ещё не скачана на это ядро')
                db = catalogdb.connect(self.app.folder)
                try:
                    catalog_settings.write(db, {'visual_model': str(visual_model)})
                finally:
                    db.close()
            controller.stop_file.unlink(missing_ok=True)
            controller.remember(chosen, selected_paths, selected)
            controller.progress_file.write_text(json.dumps({
                'status': 'preparing', 'phase': 'inventory', 'roots': chosen,
                'paths': selected_paths, 'features': selected, 'kinds': kinds, 'total': 0,
                'completed': 0, 'updated_at': time.time(), 'pid': os.getpid()},
                ensure_ascii=False), encoding='utf-8')
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            command = [sys.executable, str(ROOT / 'device_job.py'),
                       '--catalog', str(self.app.folder), '--features', json.dumps(selected),
                       '--kinds', json.dumps(kinds),
                       '--progress-file', str(controller.progress_file),
                       '--stop-file', str(controller.stop_file)]
            if force:
                command.append('--force')
            for key in chosen:
                command.extend(('--root', key))
            for key in selected_paths:
                command.extend(('--path', key))
            env = {**os.environ, 'HOMECLOUD_CORE_DIR': str(self.app.folder), 'PYTHONUTF8': '1'}
            controller.process = subprocess.Popen(
                command, cwd=ROOT, creationflags=flags, env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return controller.status()


# ---------- HTTP ----------

from http.server import BaseHTTPRequestHandler  # noqa: E402


class CoreHandler(BaseHTTPRequestHandler):
    server_version = 'HomeCloudCore/1.0'

    @property
    def app(self):
        return self.server.app

    def log_message(self, format, *args):
        path = self.path.split('?', 1)[0]
        if path in {'/api/device', '/api/device/job', '/api/core/stage', '/api/core/read',
                    '/api/core/components', '/api/core/components/file'} \
                or path.startswith('/api/core/task/'):
            return
        print(f'{self.address_string()} - {format % args}', flush=True)

    def json_response(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def error_json(self, status, message):
        self.json_response({'error': message}, status)

    # Другое ядро с пропуском от хаба может только читать модели.
    PEER_PATHS = {'/api/core/components/manifest', '/api/core/components/file'}

    def authorized(self, path=''):
        if secrets.compare_digest(self.headers.get('X-Local-Token', ''), self.app.token):
            return True
        return path in self.PEER_PATHS and self.app.components.ticket_ok(
            self.headers.get('X-Core-Ticket', ''))

    def body(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length > 4 * 1024 * 1024:
            raise ValueError('Запрос слишком большой')
        return json.loads(self.rfile.read(length) or b'{}') if length else {}

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path
        if not self.authorized(path):
            return self.error_json(403, 'Неверный токен ядра')
        try:
            if path == '/api/core/components':
                return self.json_response(self.app.components.status())
            if path == '/api/core/components/manifest':
                return self.json_response(self.app.components.manifest(query.get('id', [''])[0]))
            if path == '/api/core/components/file':
                return self.send_range(self.app.components.file_path(
                    query.get('id', [''])[0], query.get('path', [''])[0]))
            if path == '/api/device':
                return self.json_response(self.app.device.info())
            if path == '/api/device/job':
                return self.json_response(self.app.device.status())
            if path == '/api/device/browse':
                return self.json_response(self.app.device.browse(query.get('path', [''])[0]))
            if path.startswith('/api/core/task/'):
                task = self.app.tasks[path.rsplit('/', 1)[-1]]
                return self.json_response(task.status())
            if path == '/api/core/list':
                return self.json_response(self.app.list_dir(query.get('path', [''])[0]))
            if path == '/api/core/stat':
                return self.json_response(self.app.stat_path(query.get('path', [''])[0]))
            if path == '/api/core/read':
                return self.send_range(self.app.local_path(query.get('path', [''])[0]))
            if path == '/api/core/stage':
                return self.json_response({'path': self.app.stage.path(query.get('key', [''])[0])})
            if path == '/api/core/export-legacy':
                return self.json_response(self.app.exports.get('legacy') or {'status': 'idle'})
            return self.error_json(404, 'Нет такого адреса')
        except FileNotFoundError as exc:
            return self.error_json(404, f'Нет файла: {exc}')
        except (KeyError, ValueError, sources.SourceError) as exc:
            return self.error_json(400, str(exc))
        except Exception as exc:
            print(f'GET {path} failed: {exc}', file=sys.stderr, flush=True)
            return self.error_json(500, str(exc))

    def do_POST(self):
        if not self.authorized():
            return self.error_json(403, 'Неверный токен ядра')
        path = urlparse(self.path).path
        try:
            body = self.body()
            if path == '/api/device/job/start':
                return self.json_response({'ok': True, 'job': self.app.device.start(
                    body.get('roots', []), body.get('features', {}), body.get('paths', []),
                    force=bool(body.get('force')), visual_model=body.get('visual_model'),
                    video_features=body.get('video_features'))})
            if path == '/api/device/job/stop':
                return self.json_response({'ok': True, 'job': self.app.device.stop()})
            if path.startswith('/api/core/task/') and path.endswith('/start'):
                task = self.app.tasks[path.split('/')[-2]]
                self.app.refresh()
                return self.json_response({'ok': True, 'job': task.start(
                    body.get('args') or [], body.get('kwargs') or {})})
            if path.startswith('/api/core/task/') and path.endswith('/stop'):
                task = self.app.tasks[path.split('/')[-2]]
                return self.json_response({'ok': True, 'job': task.stop()})
            if path == '/api/core/components/start':
                return self.json_response(self.app.components.start(
                    str(body.get('id') or ''), str(body.get('action') or ''), body.get('peer')))
            if path == '/api/core/components/stop':
                return self.json_response(self.app.components.stop())
            if path == '/api/core/components/ticket':
                return self.json_response(self.app.components.issue_ticket())
            if path == '/api/core/semantic':
                return self.json_response({'results': self.app.semantic.query(
                    str(body.get('text') or ''), int(body.get('top') or 500))})
            if path == '/api/core/fileop':
                return self.json_response(self.app.file_operation(body))
            if path == '/api/core/refresh':
                self.app.refresh()
                return self.json_response({'ok': True})
            if path == '/api/core/export-legacy':
                return self.json_response(self.app.export_legacy())
            return self.error_json(404, 'Нет такого адреса')
        except (KeyError, ValueError, sources.SourceError) as exc:
            return self.error_json(400, str(exc))
        except Exception as exc:
            print(f'POST {path} failed: {exc}', file=sys.stderr, flush=True)
            return self.error_json(500, str(exc))

    def send_range(self, path):
        total = path.stat().st_size
        header = self.headers.get('Range', '')
        start, end, partial = 0, total - 1, False
        if header.startswith('bytes='):
            first, _, last = header[6:].split(',')[0].partition('-')
            try:
                if first:
                    start, end = int(first), int(last) if last else total - 1
                else:
                    start = max(0, total - int(last))
                partial = True
            except ValueError:
                partial = False
        end = min(end, total - 1)
        if partial and start > end:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{total}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        length = max(0, end - start + 1)
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0]
                         or 'application/octet-stream')
        self.send_header('Content-Length', str(length))
        self.send_header('Accept-Ranges', 'bytes')
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{total}')
        self.end_headers()
        with path.open('rb') as stream:
            stream.seek(start)
            left = length
            while left > 0:
                chunk = stream.read(min(1024 * 1024, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def serve(args, token):
    from http.server import ThreadingHTTPServer
    app = CoreApp(args.data, token, args.device_id or socket.gethostname().casefold(),
                  args.device_name or socket.gethostname(), args.port, args.legacy)
    server = ThreadingHTTPServer((args.host, args.port), CoreHandler)
    server.daemon_threads = True
    server.app = app

    def heartbeat():
        # Хаб перезапустили — он должен снова узнать о ядре и его версии.
        while True:
            time.sleep(300)
            app.hello()
    threading.Thread(target=heartbeat, daemon=True).start()
    print(f'HomeCloud core {app.device_id} {app.version}: порт {args.port}, хаб {app.link.url}',
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
