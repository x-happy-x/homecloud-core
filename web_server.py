"""Private localhost web interface for the local face catalog."""
import argparse
import hashlib
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import io
import ipaddress
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
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urlparse
import webbrowser
import zipfile

import numpy as np
from PIL import Image, ImageOps

import albums
import people_albums
import catalog_index
import duplicates
import pathrules
import privacy
import reverse_search
import router_learning
import settings as catalog_settings
import speaker_diarization
import speech_videos
import video as video_media
from people_gui import CatalogStore
from analyze_photos import connect as analysis_database
from prototype import cluster_embeddings, database as open_catalog_db


WEB_ROOT = Path(__file__).with_name('web').resolve()


class ScanController:
    """Start and observe the resumable D:\\trash scan used by this prototype."""
    def __init__(self):
        self.root = Path(__file__).resolve().parent
        self.catalog = self.root / 'trash-clean-catalog'
        schema = analysis_database(self.catalog)
        schema.close()
        self.progress_file = self.catalog / 'scan-progress.json'
        self.stop_file = self.catalog / 'scan-stop.request'
        self.script = self.root / 'scan-trash.ps1'
        self.process = None
        self.lock = threading.RLock()

    def _progress(self):
        try:
            return json.loads(self.progress_file.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {
                'status': 'idle', 'source': r'D:\trash', 'catalog': str(self.catalog),
                'current': '', 'total': 0, 'completed': 0, 'processed': 0,
                'ignored': 0, 'skipped': 0, 'errors': 0, 'faces_found': 0,
                'updated_at': 0,
            }

    @staticmethod
    def _pid_exists(pid):
        if not isinstance(pid, int) or pid <= 0:
            return False
        if os.name == 'nt':
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def status(self):
        with self.lock:
            payload = self._progress()
            return_code = self.process.poll() if self.process else None
            if self.process and return_code is None:
                if payload.get('status') not in {'counting', 'preparing', 'running'}:
                    payload['status'] = 'preparing'
            elif self.process and return_code is not None:
                if payload.get('status') in {'counting', 'preparing', 'running'}:
                    payload['status'] = 'completed' if return_code == 0 else 'error'
                payload['return_code'] = return_code
                self.process = None
            payload['active'] = payload.get('status') in {'counting', 'preparing', 'running'}
            managed = self.process is not None and self.process.poll() is None
            scanner_alive = managed or self._pid_exists(payload.get('pid'))
            if payload['active'] and not scanner_alive:
                payload['active'] = False
                payload['status'] = 'interrupted'
            payload['stop_requested'] = payload['active'] and self.stop_file.exists()
            database_path = self.catalog / 'catalog.sqlite'
            if database_path.is_file():
                try:
                    db = sqlite3.connect(database_path, timeout=1)
                    payload['catalog_faces'] = db.execute('SELECT COUNT(*) FROM faces').fetchone()[0]
                    payload['catalog_photos'] = db.execute(
                        "SELECT COUNT(*) FROM photos WHERE status='ok'").fetchone()[0]
                    db.close()
                except sqlite3.Error:
                    pass
            return payload

    def start(self):
        with self.lock:
            status = self.status()
            if status['active']:
                raise ValueError('Сканирование уже выполняется')
            self.catalog.mkdir(parents=True, exist_ok=True)
            self.stop_file.unlink(missing_ok=True)
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            self.process = subprocess.Popen(
                ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                 '-File', str(self.script)],
                cwd=self.root, creationflags=flags,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return self.status()

    def stop(self):
        with self.lock:
            status = self.status()
            if not status['active']:
                raise ValueError('Активного сканирования нет')
            self.stop_file.write_text('stop\n', encoding='ascii')
            status['stop_requested'] = True
            return status


class AnalysisController:
    def __init__(self):
        self.root = Path(__file__).resolve().parent
        self.catalog = self.root / 'trash-clean-catalog'
        schema = analysis_database(self.catalog)
        schema.close()
        self.progress_file = self.catalog / 'analysis-progress.json'
        self.stop_file = self.catalog / 'analysis-stop.request'
        self.script = self.root / 'analyze-trash.ps1'
        self.process = None
        self.lock = threading.RLock()

    def _progress(self):
        try:
            return json.loads(self.progress_file.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {'status': 'idle', 'total': 0, 'completed': 0, 'indexed': 0,
                    'errors': 0, 'blurry': 0, 'graphics': 0, 'current': '',
                    'source': str(self.catalog), 'updated_at': 0}

    def status(self):
        with self.lock:
            payload = self._progress()
            return_code = self.process.poll() if self.process else None
            if self.process and return_code is None and payload.get('status') not in {
                    'preparing', 'running'}:
                payload['status'] = 'preparing'
            elif self.process and return_code is not None:
                if payload.get('status') in {'preparing', 'running'}:
                    payload['status'] = 'completed' if return_code == 0 else 'error'
                payload['return_code'] = return_code
                self.process = None
            payload['active'] = payload.get('status') in {'preparing', 'running'}
            managed = self.process is not None and self.process.poll() is None
            if payload['active'] and not (managed or ScanController._pid_exists(payload.get('pid'))):
                payload['active'] = False
                payload['status'] = 'interrupted'
            payload['stop_requested'] = payload['active'] and self.stop_file.exists()
            try:
                db = sqlite3.connect(self.catalog / 'catalog.sqlite', timeout=1)
                exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='photo_analysis'").fetchone()
                if exists:
                    payload['catalog_indexed'] = db.execute(
                        "SELECT COUNT(*) FROM photo_analysis WHERE status='ok'").fetchone()[0]
                    payload['catalog_blurry'] = db.execute(
                        "SELECT COUNT(*) FROM photo_analysis WHERE status='ok' AND blur_score<65").fetchone()[0]
                    payload['catalog_graphics'] = db.execute(
                        "SELECT COUNT(*) FROM photo_analysis WHERE status='ok' AND "
                        "content_type IN ('graphics','game','screenshot','meme')").fetchone()[0]
                    payload['catalog_ocr'] = db.execute(
                        "SELECT COUNT(*) FROM photo_analysis WHERE ocr_status='ok' AND ocr_text!=''").fetchone()[0]
                    payload['catalog_captioned'] = db.execute(
                        "SELECT COUNT(*) FROM photo_analysis WHERE caption_status='ok' AND caption!=''").fetchone()[0]
                db.close()
            except sqlite3.Error:
                pass
            return payload

    def start(self):
        with self.lock:
            if self.status()['active']:
                raise ValueError('AI-анализ уже выполняется')
            self.stop_file.unlink(missing_ok=True)
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            self.process = subprocess.Popen(
                ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                 '-File', str(self.script)], cwd=self.root, creationflags=flags,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            return self.status()

    def stop(self):
        with self.lock:
            status = self.status()
            if not status['active']:
                raise ValueError('Активного AI-анализа нет')
            self.stop_file.write_text('stop\n', encoding='ascii')
            status['stop_requested'] = True
            return status


class SemanticService:
    def __init__(self, catalog):
        self.root = Path(__file__).resolve().parent
        self.catalog = Path(catalog).resolve()
        self.python = (self.root / '..' / '..' / 'work' / 'vision-venv' /
                       'Scripts' / 'python.exe').resolve()
        self.process = None
        self.model = None
        self.lock = threading.RLock()
        self.cache = {}

    def _start(self):
        model = catalog_settings.load(self.catalog)['visual_model']
        if self.process and self.process.poll() is None and self.model == model:
            return
        if self.process and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None
        env = os.environ.copy()
        env['HF_HOME'] = r'C:\cv-models\huggingface'
        env['HF_HUB_DISABLE_XET'] = '1'
        env['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        env['PYTHONUTF8'] = '1'
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self.process = subprocess.Popen(
            [str(self.python), str(self.root / 'analyze_photos.py'), 'serve',
             '--catalog', str(self.catalog), '--model', model], cwd=self.root, env=env,
            creationflags=flags, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1)
        ready = json.loads(self.process.stdout.readline())
        if not ready.get('ready'):
            raise RuntimeError('Семантическая модель не запустилась')
        self.model = model

    def query(self, text, top=500):
        model = catalog_settings.load(self.catalog)['visual_model']
        text_key = text.casefold().strip()
        if not text_key:
            return []
        key = model + '\0' + text_key
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            try:
                self._start()
                self.process.stdin.write(json.dumps({'text': text, 'top': top}) + '\n')
                self.process.stdin.flush()
                response = json.loads(self.process.stdout.readline())
                if response.get('error'):
                    raise RuntimeError(response['error'])
                result = response.get('results', [])
                self.cache[key] = result
                if len(self.cache) > 50:
                    self.cache.pop(next(iter(self.cache)))
                return result
            except Exception as exc:
                print(f'Semantic search unavailable: {exc}', file=sys.stderr, flush=True)
                if self.process:
                    self.process.kill()
                    self.process = None
                    self.model = None
                return []


class RouterController:
    """Runs zero-shot refresh and training outside the web-server process."""
    def __init__(self, catalog):
        self.root = Path(__file__).resolve().parent
        self.catalog = Path(catalog).resolve()
        self.python = (self.root / '..' / '..' / 'work' / 'vision-venv' /
                       'Scripts' / 'python.exe').resolve()
        self.progress_file = self.catalog / 'router-progress.json'
        self.process = None
        self.lock = threading.RLock()

    def status(self):
        with self.lock:
            try:
                payload = json.loads(self.progress_file.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                payload = {'status': 'idle', 'total': 0, 'completed': 0}
            code = self.process.poll() if self.process else None
            if self.process and code is None:
                payload['active'] = True
                if payload.get('status') not in {'running', 'preparing'}:
                    payload['status'] = 'preparing'
            else:
                payload['active'] = False
                if self.process:
                    if payload.get('status') in {'running', 'preparing'}:
                        payload['status'] = 'completed' if code == 0 else 'error'
                    payload['return_code'] = code
                    self.process = None
            return payload

    def start(self, action):
        if action not in {'bootstrap', 'train'}:
            raise ValueError('Неизвестная операция роутера')
        with self.lock:
            if self.status().get('active'):
                raise ValueError('Роутер уже занят')
            self.progress_file.unlink(missing_ok=True)
            env = os.environ.copy()
            env.update(HF_HOME=r'C:\cv-models\huggingface', HF_HUB_DISABLE_XET='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1')
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            self.process = subprocess.Popen(
                [str(self.python), str(self.root / 'router_learning.py'), action,
                 '--catalog', str(self.catalog), '--progress', str(self.progress_file)],
                cwd=self.root, env=env, creationflags=flags,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            return self.status()


class DeviceController:
    """Expose this computer as a configurable HomeCloud worker device."""
    FEATURES = ('inventory', 'faces', 'visual', 'ocr', 'caption', 'adult', 'speech', 'diarize',
               'authenticity')

    def __init__(self, catalog, device_id=None, device_name=None):
        self.root = Path(__file__).resolve().parent
        self.catalog = Path(catalog).resolve()
        self.device_id = device_id or socket.gethostname().casefold()
        self.device_name = device_name or socket.gethostname()
        self.progress_file = self.catalog / 'device-job-progress.json'
        self.stop_file = self.catalog / 'device-job-stop.request'
        self.process = None
        self.run_id = None
        self.lock = threading.RLock()
        self.history_db().close()
        self.reattach()

    def history_db(self):
        """Список источников заданий живёт рядом с каталогом, в нём же."""
        db = sqlite3.connect(self.catalog / 'catalog.sqlite', timeout=10)
        db.execute('''CREATE TABLE IF NOT EXISTS scan_runs (
            id INTEGER PRIMARY KEY,
            roots_json TEXT NOT NULL, paths_json TEXT NOT NULL,
            first_run_at TEXT NOT NULL, last_run_at TEXT NOT NULL,
            runs INTEGER NOT NULL DEFAULT 1,
            last_features TEXT NOT NULL, done_features TEXT NOT NULL DEFAULT '[]',
            last_status TEXT NOT NULL DEFAULT 'running')''')
        db.execute('CREATE UNIQUE INDEX IF NOT EXISTS scan_runs_source '
                   'ON scan_runs(roots_json,paths_json)')
        db.commit()
        return db

    def reattach(self):
        """Сервер перезапустили, а задание идёт: снова привязываем его к истории."""
        try:
            payload = json.loads(self.progress_file.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return
        if payload.get('status') not in {'preparing', 'running'}:
            return
        if not ScanController._pid_exists(payload.get('pid')):
            return
        roots_json = json.dumps(payload.get('roots', []), ensure_ascii=False)
        paths_json = json.dumps(payload.get('paths', []), ensure_ascii=False)
        db = self.history_db()
        try:
            row = db.execute('SELECT id FROM scan_runs WHERE roots_json=? AND paths_json=?',
                             (roots_json, paths_json)).fetchone()
            if not row:
                return
            self.run_id = row[0]
            with db:
                db.execute("UPDATE scan_runs SET last_status='running' WHERE id=?", (row[0],))
        finally:
            db.close()

    def remember(self, roots, paths, features):
        """Запоминаем набор источников, чтобы потом прогнать по нему другой этап."""
        now = datetime.now(timezone.utc).isoformat()
        chosen = json.dumps(sorted(name for name, on in features.items() if on))
        roots_json = json.dumps([str(item) for item in roots], ensure_ascii=False)
        paths_json = json.dumps([str(item) for item in paths], ensure_ascii=False)
        db = self.history_db()
        try:
            with db:
                db.execute("UPDATE scan_runs SET last_status='interrupted' "
                           "WHERE last_status='running'")
                db.execute('''INSERT INTO scan_runs(roots_json,paths_json,first_run_at,
                    last_run_at,runs,last_features,done_features,last_status)
                    VALUES(?,?,?,?,1,?,'[]','running')
                    ON CONFLICT(roots_json,paths_json) DO UPDATE SET
                    last_run_at=excluded.last_run_at,runs=scan_runs.runs+1,
                    last_features=excluded.last_features,last_status='running' ''',
                    (roots_json, paths_json, now, now, chosen))
            self.run_id = db.execute(
                'SELECT id FROM scan_runs WHERE roots_json=? AND paths_json=?',
                (roots_json, paths_json)).fetchone()[0]
        finally:
            db.close()

    def finish(self, status):
        """Завершённые этапы приписываем источнику: видно, что уже сделано."""
        run_id, self.run_id = self.run_id, None
        if not run_id:
            return
        db = self.history_db()
        try:
            row = db.execute('SELECT last_features,done_features FROM scan_runs WHERE id=?',
                             (run_id,)).fetchone()
            done = sorted(set(json.loads(row[1])) | set(json.loads(row[0]))) if (
                row and status == 'completed') else None
            with db:
                if done is None:
                    db.execute('UPDATE scan_runs SET last_status=? WHERE id=?', (status, run_id))
                else:
                    db.execute('UPDATE scan_runs SET last_status=?,done_features=? WHERE id=?',
                               (status, json.dumps(done), run_id))
        finally:
            db.close()

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
                    values = []
                    for item in roots:
                        values.extend((item, item.rstrip('\\/') + os.sep + '%'))
                    clause = ' OR '.join(['(path=? OR path LIKE ?)'] * len(roots))
                    photos = db.execute("SELECT COUNT(*) FROM photos WHERE status='ok' AND ("
                                        + clause + ')', values).fetchone()[0]
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

    @staticmethod
    def drives():
        if os.name != 'nt':
            return [{'path': '/', 'name': '/', 'free': shutil.disk_usage('/').free}]
        import ctypes
        mask = ctypes.windll.kernel32.GetLogicalDrives()
        result = []
        for index in range(26):
            if mask & (1 << index):
                path = f'{chr(65 + index)}:\\'
                try:
                    usage = shutil.disk_usage(path)
                    result.append({'path': path, 'name': path, 'free': usage.free,
                                   'total': usage.total})
                except OSError:
                    continue
        return result

    def info(self):
        worker_root = self.root / 'work'
        vision = worker_root / 'vision-venv' / 'Scripts' / 'python.exe'
        audio = worker_root / 'audio-venv' / 'Scripts' / 'python.exe'
        imgutils = worker_root / 'imgutils-venv' / 'Scripts' / 'python.exe'
        options = catalog_settings.load(self.catalog)
        return {
            'id': self.device_id, 'name': self.device_name,
            'hostname': socket.gethostname(), 'platform': sys.platform,
            'drives': self.drives(),
            'visual_model': options['visual_model'],
            'visual_models': catalog_settings.visual_models(),
            'capabilities': {
                'inventory': True,
                'faces': (self.root / 'models' / 'buffalo_l').is_dir(),
                'visual': vision.is_file() and Path(r'C:\cv-models\huggingface').is_dir(),
                'ocr': Path(r'C:\cv-ocr\Scripts\python.exe').is_file(),
                'caption': vision.is_file() and Path(
                    r'C:\cv-models\huggingface\hub\models--Qwen--Qwen3-VL-2B-Instruct').is_dir(),
                'adult': vision.is_file() and Path(
                    r'C:\cv-models\huggingface\hub\models--SmilingWolf--wd-eva02-large-tagger-v3').is_dir(),
                'speech': audio.is_file(),
                'diarize': (audio.is_file()
                           and (Path(r'C:\cv-models\huggingface').is_dir()
                                or (self.root / 'hf-token.txt').is_file())),
                'authenticity': imgutils.is_file(),
            },
        }

    def browse(self, raw_path=''):
        if not raw_path:
            return {'path': '', 'parent': None, 'directories': self.drives()}
        path = Path(raw_path).resolve()
        roots = [Path(item['path']).resolve() for item in self.drives()]
        if not any(path == root or root in path.parents for root in roots) or not path.is_dir():
            raise ValueError('Папка не найдена или недоступна')
        directories = []
        try:
            for child in sorted(path.iterdir(), key=lambda item: item.name.casefold()):
                try:
                    if child.is_dir() and not child.is_symlink():
                        directories.append({'path': str(child), 'name': child.name})
                except OSError:
                    continue
                if len(directories) >= 300:
                    break
        except PermissionError:
            raise ValueError('Нет доступа к папке')
        parent = str(path.parent) if path.parent != path else None
        return {'path': str(path), 'parent': parent, 'directories': directories}

    def status(self):
        with self.lock:
            try:
                payload = json.loads(self.progress_file.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                payload = {'status': 'idle', 'phase': '', 'roots': [], 'features': {}}
            return_code = self.process.poll() if self.process else None
            if self.process and return_code is not None:
                if payload.get('status') in {'preparing', 'running'}:
                    payload['status'] = 'completed' if return_code == 0 else 'error'
                payload['return_code'] = return_code
                self.process = None
            payload['active'] = payload.get('status') in {'preparing', 'running'}
            managed = self.process is not None and self.process.poll() is None
            if payload['active'] and not (managed or ScanController._pid_exists(payload.get('pid'))):
                payload['active'] = False
                payload['status'] = 'interrupted'
            payload['stop_requested'] = payload['active'] and self.stop_file.exists()
            # Задание закончилось — записываем в историю, какие этапы дошли до конца.
            if self.run_id and not payload['active']:
                self.finish(payload.get('status', 'interrupted'))
            try:
                db = sqlite3.connect(self.catalog / 'catalog.sqlite', timeout=1)
                payload['catalog_photos'] = db.execute(
                    "SELECT COUNT(*) FROM photos WHERE status='ok'").fetchone()[0]
                payload['catalog_faces'] = db.execute('SELECT COUNT(*) FROM faces').fetchone()[0]
                payload['catalog_videos'] = db.execute(
                    "SELECT COUNT(*) FROM photos WHERE status='ok' AND kind='video'").fetchone()[0]
                payload['catalog_indexed'] = db.execute(
                    "SELECT COUNT(*) FROM photo_analysis WHERE status='ok'").fetchone()[0]
                payload['catalog_ocr'] = db.execute(
                    "SELECT COUNT(*) FROM photo_analysis WHERE ocr_status='ok' AND ocr_text!=''").fetchone()[0]
                payload['catalog_captioned'] = db.execute(
                    "SELECT COUNT(*) FROM photo_analysis WHERE caption_status='ok' AND caption!=''").fetchone()[0]
                payload['catalog_adult_analyzed'] = db.execute(
                    "SELECT COUNT(*) FROM photo_adult_analysis WHERE status='ok'").fetchone()[0]
                payload['catalog_adult_flagged'] = db.execute(
                    "SELECT COUNT(*) FROM photo_adult_analysis WHERE status='ok' AND rating!='safe'").fetchone()[0]
                payload['catalog_speech'] = db.execute(
                    "SELECT COUNT(*) FROM video_speech WHERE status='ok' AND text!=''").fetchone()[0]
                payload['catalog_diarized'] = db.execute(
                    "SELECT COUNT(*) FROM video_diarization WHERE status='ok'").fetchone()[0]
                payload['catalog_authenticity'] = db.execute(
                    'SELECT COUNT(*) FROM face_authenticity').fetchone()[0]
                payload['catalog_anime_faces'] = db.execute(
                    'SELECT COUNT(*) FROM face_authenticity WHERE anime_score>=0.85').fetchone()[0]
                db.close()
            except sqlite3.Error:
                pass
            return payload

    def start(self, roots, features, paths=None, force=False, visual_model=None):
        with self.lock:
            if self.status()['active']:
                raise ValueError('На устройстве уже выполняется задание')
            resolved = []
            available_roots = [Path(item['path']).resolve() for item in self.drives()]
            for raw in roots or []:
                path = Path(str(raw)).resolve()
                if not path.is_dir() or not any(
                        path == drive or drive in path.parents for drive in available_roots):
                    raise ValueError(f'Недоступная папка: {raw}')
                if path not in resolved:
                    resolved.append(path)
            selected_paths = []
            raw_paths = list(dict.fromkeys(str(path) for path in (paths or [])))
            if len(raw_paths) > 500:
                raise ValueError('За один запуск можно выбрать не более 500 фотографий')
            if raw_paths:
                placeholders = ','.join('?' for _ in raw_paths)
                with sqlite3.connect(self.catalog / 'catalog.sqlite') as catalog_db:
                    known = {row[0] for row in catalog_db.execute(
                        f"SELECT path FROM photos WHERE status='ok' AND path IN ({placeholders})",
                        raw_paths)}
                for raw in raw_paths:
                    path = Path(raw).resolve()
                    if raw not in known or not path.is_file() or not any(
                            path == drive or drive in path.parents for drive in available_roots):
                        raise ValueError(f'Фотография недоступна: {raw}')
                    selected_paths.append(path)
            if not resolved and not selected_paths:
                raise ValueError('Выберите хотя бы один диск, папку или фотографию')
            supported = self.info()['capabilities']
            selected = {name: bool((features or {}).get(name)) for name in self.FEATURES}
            if not any(selected.values()):
                raise ValueError('Выберите хотя бы одну возможность')
            if selected['caption']:
                selected['visual'] = True
                # Подробное описание использует рейтинг и уверенные теги WD.
                selected['adult'] = bool(supported.get('adult'))
            unavailable = [name for name, enabled in selected.items()
                           if enabled and not supported.get(name)]
            if unavailable:
                raise ValueError('Недоступно на устройстве: ' + ', '.join(unavailable))
            if selected['ocr']:
                selected['visual'] = True
            if selected['visual'] and visual_model is not None:
                model = str(visual_model)
                available = {item['id']: item for item in catalog_settings.visual_models()}
                if model not in available:
                    raise ValueError('Неизвестная модель визуального индекса')
                if not available[model]['installed']:
                    raise ValueError('Модель ещё не скачана на это устройство')
                settings_db = self.history_db()
                try:
                    catalog_settings.write(settings_db, {'visual_model': model})
                finally:
                    settings_db.close()
            if selected['inventory'] and not resolved:
                raise ValueError('Опись собирается по папкам, выберите хотя бы одну')
            self.stop_file.unlink(missing_ok=True)
            self.remember(resolved, selected_paths, selected)
            # Файл прогресса пока от прошлого задания: если не перебить его
            # сейчас, первый же опрос сочтёт новое задание завершённым.
            self.progress_file.write_text(json.dumps({
                'status': 'preparing', 'phase': 'inventory',
                'roots': [str(item) for item in resolved],
                'paths': [str(item) for item in selected_paths],
                'features': selected, 'total': 0, 'completed': 0,
                'updated_at': time.time(), 'pid': os.getpid()}, ensure_ascii=False),
                encoding='utf-8')
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            command = [sys.executable, str(self.root / 'device_job.py'),
                       '--catalog', str(self.catalog), '--features', json.dumps(selected),
                       '--progress-file', str(self.progress_file), '--stop-file', str(self.stop_file)]
            if force:
                command.append('--force')
            for path in resolved:
                command.extend(('--root', str(path)))
            for path in selected_paths:
                command.extend(('--path', str(path)))
            self.process = subprocess.Popen(
                command, cwd=self.root, creationflags=flags,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return self.status()

    def stop(self):
        with self.lock:
            status = self.status()
            if not status['active']:
                raise ValueError('Активного задания нет')
            self.stop_file.write_text('stop\n', encoding='ascii')
            status['stop_requested'] = True
            return status


class DuplicateService:
    """Считает хеши в отдельном потоке: на большом каталоге это надолго."""

    def __init__(self, catalog):
        self.catalog = Path(catalog).resolve()
        self.lock = threading.Lock()
        self.thread = None
        self.stopping = False
        self.state = {'status': 'idle', 'done': 0, 'total': 0, 'errors': 0,
                      'current': '', 'similar': False, 'started_at': 0}

    def status(self):
        with self.lock:
            db = duplicates.connect(self.catalog)
            try:
                return {**self.state, **duplicates.stats(db)}
            finally:
                db.close()

    def start(self, similar=False):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return self.state
            self.stopping = False
            self.state = {'status': 'counting', 'done': 0, 'total': 0, 'errors': 0,
                          'current': '', 'similar': bool(similar),
                          'started_at': time.time()}
            self.thread = threading.Thread(target=self._run, args=(bool(similar),),
                                           daemon=True)
            self.thread.start()
            return self.state

    def stop(self):
        self.stopping = True
        return self.state

    def _run(self, similar):
        db = duplicates.connect(self.catalog)
        try:
            privacy.ensure_schema(db)
            disk = privacy.stored_paths(db)
            rows = duplicates.candidates(db, similar)
            with self.lock:
                self.state.update(status='running', total=len(rows))

            def progress(done, errors, current):
                with self.lock:
                    self.state.update(done=done, errors=errors, current=current)

            duplicates.compute(db, rows, similar, progress=progress,
                               stop=lambda: self.stopping, disk=disk)
            with self.lock:
                self.state.update(status='stopped' if self.stopping else 'completed',
                                  current='')
        except Exception as exc:
            print(f'Duplicate scan failed: {exc}', file=sys.stderr, flush=True)
            with self.lock:
                self.state.update(status='error', current=str(exc))
        finally:
            db.close()


class ReclusterController:
    """Полная пересборка автоматических групп лиц.

    Считается в фоновом потоке на СВОЁМ соединении с БД, а не на потоке
    запроса и не через общее `self.store.db` — раньше пересборка держала
    общий `App.lock` и одновременно долбила общее соединение sqlite, а
    /api/state читает то же соединение без блокировки вовсе: конкурентная
    работа с одним объектом Connection из двух потоков и была причиной
    зависания страницы и сырых ошибок sqlite в интерфейсе. Отдельное
    соединение снимает конфликт полностью — sqlite сама умеет несколько
    соединений к одному файлу.
    """
    STEPS = [
        {'key': 'prepare', 'title': 'Готовлю данные'},
        {'key': 'vectors', 'title': 'Загружаю векторы лиц'},
        {'key': 'cluster', 'title': 'Кластеризую лица'},
        {'key': 'save', 'title': 'Сохраняю результат'},
    ]

    def __init__(self, app):
        self.app = app
        self.lock = threading.Lock()
        self.thread = None
        self.stopping = False
        self.state = self._idle_state()

    @staticmethod
    def _idle_state():
        return {'status': 'idle', 'step': '', 'step_index': 0,
                'steps_total': len(ReclusterController.STEPS), 'done': 0, 'total': 0,
                'faces_total': 0, 'message': '', 'error': '',
                'started_at': 0, 'step_started_at': 0, 'updated_at': 0}

    def status(self):
        with self.lock:
            return {**self.state, 'steps': self.STEPS}

    def _set(self, **kwargs):
        with self.lock:
            self.state.update(kwargs, updated_at=time.time())

    def _should_stop(self):
        with self.lock:
            return self.stopping

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return {**self.state, 'steps': self.STEPS}
            self.stopping = False
            now = time.time()
            self.state = {**self._idle_state(), 'status': 'running',
                          'step': self.STEPS[0]['key'], 'step_index': 1,
                          'message': self.STEPS[0]['title'], 'started_at': now,
                          'step_started_at': now}
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            return {**self.state, 'steps': self.STEPS}

    def stop(self):
        with self.lock:
            if not (self.thread and self.thread.is_alive()):
                raise ValueError('Пересборка сейчас не выполняется')
            self.stopping = True
            self.state['message'] = 'Останавливаю после текущего шага…'
            return {**self.state, 'steps': self.STEPS}

    def _run(self):
        db = None
        try:
            db = open_catalog_db(self.app.store.folder, check_same_thread=False)
            rows = db.execute('SELECT id,embedding FROM faces ORDER BY id').fetchall()
            total = len(rows)
            # Старые группы не трогаем, пока не готовы полностью новые: если
            # остановить пересборку на любом из первых трёх шагов, каталог
            # остаётся ровно таким, каким был — DELETE происходит одной
            # транзакцией с первой же пачкой новых данных на шаге "Сохраняю".
            self._set(step='prepare', step_index=1, done=0, total=total, faces_total=total,
                      message='Готовлю данные')
            if self._should_stop():
                self._set(status='stopped', message='Остановлено — старые группы не тронуты')
                return
            self._set(step='vectors', step_index=2, done=0, total=total,
                      step_started_at=time.time(), message='Загружаю векторы лиц')
            ids, vectors = [], []
            for face_id, blob in rows:
                vector = np.frombuffer(blob, dtype='<f4').astype('<f4')
                norm = float(np.linalg.norm(vector))
                vectors.append(vector / max(norm, 1e-12))
                ids.append(face_id)
                if len(ids) % 2000 == 0:
                    self._set(done=len(ids))
            self._set(done=total, total=total)
            if self._should_stop():
                self._set(status='stopped', message='Остановлено — старые группы не тронуты')
                return
            self._set(step='cluster', step_index=3, done=0, total=1,
                      step_started_at=time.time(),
                      message='Кластеризую лица — самый долгий шаг, потерпите')
            min_cluster_size = self.app.store.min_cluster_size
            if vectors and len(vectors) >= min_cluster_size:
                matrix = np.stack(vectors)
                labels, probabilities = cluster_embeddings(
                    matrix, algorithm='hdbscan', min_cluster_size=min_cluster_size)
            else:
                labels = [-1] * len(ids)
                probabilities = [0.0] * len(ids)
            self._set(done=1, total=1)
            if self._should_stop():
                self._set(status='stopped', message='Остановлено после кластеризации — '
                                                     'старые группы не тронуты')
                return
            self._set(step='save', step_index=4, done=0, total=len(ids),
                      step_started_at=time.time(), message='Сохраняю результат')
            now = datetime.now(timezone.utc).isoformat()
            stopped_midway = False
            cleared = False
            chunk = 2000
            for offset in range(0, len(ids), chunk):
                batch = list(zip(ids[offset:offset + chunk], labels[offset:offset + chunk],
                                 probabilities[offset:offset + chunk]))
                with db:
                    if not cleared:
                        # Старое стирается в той же транзакции, что и первая
                        # пачка нового — снаружи каталог не бывает пустым.
                        db.execute('DELETE FROM face_clusters')
                        cleared = True
                    db.executemany(
                        'INSERT INTO face_clusters(face_id,label,probability,method,computed_at) '
                        'VALUES(?,?,?,?,?) ON CONFLICT(face_id) DO UPDATE SET label=excluded.label,'
                        'probability=excluded.probability,method=excluded.method,'
                        'computed_at=excluded.computed_at',
                        [(int(face_id), int(label), float(probability), 'hdbscan', now)
                         for face_id, label, probability in batch])
                self._set(done=min(offset + chunk, len(ids)), total=len(ids))
                if self._should_stop():
                    stopped_midway = True
                    break
            db.close()
            db = None
            with self.app.lock:
                self.app.store.reload_faces()
            if stopped_midway:
                self._set(status='stopped', message='Остановлено — часть групп уже пересобрана, '
                                                     'остальные соберутся при следующем запуске')
            else:
                self._set(step='save', step_index=len(self.STEPS), done=len(ids), total=len(ids))
                self._set(status='completed', message='Готово')
        except Exception as exc:
            print(f'Recluster failed: {exc}', file=sys.stderr, flush=True)
            self._set(status='error', error=str(exc), message='Ошибка пересборки')
        finally:
            if db is not None:
                try:
                    db.close()
                except sqlite3.Error:
                    pass


class App:
    def __init__(self, data, min_cluster_size=8, token=None,
                 device_id=None, device_name=None, max_faces=0):
        schema = analysis_database(Path(data).resolve())
        schema.close()
        self.store = CatalogStore(data, min_cluster_size, thread_safe=True,
                                  max_faces=max_faces)
        albums.ensure_schema(self.store.db)
        people_albums.ensure_schema(self.store.db)
        privacy.ensure_schema(self.store.db)
        pathrules.ensure_column(self.store.db)
        duplicates.ensure_schema(self.store.db)
        router_learning.connect(data).close()
        speaker_diarization.connect(data).close()
        speech_videos.connect(data).close()
        self.folders = albums.Folders(self.store.db)
        self.duplicates = DuplicateService(data)
        self.recluster_job = ReclusterController(self)
        self.catalog_folder = Path(data).resolve()
        self.scanner = ScanController()
        self.analyzer = AnalysisController()
        self.semantic = SemanticService(data)
        self.router = RouterController(data)
        self.device = DeviceController(data, device_id, device_name)
        self.token = token or secrets.token_urlsafe(32)
        self._centroids_stamp = None
        self._centroids_cache = {}
        self.lock = threading.RLock()

    def file_for(self, path):
        """Скрытый снимок живёт в личной папке — оттуда его и читаем."""
        row = self.store.db.execute(
            'SELECT stored FROM hidden_photos WHERE path=?', (str(path),)).fetchone()
        return Path(row[0]) if row else Path(path)

    def router_batch_archive(self, hide_adult=False):
        """Build a private offline review pack without exposing original paths."""
        reserved = router_learning.pending_batch_paths(self.catalog_folder)
        candidates = router_learning.review_queue(
            self.catalog_folder, 100, hide_adult=hide_adult)
        encoded = []
        for item in candidates:
            raw_path = item['path']
            if raw_path in reserved:
                continue
            path = self.file_for(raw_path).resolve()
            if not path.is_file():
                continue
            try:
                source_image = video_media.open_frame(path)
                try:
                    image = source_image.convert('RGB')
                    try:
                        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                        output = io.BytesIO()
                        image.save(output, 'JPEG', quality=88, optimize=True)
                        encoded.append((raw_path, output.getvalue()))
                    finally:
                        image.close()
                finally:
                    source_image.close()
            except (OSError, ValueError) as exc:
                print(f'Router batch skipped {raw_path}: {exc}', file=sys.stderr, flush=True)
                continue
            if len(encoded) == 10:
                break
        batch = router_learning.create_batch(
            self.catalog_folder, [path for path, _ in encoded])
        pictures = dict(encoded)
        public_items = [{'file': item['file']} for item in batch['items']]
        template = {'batch_id': batch['batch_id'],
                    'items': [{'file': item['file'], 'labels': []} for item in public_items]}
        manifest = {'batch_id': batch['batch_id'],
                    'embedding_model': batch['embedding_model'], 'items': public_items}
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as package:
            for item in batch['items']:
                package.writestr(item['file'], pictures[item['path']])
            package.writestr('categories.json', json.dumps(
                router_learning.batch_categories(), ensure_ascii=False, indent=2))
            package.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
            package.writestr('answer-template.json', json.dumps(
                template, ensure_ascii=False, indent=2))
            package.writestr('prompt.txt', router_learning.batch_prompt(
                batch['batch_id'], [item['file'] for item in public_items]))
        stamp = datetime.now().strftime('%Y%m%d-%H%M')
        return archive.getvalue(), f'homecloud-review-{stamp}-{batch["batch_id"][:8]}.zip'

    def masked_faces(self, viewer='', admin=False, hide_adult=False):
        """Лица, которых этому зрителю видеть не положено."""
        paths = set()
        with self.lock:
            # Спрятанное не мелькает и в лицах — ни у соседа, ни у самого владельца:
            # раздел «Люди» общий, а скрытый альбом открывают отдельно.
            paths |= {row[0] for row in self.store.db.execute(
                'SELECT path FROM hidden_photos')}
            if hide_adult:
                paths |= {row[0] for row in self.store.db.execute(
                    "SELECT path FROM photo_adult_analysis WHERE status='ok' "
                    "AND rating NOT IN ('safe','unknown','sensitive')")}
            if not paths:
                return frozenset()
            hidden = set()
            wanted = list(paths)
            for offset in range(0, len(wanted), 400):
                batch = wanted[offset:offset + 400]
                marks = ','.join('?' * len(batch))
                hidden.update(row[0] for row in self.store.db.execute(
                    f'SELECT id FROM faces WHERE path IN ({marks})', batch))
            return frozenset(hidden)

    @staticmethod
    def without(group, masked):
        """Копия группы без спрятанных лиц."""
        if not masked:
            return group
        return {**group, 'face_ids': [face_id for face_id in group['face_ids']
                                      if face_id not in masked]}

    def group_payload(self, group, include_faces=False, albums_by_key=None, hidden_keys=None):
        face_ids = group['face_ids']
        payload = {
            'key': group['key'], 'title': group['title'], 'name': group['name'],
            'bigfam_id': group.get('bigfam_id'),
            'kind': group['kind'], 'count': len(face_ids),
            'photos': len({self.store.by_id[face_id][1] for face_id in face_ids}),
            'covers': [f'/media/thumb/{face_id}' for face_id in face_ids[:4]],
            'avatar_face': group.get('avatar_face'),
            'avatar_pinned': bool(group.get('avatar_pinned')),
            'avatar': (f"/media/face-crop/{group['avatar_face']}?size=400"
                       if group.get('avatar_face') else ''),
            'albums': (albums_by_key or {}).get(group['key'], []),
            'hidden': group['key'] in (hidden_keys or ()),
        }
        if include_faces:
            payload['faces'] = [self.face_payload(face_id) for face_id in face_ids]
        return payload

    def face_payload(self, face_id):
        row = self.store.by_id[face_id]
        moment = row[4] if len(row) > 4 else None
        return {
            'id': face_id, 'filename': Path(row[1]).name, 'path': row[1],
            'kind': 'video' if video_media.is_video(row[1]) else 'photo',
            'frame_time': moment,
            'thumbnail': f'/media/thumb/{face_id}',
            'original': f'/media/original/{face_id}',
            'confidence': self.store.auto_confidence.get(face_id, 0),
        }

    def state(self, viewer='', admin=False, hide_adult=False):
        masked = self.masked_faces(viewer, admin, hide_adult)
        with self.lock:
            face_count = self.store.db.execute('SELECT COUNT(*) FROM faces').fetchone()[0]
            if face_count != len(self.store.rows) and not self.device.status().get('active'):
                self.store.reload_faces()
            else:
                # Скан мог обновить лица, не меняя их число: метки досчитываем.
                self.store.refresh_labels()
            groups = [self.without(group, masked) for group in self.store.groups()]
            groups = [group for group in groups if group['face_ids']]
            # Скрытый альбом — решение владельца картотеки, не приватность
            # снимка: обычный зритель группу не видит вовсе, админ видит и
            # управляет (group_payload помечает hidden, чтобы показать иначе).
            hidden_keys = people_albums.hidden_group_keys(self.store.db)
            if not admin:
                groups = [group for group in groups if group['key'] not in hidden_keys]
            albums_by_key = people_albums.group_albums(
                self.store.db, [group['key'] for group in groups])
            album_tree = people_albums.tree(self.store.db)
            if not admin:
                # Обычный зритель не должен даже знать о существовании
                # скрытого альбома, не то что о его составе.
                album_tree = [item for item in album_tree if not item['effectively_hidden']]
            named = [group for group in groups if group['kind'] == 'person']
            shown = {face_id for group in groups for face_id in group['face_ids']}
            return {
                'stats': {
                    'faces': len(shown),
                    'photos': self.store.db.execute(
                        "SELECT COUNT(*) FROM photos WHERE status='ok' "
                        'AND COALESCE(blocked,0)=0').fetchone()[0],
                    'videos': self.store.db.execute(
                        "SELECT COUNT(*) FROM photos WHERE status='ok' AND kind='video'"
                        ' AND COALESCE(blocked,0)=0').fetchone()[0],
                    'excluded': self.store.db.execute(
                        'SELECT COUNT(*) FROM photos WHERE COALESCE(blocked,0)=1'
                    ).fetchone()[0],
                    'with_faces': len({self.store.by_id[face_id][1] for face_id in shown}),
                    'hidden': self.store.db.execute(
                        'SELECT COUNT(*) FROM hidden_photos'
                        + ('' if admin else ' WHERE owner=?'),
                        () if admin else (viewer or '',)).fetchone()[0],
                    'people': len(named),
                    'groups': sum(group['kind'] == 'auto' for group in groups),
                    'review': sum(len(group['face_ids']) for group in groups
                                  if group['kind'] in {'noise', 'excluded'}),
                },
                'groups': [self.group_payload(group, albums_by_key=albums_by_key,
                                              hidden_keys=hidden_keys) for group in groups],
                'people_albums': album_tree,
                'people': [{'name': group['name'], 'count': len(group['face_ids']),
                            'bigfam_id': group.get('bigfam_id'),
                            'avatar': (f"/media/face-crop/{group['avatar_face']}?size=200"
                                       if group.get('avatar_face') else '')} for group in named],
            }

    # Косинус между векторами лиц: ArcFace отвечает на вопрос «тот же человек?»,
    # родство он не измеряет, поэтому высокие значения — повод объединить группы.
    VERDICTS = ((0.62, 'скорее всего один человек'), (0.50, 'очень похожи'),
                (0.38, 'заметное сходство'), (-1.0, 'мало общего'))

    @staticmethod
    def verdict(score):
        return next(text for edge, text in App.VERDICTS if score >= edge)

    @staticmethod
    def unit(vector):
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def group_vectors(self, group, limit=400):
        """Берём до `limit` самых уверенных лиц группы — этого хватает для оценки."""
        matrix, ids = self.store.vectors(group['face_ids'][:limit])
        return matrix, ids

    def centroids(self, groups):
        """Средний вектор каждой группы; пересчитывается, когда каталог менялся."""
        stamp = (len(self.store.rows), len(groups),
                 tuple(sorted((group['key'], len(group['face_ids'])) for group in groups)))
        if self._centroids_stamp == stamp:
            return self._centroids_cache
        cache = {}
        for group in groups:
            matrix, ids = self.group_vectors(group, limit=200)
            if len(ids):
                cache[group['key']] = self.unit(matrix.mean(axis=0))
        self._centroids_stamp, self._centroids_cache = stamp, cache
        return cache

    def group_summary(self, group):
        payload = self.group_payload(group)
        return {key: payload[key] for key in
                ('key', 'title', 'name', 'kind', 'count', 'photos', 'bigfam_id', 'avatar')}

    def similar_groups(self, key, limit=10, minimum=0.3):
        with self.lock:
            groups = [group for group in self.store.groups()
                      if group['kind'] in {'person', 'auto'}]
            target = next((group for group in groups if group['key'] == key), None)
            if target is None:
                raise KeyError('Группа больше не существует')
            centroids = self.centroids(groups)
            base = centroids.get(key)
            if base is None:
                return {'group': self.group_summary(target), 'similar': []}
            scored = []
            for group in groups:
                if group['key'] == key or group['key'] not in centroids:
                    continue
                score = float(base @ centroids[group['key']])
                if score >= minimum:
                    scored.append((score, group))
            scored.sort(key=lambda item: -item[0])
            similar = []
            for score, group in scored[:limit]:
                similar.append({**self.group_summary(group), 'score': round(score, 4),
                                'verdict': self.verdict(score)})
            return {'group': self.group_summary(target), 'similar': similar}

    def similar_pairs(self, limit=20, minimum=0.38, named_only=False, smallest=2):
        """Самые похожие пары групп — кандидаты на объединение."""
        with self.lock:
            groups = [group for group in self.store.groups()
                      if group['kind'] in {'person', 'auto'}
                      and len(group['face_ids']) >= smallest
                      and (not named_only or group['kind'] == 'person')]
            centroids = self.centroids(self.store.groups())
            usable = [group for group in groups if group['key'] in centroids]
            if len(usable) < 2:
                return {'pairs': []}
            matrix = np.stack([centroids[group['key']] for group in usable])
            scores = matrix @ matrix.T
            np.fill_diagonal(scores, -1.0)
            pairs = []
            for first in range(len(usable)):
                for second in range(first + 1, len(usable)):
                    score = float(scores[first][second])
                    if score >= minimum:
                        pairs.append((score, usable[first], usable[second]))
            pairs.sort(key=lambda item: -item[0])
            return {'pairs': [{'a': self.group_summary(first), 'b': self.group_summary(second),
                               'score': round(score, 4), 'verdict': self.verdict(score)}
                              for score, first, second in pairs[:limit]]}

    def compare_groups(self, first_key, second_key, samples=6):
        with self.lock:
            groups = self.store.groups()
            first = next((group for group in groups if group['key'] == first_key), None)
            second = next((group for group in groups if group['key'] == second_key), None)
            if first is None or second is None:
                raise KeyError('Группа больше не существует')
            left, left_ids = self.group_vectors(first)
            right, right_ids = self.group_vectors(second)
            if not len(left_ids) or not len(right_ids):
                raise ValueError('У группы нет векторов для сравнения')
            scores = left @ right.T
            centroid = float(self.unit(left.mean(axis=0)) @ self.unit(right.mean(axis=0)))
            best = float(scores.max())
            typical = float(np.median(scores.max(axis=1)))
            pairs = []
            used_left, used_right = set(), set()
            flat = np.dstack(np.unravel_index(np.argsort(-scores, axis=None), scores.shape))[0]
            for row, column in flat:
                if len(pairs) >= samples:
                    break
                if row in used_left or column in used_right:
                    continue
                used_left.add(row)
                used_right.add(column)
                pairs.append({'a': left_ids[int(row)], 'b': right_ids[int(column)],
                              'score': round(float(scores[row][column]), 4)})
            return {'a': self.group_summary(first), 'b': self.group_summary(second),
                    'score': round(centroid, 4), 'best': round(best, 4),
                    'typical': round(typical, 4), 'verdict': self.verdict(centroid),
                    'compared': [len(left_ids), len(right_ids)], 'pairs': pairs,
                    **self.rank_among_people(centroid, groups)}

    def rank_among_people(self, score, groups):
        """Голый косинус ничего не говорит: показываем место среди пар своего архива.

        ArcFace отвечает на вопрос «тот же человек», у разных людей он почти
        всегда около нуля. Зато сравнение с остальными парами каталога понятно:
        «ближе, чем столько-то процентов пар».
        """
        centroids = self.centroids(groups)
        named = [group['key'] for group in groups
                 if group['kind'] == 'person' and group['key'] in centroids]
        if len(named) < 3:
            return {}
        matrix = np.stack([centroids[key] for key in named])
        scores = matrix @ matrix.T
        upper = scores[np.triu_indices(len(named), k=1)]
        if not upper.size:
            return {}
        return {'rank': round(float((upper < score).mean()), 3), 'among': int(upper.size),
                'typical_pair': round(float(np.median(upper)), 4)}

    def get_group(self, key, viewer='', admin=False, hide_adult=False):
        masked = self.masked_faces(viewer, admin, hide_adult)
        with self.lock:
            group = next((group for group in self.store.groups() if group['key'] == key), None)
            if group is None:
                raise KeyError('Группа больше не существует')
            return self.group_payload(self.without(group, masked), include_faces=True)

    # Колонки снимка: один и тот же набор для всех выборок.
    PHOTO_COLUMNS = (
        'photos.path,MIN(faces.id),photo_analysis.content_type,'
        'photo_analysis.blur_score,photo_analysis.caption,photo_analysis.ocr_text,'
        'photo_analysis.ocr_status,photo_analysis.caption_status,'
        'photo_adult_analysis.rating,photo_adult_analysis.adult_score,'
        'photo_adult_analysis.tags_json,photo_adult_analysis.regions_json,'
        'photo_adult_analysis.description,photos.modified,photos.kind,photos.duration,'
        'photo_analysis.caption_short,photo_analysis.caption_search,'
        'photo_analysis.caption_tags_json,photo_analysis.caption_json,'
        'video_speech.text,photos.size,photo_analysis.width,photo_analysis.height')
    PHOTO_SOURCE = (
        'FROM photos LEFT JOIN faces ON faces.path=photos.path '
        'LEFT JOIN photo_analysis ON photo_analysis.path=photos.path '
        'LEFT JOIN photo_adult_analysis ON photo_adult_analysis.path=photos.path '
        'LEFT JOIN video_speech ON video_speech.path=photos.path '
        "WHERE photos.status='ok'")
    # Для счёта и поиска лица не нужны, а без их соединения запрос в разы быстрее.
    LIGHT_SOURCE = (
        'FROM photos LEFT JOIN photo_analysis ON photo_analysis.path=photos.path '
        'LEFT JOIN photo_adult_analysis ON photo_adult_analysis.path=photos.path '
        'LEFT JOIN video_speech ON video_speech.path=photos.path '
        "WHERE photos.status='ok'")
    LIGHT_COLUMNS = ('photos.path,photo_analysis.content_type,photo_analysis.caption,'
                     'photo_analysis.ocr_text,photo_adult_analysis.tags_json,'
                     'photo_adult_analysis.description,photos.modified,'
                     'photo_analysis.caption_short,photo_analysis.caption_search,'
                     'photo_analysis.caption_tags_json,video_speech.text')

    def photo_filters(self, exact_path='', content_type='', blurry=False, adult=False,
                      folder='', folder_deep=True, album=0, album_deep=True, kind='',
                      viewer='', admin=False, hidden=False, wanted=()):
        """Фильтры галереи прямо в SQL — иначе пришлось бы тянуть весь каталог."""
        clause, arguments = privacy.where(viewer, admin, hidden)
        # Исключённые правилами пути не показываем никому и нигде.
        where, values = clause + pathrules.sql(), list(arguments)
        if wanted:
            marks = ','.join('?' * len(wanted))
            where += f' AND photos.path IN ({marks})'
            values.extend(wanted)
        if kind in {'photo', 'video'}:
            # Старые записи каталога без вида считаем фотографиями.
            where += (" AND COALESCE(photos.kind,'photo')=?")
            values.append(kind)
        if folder:
            clause, arguments = albums.folder_clause(folder, folder_deep)
            where += clause
            values.extend(arguments)
        if album:
            ids = albums.descendants(self.store.db, album) if album_deep else [int(album)]
            marks = ','.join('?' * len(ids))
            where += (f' AND photos.path IN (SELECT path FROM album_photos '
                      f'WHERE album_id IN ({marks}))')
            values.extend(ids)
        if exact_path:
            where += ' AND photos.path=?'
            values.append(exact_path)
        if content_type:
            wanted = (('graphics', 'game', 'screenshot', 'meme')
                      if content_type == 'graphics' else (content_type,))
            where += f' AND photo_analysis.content_type IN ({",".join("?" * len(wanted))})'
            values.extend(wanted)
        if blurry:
            where += ' AND photo_analysis.blur_score IS NOT NULL AND photo_analysis.blur_score < 65'
        if adult:
            # «sensitive» — это «на грани», в подборку 18+ такое не тянем.
            where += (" AND photo_adult_analysis.rating IS NOT NULL"
                      " AND photo_adult_analysis.rating NOT IN ('safe','unknown','sensitive')")
        return where, values

    def hydrate(self, paths):
        """Полные строки для готового списка путей, порядок сохраняется."""
        if not paths:
            return []
        found = {}
        for offset in range(0, len(paths), 400):
            batch = paths[offset:offset + 400]
            marks = ','.join('?' * len(batch))
            for row in self.store.db.execute(
                    f'SELECT {self.PHOTO_COLUMNS} {self.PHOTO_SOURCE} '
                    f'AND photos.path IN ({marks}) GROUP BY photos.path', batch):
                found[row[0]] = row
        return [found[path] for path in paths if path in found]

    def photo_payloads(self, names=None, query='', content_type='', blurry=False,
                       exact_path='', adult=False, limit=200, offset=0,
                       folder='', folder_deep=True, album=0, album_deep=True, kind='',
                       viewer='', admin=False, hidden=False, wanted=()):
        """Возвращает страницу снимков и общее число подходящих."""
        names = [name for name in (names or []) if name]
        query = query.casefold().strip()
        limit = min(max(int(limit or 0), 1), 500)
        offset = max(int(offset or 0), 0)
        with self.lock:
            if names or query:
                # Поиск и фильтр по людям считаются в памяти: там нужен порядок выдачи.
                where, values = self.photo_filters(
                    exact_path, content_type, blurry, adult,
                    folder, folder_deep, album, album_deep, kind,
                    viewer, admin, hidden, wanted)
                if names:
                    placeholders = ','.join('?' for _ in names)
                    rows = self.store.db.execute(
                        f'SELECT {self.PHOTO_COLUMNS.replace("photos.path,", "faces.path,")} '
                        f'FROM faces JOIN photos ON photos.path=faces.path '
                        f'JOIN face_people ON face_people.face_id=faces.id '
                        f'JOIN people ON people.id=face_people.person_id '
                        f'LEFT JOIN photo_analysis ON photo_analysis.path=faces.path '
                        f'LEFT JOIN photo_adult_analysis ON photo_adult_analysis.path=faces.path '
                        f'LEFT JOIN video_speech ON video_speech.path=faces.path '
                        f'WHERE people.name IN ({placeholders}){where} GROUP BY faces.path '
                        f'HAVING COUNT(DISTINCT people.name)=?',
                        [*names, *values, len(names)]).fetchall()
                else:
                    rows = self.store.db.execute(
                        f'SELECT {self.LIGHT_COLUMNS} {self.LIGHT_SOURCE}{where}',
                        values).fetchall()
                if query:
                    # Совпадение по тексту ищем по лёгким колонкам, без картинок и векторов.
                    index = ((0, 1, 2, 3, 4, 5, 7, 8, 9, 10) if not names
                             else (0, 2, 4, 5, 10, 12, 16, 17, 18, 20))
                    textual = [row for row in rows if query in ' '.join(
                        str(row[position] or '') for position in index).casefold()]
                    semantic = []
                    if len(query) >= 3 and self.store.db.execute(
                            'SELECT 1 FROM photo_analysis WHERE embedding IS NOT NULL '
                            'LIMIT 1').fetchone():
                        rank = {item['path']: position for position, item in enumerate(
                            self.semantic.query(query, 500))}
                        semantic = sorted((row for row in rows if row[0] in rank),
                                          key=lambda row: rank[row[0]])
                    seen = set()
                    rows = [row for row in [*textual, *semantic]
                            if not (row[0] in seen or seen.add(row[0]))]
                elif names:
                    rows.sort(key=lambda row: (-(row[13] or 0), row[0]))
                else:
                    rows.sort(key=lambda row: (-(row[6] or 0), row[0]))
                total = len(rows)
                rows = rows[offset:offset + limit]
                if not names:
                    # Лёгкие строки годятся для отбора, но не для выдачи — добираем целиком.
                    rows = self.hydrate([row[0] for row in rows])
            else:
                where, values = self.photo_filters(
                    exact_path, content_type, blurry, adult,
                    folder, folder_deep, album, album_deep, kind,
                    viewer, admin, hidden, wanted)
                total = self.store.db.execute(
                    f'SELECT COUNT(*) {self.LIGHT_SOURCE}{where}', values).fetchone()[0]
                # Свежие сверху — как в галерее телефона.
                rows = self.store.db.execute(
                    f'SELECT {self.PHOTO_COLUMNS} {self.PHOTO_SOURCE}{where} '
                    'GROUP BY photos.path ORDER BY photos.modified DESC, photos.path '
                    'LIMIT ? OFFSET ?', [*values, limit, offset]).fetchall()
        paths = [row[0] for row in rows]
        with self.lock:
            album_by_path = albums.photo_albums(self.store.db, paths)
            hidden_by_path = {}
            for offset in range(0, len(paths), 400):
                batch = paths[offset:offset + 400]
                marks = ','.join('?' * len(batch))
                hidden_by_path.update(self.store.db.execute(
                    f'SELECT path,owner FROM hidden_photos WHERE path IN ({marks})', batch))
        people_by_path = {path: [] for path in paths}
        face_counts = {path: 0 for path in paths}
        faces_by_path = {path: [] for path in paths}
        if paths:
            placeholders = ','.join('?' for _ in paths)
            with self.lock:
                for path, count in self.store.db.execute(
                        f'SELECT path,COUNT(*) FROM faces WHERE path IN ({placeholders}) GROUP BY path',
                        paths):
                    face_counts[path] = count
                for path, name, bigfam_id in self.store.db.execute(
                        f'SELECT faces.path,people.name,people.bigfam_id FROM faces '
                        f'JOIN face_people ON face_people.face_id=faces.id '
                        f'JOIN people ON people.id=face_people.person_id '
                        f'WHERE faces.path IN ({placeholders}) GROUP BY faces.path,people.id '
                        f'ORDER BY people.name', paths):
                    people_by_path[path].append({'name': name, 'bigfam_id': bigfam_id})
                # Каждое отдельное лицо — чтобы просмотрщик мог назвать и
                # неназванные лица прямо на месте, а не только через группу.
                # Группа — та же самая, что и в разделе «Люди»: по имени, если
                # оно есть, иначе по автокластеру (face_clusters), а не по
                # случайному совпадению строки имени или отдельной карточке.
                for face_id, path, moment, person_id, name, bigfam_id in self.store.db.execute(
                        f'SELECT faces.id,faces.path,faces.frame_time,face_people.person_id,'
                        f'people.name,people.bigfam_id FROM faces '
                        f'LEFT JOIN face_people ON face_people.face_id=faces.id '
                        f'LEFT JOIN people ON people.id=face_people.person_id '
                        f'WHERE faces.path IN ({placeholders}) ORDER BY faces.id', paths):
                    label = self.store.auto_labels.get(face_id, -1)
                    group = (f'person:{person_id}' if person_id is not None
                            else f'auto:{label}' if label != -1 else f'noise:{face_id}')
                    faces_by_path[path].append({
                        'id': face_id, 'thumbnail': f'/media/thumb/{face_id}',
                        'frame_time': moment, 'name': name, 'bigfam_id': bigfam_id,
                        'group': group})
        # Multi-label роутер показываем рядом с описанием. Ручная разметка имеет
        # приоритет: подтверждённый отрицательный тег не должен всплывать из AI.
        router_by_path = {path: [] for path in paths}
        if paths:
            with self.lock:
                embedding_model = catalog_settings.read(self.store.db)['visual_model']
                source, version = router_learning.prediction_source(
                    self.store.db, embedding_model)
                scores = {}
                verified = {}
                seen_labels = {path: set() for path in paths}
                for start in range(0, len(paths), 400):
                    batch = paths[start:start + 400]
                    marks = ','.join('?' * len(batch))
                    for raw, label, score in self.store.db.execute(
                            f'SELECT path,label,score FROM router_predictions '
                            f'WHERE source=? AND model_version=? AND path IN ({marks})',
                            [source, version, *batch]):
                        scores[(raw, label)] = float(score)
                        seen_labels[raw].add(label)
                    for raw, label, value in self.store.db.execute(
                            f'SELECT path,label,value FROM router_training_labels '
                            f'WHERE path IN ({marks})', batch):
                        verified[(raw, label)] = bool(value)
                        seen_labels[raw].add(label)
                titles = {item['id']: item for item in router_learning.label_payload()}
                for raw in paths:
                    for label in seen_labels[raw]:
                        manual = verified.get((raw, label))
                        if manual is False:
                            continue
                        item = titles.get(label, {'title': label, 'group': 'Другое'})
                        router_by_path[raw].append({
                            'id': label, 'title': item['title'], 'group': item['group'],
                            'score': round(scores.get((raw, label), 0), 4),
                            'verified': manual is True,
                        })
                    router_by_path[raw].sort(
                        key=lambda item: (not item['verified'], -item['score'], item['title']))
        return [{
            'path': path, 'filename': Path(path).name, 'folder': str(Path(path).parent),
            'preview': f'/media/photo?path={quote(path, safe="")}&v={round((modified or 0) / 1e6)}',
            'face_id': face_id,
            'content_type': content_type, 'blur_score': blur_score,
            'caption': caption or '', 'ocr_text': ocr_text or '',
            'caption_short': caption_short or '', 'caption_search': caption_search or '',
            'caption_tags': json.loads(caption_tags or '[]'),
            'caption_data': json.loads(caption_json or '{}'),
            'ocr_status': ocr_status or '', 'caption_status': caption_status or '',
            'adult_rating': adult_rating or 'unknown', 'adult_score': adult_score or 0,
            'adult_tags': json.loads(adult_tags or '{}'),
            'adult_regions': json.loads(adult_regions or '[]'),
            'adult_description': adult_description or '',
            'people': people_by_path.get(path, []), 'face_count': face_counts.get(path, 0),
            'faces': faces_by_path.get(path, []),
            'router_labels': router_by_path.get(path, []),
            'albums': album_by_path.get(path, []),
            'kind': kind or 'photo', 'duration': duration or 0,
            'speech_text': speech_text or '',
            'hidden_owner': hidden_by_path.get(path, ''),
            'video': (f'/media/video?path={quote(path, safe="")}' if kind == 'video' else ''),
            # Время файла в миллисекундах: галерее нужна дата снимка.
            'taken': round((modified or 0) / 1e6) or None,
            'size': size or 0, 'width': width or 0, 'height': height or 0,
        } for path, face_id, content_type, blur_score, caption, ocr_text,
              ocr_status, caption_status, adult_rating, adult_score, adult_tags,
              adult_regions, adult_description, modified, kind, duration,
              caption_short, caption_search, caption_tags, caption_json,
              speech_text, size, width, height in rows], total

    # Сколько копий показываем в группе: остальные считаются, но не рисуются.
    CARDS = 12

    def duplicate_groups(self, similar=False, limit=60, offset=0, viewer='', admin=False):
        """Группы копий с карточками снимков и подсказкой, какой оставить."""
        with self.lock:
            # Чужое спрятанное в дубликаты не показываем: это тот же каталог.
            skip = {row[0] for row in self.store.db.execute(
                'SELECT path FROM hidden_photos' + ('' if admin else ' WHERE owner<>?'),
                () if admin else (viewer or '',))}
            found, total = duplicates.groups(
                self.store.db, similar=similar, limit=limit, offset=offset, skip=skip)
            shapes = {}
            paths = [path for group in found for path in group['paths']]
            # Карточки готовим только для видимой части группы: копий одной
            # иконки бывают сотни, а показываем мы дюжину.
            shown = [path for group in found for path in group['paths'][:self.CARDS]]
            for start in range(0, len(paths), 400):
                batch = paths[start:start + 400]
                marks = ','.join('?' * len(batch))
                shapes.update({row[0]: row[1:] for row in self.store.db.execute(
                    f'SELECT path,size,width,height FROM photo_hashes '
                    f'WHERE path IN ({marks})', batch)})
        cards = {item['path']: item for item in self.hydrate_payloads(shown)}
        result = []
        for group in found:
            items = []
            for path in group['paths'][:self.CARDS]:
                card = cards.get(path)
                if card is None:
                    continue
                size, width, height = shapes.get(path, (0, 0, 0))
                items.append({**card, 'size': size or 0,
                              'width': width or 0, 'height': height or 0})
            if len(items) < 2:
                continue
            # Оставляем самый крупный кадр, при равенстве — самый тяжёлый файл.
            best = max(items, key=lambda item: ((item['width'] or 0) * (item['height'] or 0),
                                                item['size'], -len(item['path'])))
            weight = sum((shapes.get(path, (0,))[0] or 0) for path in group['paths']
                         if path != best['path'])
            result.append({'key': group['key'], 'kind': group['kind'],
                           'keep': best['path'], 'photos': items,
                           'paths': group['paths'], 'count': len(group['paths']),
                           'extra': weight})
        return {'groups': result, 'total': total, 'offset': offset, 'limit': limit}

    def hydrate_payloads(self, paths):
        """Карточки снимков для готового списка путей, одним запросом на порцию."""
        cards = []
        for start in range(0, len(paths), 200):
            batch = paths[start:start + 200]
            found, _ = self.photo_payloads(limit=len(batch), hidden=None, wanted=batch)
            cards.extend(found)
        return cards

    def drop_previews(self, raw_path, stamp):
        """Уменьшенные копии удалённого снимка больше не нужны."""
        if not stamp:
            return
        folder = self.store.folder / 'previews'
        for size in (0, 240, 400, 640, 960, 1440, 2200):
            for blur in ('', 'explicit', 'regions', 'full'):
                key = hashlib.sha1(
                    f'{Path(raw_path).resolve()}|{stamp}|{size}|{blur}'.encode('utf-8')).hexdigest()
                (folder / key[:2] / f'{key}.jpg').unlink(missing_ok=True)

    def delete_photos(self, paths):
        from send2trash import send2trash
        paths = list(dict.fromkeys(str(path) for path in (paths or [])))
        if not paths or len(paths) > 500:
            raise ValueError('Выберите от 1 до 500 фотографий')
        placeholders = ','.join('?' for _ in paths)
        with self.lock:
            known = {row[0] for row in self.store.db.execute(
                f"SELECT path FROM photos WHERE status='ok' AND path IN ({placeholders})", paths)}
            stored = dict(self.store.db.execute(
                f'SELECT path,stored FROM hidden_photos WHERE path IN ({placeholders})', paths))
            thumbnails = {path: [row[0] for row in self.store.db.execute(
                'SELECT thumbnail FROM faces WHERE path=?', (path,))] for path in known}
            stamps = dict(self.store.db.execute(
                f'SELECT path,modified FROM photos WHERE path IN ({placeholders})', paths))
            deleted, errors = [], []
            for raw in paths:
                if raw not in known:
                    errors.append({'path': raw, 'error': 'Фотография не найдена в каталоге'})
                    continue
                try:
                    # Скрытый снимок лежит в личной папке, а не по исходному пути.
                    send2trash(stored.get(raw, raw))
                    deleted.append(raw)
                except OSError as exc:
                    errors.append({'path': raw, 'error': str(exc)})
            if deleted:
                deleted_marks = ','.join('?' for _ in deleted)
                with self.store.db:
                    self.store.db.execute(
                        f'DELETE FROM face_people WHERE face_id IN '
                        f'(SELECT id FROM faces WHERE path IN ({deleted_marks}))', deleted)
                    self.store.db.execute(
                        f'DELETE FROM face_exclusions WHERE face_id IN '
                        f'(SELECT id FROM faces WHERE path IN ({deleted_marks}))', deleted)
                    self.store.db.execute(
                        f'DELETE FROM faces WHERE path IN ({deleted_marks})', deleted)
                    self.store.db.execute(
                        f'DELETE FROM photo_analysis WHERE path IN ({deleted_marks})', deleted)
                    self.store.db.execute(
                        f'DELETE FROM photos WHERE path IN ({deleted_marks})', deleted)
                privacy.forget(self.store.db, deleted)
                duplicates.forget(self.store.db, deleted)
                for raw in deleted:
                    self.drop_previews(raw, stamps.get(raw))
                    for thumbnail in thumbnails.get(raw, []):
                        try:
                            (self.store.folder / thumbnail).unlink(missing_ok=True)
                        except OSError:
                            pass
                self.store.reload_faces()
            return {'deleted': len(deleted), 'errors': errors}

    def resolve_groups(self, keys):
        wanted = set(keys)
        return [face_id for group in self.store.groups() if group['key'] in wanted
                for face_id in group['face_ids']]


class Handler(BaseHTTPRequestHandler):
    server_version = 'LocalFaces/1.0'

    @property
    def app(self):
        return self.server.app

    @property
    def viewer(self):
        """Логин и роль подставляет фронт на Proxmox: он держит сессии картотеки."""
        return (unquote(self.headers.get('X-HomeCloud-User', '')).strip()[:120],
                self.headers.get('X-HomeCloud-Role', '').strip().lower() == 'admin')

    def log_message(self, format, *args):
        if self.path.split('?', 1)[0] in {'/api/scan', '/api/analysis'}:
            return
        print(f'{self.address_string()} - {format % args}', flush=True)

    def allowed_host(self):
        host = self.headers.get('Host', '').split(':', 1)[0].lower()
        if host in {'127.0.0.1', 'localhost', socket.gethostname().lower()}:
            return True
        try:
            address = ipaddress.ip_address(host)
            return address.is_private or address.is_loopback
        except ValueError:
            return False

    def send_headers(self, status, content_type, length=None, cache='no-store'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', cache)
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        if content_type.startswith('text/html'):
            self.send_header('Content-Security-Policy',
                             "default-src 'self'; img-src 'self' data:; script-src 'self'; "
                             "style-src 'self'; connect-src 'self'; object-src 'none'; "
                             "frame-ancestors 'none'; base-uri 'none'")
        if length is not None:
            self.send_header('Content-Length', str(length))
        self.end_headers()

    def json_response(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self.send_headers(status, 'application/json; charset=utf-8', len(body))
        self.wfile.write(body)

    def download_response(self, body, filename, content_type='application/octet-stream'):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def error_json(self, status, message):
        self.json_response({'error': message}, status)

    def do_GET(self):
        if not self.allowed_host():
            return self.error_json(403, 'Недопустимый Host')
        parsed = urlparse(self.path)
        try:
            if parsed.path == '/api/state':
                viewer, admin = self.viewer
                return self.json_response(self.app.state(
                    viewer, admin,
                    parse_qs(parsed.query).get('adult', [''])[0] == 'hide'))
            if parsed.path == '/api/scan':
                return self.json_response(self.app.scanner.status())
            if parsed.path == '/api/analysis':
                return self.json_response(self.app.analyzer.status())
            if parsed.path == '/api/router/status':
                return self.json_response(self.app.router.status())
            if parsed.path == '/api/router/summary':
                return self.json_response(router_learning.summary(self.app.catalog_folder))
            if parsed.path == '/api/router/review':
                query = parse_qs(parsed.query)
                queue = router_learning.review_queue(
                    self.app.catalog_folder, int(query.get('limit', ['24'])[0]),
                    query.get('adult', [''])[0] == 'hide')
                cards = {item['path']: item for item in self.app.hydrate_payloads(
                    [item['path'] for item in queue])}
                return self.json_response({'photos': [
                    {**cards[item['path']], 'router_scores': item['scores'],
                     'router_uncertainty': item['uncertainty']}
                    for item in queue if item['path'] in cards]})
            if parsed.path == '/api/router/export':
                query = parse_qs(parsed.query)
                body, filename = self.app.router_batch_archive(
                    query.get('adult', [''])[0] == 'hide')
                return self.download_response(body, filename, 'application/zip')
            if parsed.path == '/api/device':
                return self.json_response(self.app.device.info())
            if parsed.path == '/api/device/job':
                return self.json_response(self.app.device.status())
            if parsed.path == '/api/device/tree':
                db = catalog_index.connect(self.app.device.catalog)
                try:
                    return self.json_response(catalog_index.tree(
                        db, parse_qs(parsed.query).get('path', [''])[0]))
                finally:
                    db.close()
            if parsed.path == '/api/device/history':
                return self.json_response(self.app.device.history())
            if parsed.path == '/api/device/browse':
                path = parse_qs(parsed.query).get('path', [''])[0]
                return self.json_response(self.app.device.browse(path))
            if parsed.path == '/api/group':
                query = parse_qs(parsed.query)
                viewer, admin = self.viewer
                return self.json_response(self.app.get_group(
                    query.get('key', [''])[0], viewer, admin,
                    query.get('adult', [''])[0] == 'hide'))
            if parsed.path == '/api/similar':
                query = parse_qs(parsed.query)
                return self.json_response(self.app.similar_groups(
                    query.get('key', [''])[0], int(query.get('limit', ['10'])[0])))
            if parsed.path == '/api/similar-pairs':
                query = parse_qs(parsed.query)
                return self.json_response(self.app.similar_pairs(
                    int(query.get('limit', ['20'])[0]),
                    float(query.get('min', ['0.38'])[0]),
                    query.get('named', ['0'])[0] == '1'))
            if parsed.path == '/api/compare':
                query = parse_qs(parsed.query)
                return self.json_response(self.app.compare_groups(
                    query.get('a', [''])[0], query.get('b', [''])[0]))
            if parsed.path == '/api/folders':
                query = parse_qs(parsed.query)
                path = query.get('path', [''])[0]
                with self.app.lock:
                    return self.json_response({
                        'path': path, 'trail': self.app.folders.trail(path),
                        'folders': self.app.folders.children(path)})
            if parsed.path == '/api/recluster/status':
                return self.json_response(self.app.recluster_job.status())
            if parsed.path == '/api/duplicates/status':
                return self.json_response(self.app.duplicates.status())
            if parsed.path == '/api/duplicates':
                query = parse_qs(parsed.query)
                viewer, admin = self.viewer
                return self.json_response(self.app.duplicate_groups(
                    similar=query.get('similar', ['0'])[0] == '1',
                    limit=int(query.get('limit', ['60'])[0] or 60),
                    offset=int(query.get('offset', ['0'])[0] or 0),
                    viewer=viewer, admin=admin))
            if parsed.path == '/api/settings':
                with self.app.lock:
                    return self.json_response({
                        'settings': catalog_settings.read(self.app.store.db),
                        'defaults': catalog_settings.DEFAULTS,
                        'visual_models': catalog_settings.visual_models()})
            if parsed.path == '/api/albums':
                with self.app.lock:
                    return self.json_response({'albums': albums.tree(self.app.store.db)})
            if parsed.path == '/api/photos':
                query = parse_qs(parsed.query)
                viewer, admin = self.viewer
                names = query.get('person', [])
                limit = int(query.get('limit', ['200'])[0] or 200)
                offset = int(query.get('offset', ['0'])[0] or 0)
                photos, total = self.app.photo_payloads(
                    names, query.get('q', [''])[0], query.get('type', [''])[0],
                    query.get('blurry', ['0'])[0] == '1',
                    adult=query.get('adult', ['0'])[0] == '1', limit=limit, offset=offset,
                    folder=query.get('folder', [''])[0],
                    folder_deep=query.get('folder_deep', ['1'])[0] == '1',
                    album=int(query.get('album', ['0'])[0] or 0),
                    album_deep=query.get('album_deep', ['1'])[0] == '1',
                    kind=query.get('kind', [''])[0],
                    viewer=viewer, admin=admin,
                    hidden=query.get('hidden', ['0'])[0] == '1')
                return self.json_response({'photos': photos, 'total': total,
                                           'offset': offset, 'limit': limit})
            if parsed.path == '/api/photo':
                query = parse_qs(parsed.query)
                path = query.get('path', [''])[0]
                viewer, admin = self.viewer
                photos, _ = self.app.photo_payloads(
                    exact_path=path, viewer=viewer, admin=admin,
                    hidden=query.get('hidden', ['0'])[0] == '1')
                if not photos:
                    return self.error_json(404, 'Фотография не найдена')
                return self.json_response({'photo': photos[0]})
            if parsed.path == '/api/speech':
                query = parse_qs(parsed.query)
                raw_path = query.get('path', [''])[0]
                viewer, admin = self.viewer
                photos, _ = self.app.photo_payloads(
                    exact_path=raw_path, viewer=viewer, admin=admin,
                    hidden=query.get('hidden', ['0'])[0] == '1')
                if not photos:
                    return self.error_json(404, 'Файл не найден')
                with self.app.lock:
                    row = self.app.store.db.execute(
                        'SELECT language,status,model FROM video_speech WHERE path=?',
                        (raw_path,)).fetchone()
                    segments = self.app.store.db.execute(
                        'SELECT start,stop,text,speaker FROM video_speech_segments '
                        'WHERE path=? ORDER BY ord', (raw_path,)).fetchall()
                    speakers = self.app.store.db.execute(
                        'SELECT speaker,seconds FROM video_speakers WHERE path=?',
                        (raw_path,)).fetchall()
                    # Кто говорит — по имени, если получилось узнать. Порядок
                    # доверия: назначено руками (решение человека) → лицо в
                    # кадре (надёжная связка) → похожий голос, встречавшийся у
                    # названного человека раньше (это уже подсказка, не факт).
                    people = {}
                    for speaker, person_id, name in self.app.store.db.execute(
                            'SELECT video_speakers.speaker,people.id,people.name '
                            'FROM video_speakers JOIN people '
                            'ON people.id=video_speakers.assigned_person_id '
                            'WHERE video_speakers.path=?', (raw_path,)):
                        people[speaker] = {'name': name, 'confidence': 1.0, 'source': 'manual'}
                    for speaker, name, confidence in self.app.store.db.execute(
                            'SELECT video_speaker_faces.speaker,people.name,'
                            'video_speaker_faces.confidence FROM video_speaker_faces '
                            'JOIN faces ON faces.id=video_speaker_faces.face_id '
                            'JOIN face_people ON face_people.face_id=faces.id '
                            'JOIN people ON people.id=face_people.person_id '
                            'WHERE video_speaker_faces.path=?', (raw_path,)):
                        if speaker in people:
                            continue  # ручное назначение уже решило вопрос
                        people[speaker] = {'name': name, 'confidence': confidence,
                                           'source': 'face'}
                    for speaker, person_id, confidence in self.app.store.db.execute(
                            'SELECT speaker,suggested_person_id,suggested_confidence '
                            'FROM video_speakers WHERE path=? AND suggested_person_id IS NOT NULL',
                            (raw_path,)):
                        if speaker in people:
                            continue  # лицо или ручное назначение надёжнее догадки
                        name = self.app.store.db.execute(
                            'SELECT name FROM people WHERE id=?', (person_id,)).fetchone()
                        if name:
                            people[speaker] = {'name': name[0], 'confidence': confidence,
                                               'source': 'voice'}
                return self.json_response({
                    'status': row[1] if row else '', 'language': (row and row[0]) or '',
                    'model': (row and row[2]) or '', 'speakers': len(speakers),
                    'segments': [{'start': start, 'stop': stop, 'text': text,
                                  'speaker': speaker, 'person': people.get(speaker)}
                                 for start, stop, text, speaker in segments]})
            if parsed.path.startswith('/media/thumb/'):
                return self.send_media(parsed.path.rsplit('/', 1)[-1], original=False)
            if parsed.path.startswith('/media/original/'):
                return self.send_media(parsed.path.rsplit('/', 1)[-1], original=True)
            if parsed.path.startswith('/media/face-crop/'):
                query = parse_qs(parsed.query)
                return self.send_face_crop(
                    parsed.path.rsplit('/', 1)[-1],
                    float(query.get('margin', ['0.5'])[0]),
                    int(query.get('size', ['640'])[0]))
            if parsed.path == '/media/video':
                query = parse_qs(parsed.query)
                return self.send_video(query.get('path', [''])[0])
            if parsed.path == '/media/photo':
                query = parse_qs(parsed.query)
                path = query.get('path', [''])[0]
                return self.send_photo(path, query.get('blur', [''])[0],
                                       int(query.get('size', ['0'])[0] or 0),
                                       float(query.get('t', ['0'])[0] or 0))
            return self.send_static(parsed.path)
        except (KeyError, ValueError) as exc:
            return self.error_json(404, str(exc))
        except Exception as exc:
            print(f'GET error: {exc}', file=sys.stderr, flush=True)
            return self.error_json(500, 'Внутренняя ошибка локального сервера')

    def do_POST(self):
        if not self.allowed_host():
            return self.error_json(403, 'Недопустимый Host')
        origin = self.headers.get('Origin')
        expected = f'http://{self.headers.get("Host")}'
        if origin and origin != expected:
            return self.error_json(403, 'Недопустимый Origin')
        if not secrets.compare_digest(self.headers.get('X-Local-Token', ''), self.app.token):
            return self.error_json(403, 'Неверный локальный токен')
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length > 1024 * 1024:
                return self.error_json(413, 'Запрос слишком большой')
            body = json.loads(self.rfile.read(length) or b'{}')
            path = urlparse(self.path).path
            if path == '/api/scan/start':
                return self.json_response({'ok': True, 'scan': self.app.scanner.start()})
            if path == '/api/scan/stop':
                return self.json_response({'ok': True, 'scan': self.app.scanner.stop()})
            if path == '/api/analysis/start':
                return self.json_response({'ok': True, 'analysis': self.app.analyzer.start()})
            if path == '/api/analysis/stop':
                return self.json_response({'ok': True, 'analysis': self.app.analyzer.stop()})
            if path == '/api/router/bootstrap':
                return self.json_response({'ok': True, 'job': self.app.router.start('bootstrap')})
            if path == '/api/router/train':
                return self.json_response({'ok': True, 'job': self.app.router.start('train')})
            if path == '/api/router/label':
                viewer, _ = self.viewer
                labels = router_learning.save_review(
                    self.app.catalog_folder, body.get('path', ''),
                    body.get('labels', {}), viewer)
                similar = []
                if body.get('propagate'):
                    similar = router_learning.save_similar(
                        self.app.catalog_folder, body.get('path', ''), labels, viewer)
                overview = router_learning.summary(self.app.catalog_folder)
                auto_started = False
                if (overview['auto_train'] and overview['reviewed'] >= 12
                        and overview['new_since_training'] >= overview['auto_train_every']
                        and not self.app.router.status().get('active')):
                    self.app.router.start('train')
                    auto_started = True
                return self.json_response({'ok': True, 'labels': labels,
                                           'similar': similar, 'auto_started': auto_started})
            if path == '/api/router/import':
                viewer, _ = self.viewer
                result = router_learning.import_batch(
                    self.app.catalog_folder, body, viewer)
                overview = router_learning.summary(self.app.catalog_folder)
                auto_started = False
                if (overview['auto_train'] and overview['reviewed'] >= 12
                        and overview['new_since_training'] >= overview['auto_train_every']
                        and not self.app.router.status().get('active')):
                    self.app.router.start('train')
                    auto_started = True
                return self.json_response({'ok': True, **result,
                                           'auto_started': auto_started})
            if path == '/api/router/activate':
                router_learning.activate(self.app.catalog_folder, body.get('version', ''))
                return self.json_response({'ok': True})
            if path == '/api/device/exclusions':
                db = catalog_index.connect(self.app.device.catalog)
                try:
                    return self.json_response(catalog_index.set_exclusions(
                        db, body.get('add', []), body.get('remove', [])))
                finally:
                    db.close()
            if path == '/api/device/history/forget':
                return self.json_response(self.app.device.forget(body.get('id')))
            if path == '/api/device/job/start':
                return self.json_response({'ok': True, 'job': self.app.device.start(
                    body.get('roots', []), body.get('features', {}), body.get('paths', []),
                    force=bool(body.get('force')), visual_model=body.get('visual_model'))})
            if path == '/api/device/job/stop':
                return self.json_response({'ok': True, 'job': self.app.device.stop()})
            if path == '/api/photos/process':
                return self.json_response({'ok': True, 'job': self.app.device.start(
                    [], body.get('features', {}), body.get('paths', []),
                    force=bool(body.get('force')), visual_model=body.get('visual_model'))})
            if path == '/api/photos/search-upload':
                raw_path = body.get('path', '')
                row = self.app.store.db.execute(
                    "SELECT path FROM photos WHERE path=? AND status='ok'",
                    (raw_path,)).fetchone()
                if row is None:
                    return self.error_json(404, 'Снимок не найден в каталоге')
                file_path = self.app.file_for(row[0]).resolve()
                if not file_path.is_file():
                    return self.error_json(404, 'Файл не найден')
                try:
                    # У ролика для поиска берём тот же представительный кадр,
                    # что и для остальных этапов анализа.
                    image = video_media.open_frame(file_path)
                    image.thumbnail((1600, 1600))
                    stream = io.BytesIO()
                    image.save(stream, 'JPEG', quality=88, optimize=True)
                    url = reverse_search.upload(stream.getvalue(), Path(raw_path).name)
                except Exception as exc:
                    print(f'Reverse search upload failed for {raw_path}: {exc}',
                         file=sys.stderr, flush=True)
                    return self.error_json(502, f'Не удалось загрузить снимок для поиска: {exc}')
                return self.json_response({'ok': True, 'url': url})
            if path == '/api/photos/assign-speaker':
                raw_path = body.get('path', '')
                speaker = body.get('speaker', '')
                if not speaker:
                    return self.error_json(400, 'Не указан голос')
                with self.app.lock:
                    row = self.app.store.db.execute(
                        'SELECT embedding FROM video_speakers WHERE path=? AND speaker=?',
                        (raw_path, speaker)).fetchone()
                    if row is None:
                        return self.error_json(404, 'Голос не найден')
                    try:
                        with self.app.store.db:
                            person_id = self.app.store.find_or_create_person(
                                body.get('name', ''), body.get('bigfam_id'))
                            self.app.store.db.execute(
                                'UPDATE video_speakers SET assigned_person_id=? '
                                'WHERE path=? AND speaker=?', (person_id, raw_path, speaker))
                            # Человек уже подтвердил голос руками — это такой же
                            # надёжный образец, как связка через лицо в кадре.
                            if row[0]:
                                speaker_diarization.apply_voice_sample(
                                    self.app.store.db, person_id,
                                    np.frombuffer(row[0], dtype='<f4'))
                    except ValueError as exc:
                        return self.error_json(400, str(exc))
                return self.json_response({'ok': True})
            if path == '/api/photos/hide':
                viewer, admin = self.viewer
                with self.app.lock:
                    result = privacy.hide(
                        self.app.store.db, self.app.catalog_folder, viewer,
                        body.get('paths', []),
                        catalog_settings.read(self.app.store.db))
                return self.json_response({'ok': True, **result})
            if path == '/api/photos/reveal':
                viewer, admin = self.viewer
                with self.app.lock:
                    result = privacy.reveal(self.app.store.db, viewer, admin,
                                            body.get('paths', []))
                return self.json_response({'ok': True, **result})
            if path == '/api/recluster/start':
                return self.json_response({'ok': True, 'job': self.app.recluster_job.start()})
            if path == '/api/recluster/stop':
                return self.json_response({'ok': True, 'job': self.app.recluster_job.stop()})
            if path == '/api/duplicates/scan':
                return self.json_response({'ok': True, 'job': self.app.duplicates.start(
                    bool(body.get('similar')))})
            if path == '/api/duplicates/stop':
                return self.json_response({'ok': True, 'job': self.app.duplicates.stop()})
            if path == '/api/settings':
                with self.app.lock:
                    before = catalog_settings.read(self.app.store.db)
                    values = catalog_settings.write(self.app.store.db, body.get('settings', {}))
                    excluded = None
                    if any(before[key] != values[key]
                           for key in ('block_paths', 'allow_paths')):
                        # Правила поменялись — пересчитываем отметки сразу,
                        # чтобы галерея обновилась без пересканирования.
                        excluded = pathrules.apply(self.app.store.db, values)
                return self.json_response({'ok': True, 'settings': values,
                                           'excluded': excluded})
            if path.startswith('/api/albums/'):
                action = path.rsplit('/', 1)[-1]
                with self.app.lock:
                    db = self.app.store.db
                    if action == 'create':
                        album_id = albums.create(db, body.get('title', ''),
                                                 body.get('parent_id', 0))
                        if body.get('paths'):
                            albums.set_photos(db, album_id, add=body['paths'])
                        result = {'id': album_id}
                    elif action == 'rename':
                        albums.rename(db, body.get('id'), body.get('title', ''))
                        result = {'id': body.get('id')}
                    elif action == 'move':
                        albums.move(db, body.get('id'), body.get('parent_id', 0))
                        result = {'id': body.get('id')}
                    elif action == 'delete':
                        result = {'removed': albums.remove(db, body.get('id'))}
                    elif action == 'photos':
                        result = {'photos': albums.set_photos(
                            db, body.get('id'), body.get('add', []), body.get('remove', []))}
                    else:
                        return self.error_json(404, 'Неизвестное действие с альбомом')
                    return self.json_response({'ok': True, **result,
                                               'albums': albums.tree(db)})
            if path.startswith('/api/people-albums/'):
                action = path.rsplit('/', 1)[-1]
                viewer, admin = self.viewer
                with self.app.lock:
                    db = self.app.store.db
                    if action == 'hidden' and not admin:
                        return self.error_json(403, 'Скрывать альбомы может только администратор')
                    if action == 'create':
                        album_id = people_albums.create(db, body.get('title', ''),
                                                        body.get('parent_id', 0))
                        if body.get('group_keys'):
                            people_albums.set_members(db, album_id, add=body['group_keys'])
                        result = {'id': album_id}
                    elif action == 'rename':
                        people_albums.rename(db, body.get('id'), body.get('title', ''))
                        result = {'id': body.get('id')}
                    elif action == 'move':
                        people_albums.move(db, body.get('id'), body.get('parent_id', 0))
                        result = {'id': body.get('id')}
                    elif action == 'delete':
                        result = {'removed': people_albums.remove(db, body.get('id'))}
                    elif action == 'members':
                        result = {'members': people_albums.set_members(
                            db, body.get('id'), body.get('add', []), body.get('remove', []))}
                    elif action == 'hidden':
                        people_albums.set_hidden(db, body.get('id'), bool(body.get('hidden')))
                        result = {'id': body.get('id')}
                    else:
                        return self.error_json(404, 'Неизвестное действие с альбомом')
                    return self.json_response({'ok': True, **result,
                                               'state': self.app.state(viewer, admin)})
            if path == '/api/photos/delete':
                return self.json_response({'ok': True, **self.app.delete_photos(
                    body.get('paths', []))})
            with self.app.lock:
                if path == '/api/assign-groups':
                    self.app.store.assign(self.app.resolve_groups(body.get('group_keys', [])),
                                          body.get('name', ''), body.get('bigfam_id'))
                elif path == '/api/assign-faces':
                    self.app.store.assign(body.get('face_ids', []), body.get('name', ''),
                                          body.get('bigfam_id'))
                elif path == '/api/set-avatar':
                    self.app.store.set_avatar(body.get('key', ''), body.get('face_id'))
                elif path == '/api/clear-avatar':
                    self.app.store.clear_avatar(body.get('key', ''))
                elif path == '/api/exclude':
                    self.app.store.exclude(body.get('face_ids', []))
                elif path == '/api/undo':
                    description = self.app.store.undo()
                    return self.json_response({'ok': True, 'description': description,
                                               'state': self.app.state()})
                else:
                    return self.error_json(404, 'Неизвестное действие')
            return self.json_response({'ok': True, 'state': self.app.state()})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            return self.error_json(400, str(exc))
        except Exception as exc:
            print(f'POST error: {exc}', file=sys.stderr, flush=True)
            return self.error_json(500, 'Внутренняя ошибка локального сервера')

    def send_media(self, raw_id, original):
        try:
            face_id = int(raw_id)
            row = self.app.store.by_id[face_id]
        except (ValueError, KeyError):
            return self.error_json(404, 'Лицо не найдено')
        path = (self.app.file_for(row[1]).resolve() if original
                else (self.app.store.folder / row[3]).resolve())
        if not path.is_file():
            return self.error_json(404, 'Файл не найден')
        if original and video_media.is_video(path):
            return self.send_range(path, mimetypes.guess_type(path.name)[0] or 'video/mp4')
        content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        self.send_headers(200, content_type, path.stat().st_size, 'private, max-age=3600')
        with path.open('rb') as stream:
            shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)

    def send_face_crop(self, raw_id, margin=0.5, size=640):
        """Квадратный кадр вокруг лица из оригинала — на аватарку миниатюры мелковаты."""
        try:
            face_id = int(raw_id)
            row = self.app.store.by_id[face_id]
        except (ValueError, KeyError):
            return self.error_json(404, 'Лицо не найдено')
        box = self.app.store.db.execute(
            'SELECT box,frame_time FROM faces WHERE id=?', (face_id,)).fetchone()
        path = self.app.file_for(row[1]).resolve()
        if box is None or not box[0] or not path.is_file():
            return self.error_json(404, 'Файл не найден')
        margin = min(max(margin, 0.0), 2.0)
        size = min(max(size, 64), 1600)
        try:
            if video_media.is_video(path):
                # Лицо жило на конкретной секунде ролика — туда и перематываем.
                image = video_media.to_image(video_media.poster(path, box[1] or 0))
            else:
                with Image.open(path) as original:
                    image = ImageOps.exif_transpose(original).convert('RGB')
            left, top, right, bottom = (float(value) for value in json.loads(box[0]))
            side = max(right - left, bottom - top) * (1 + margin)
            center_x, center_y = (left + right) / 2, (top + bottom) / 2
            crop = image.crop((
                max(0, int(center_x - side / 2)), max(0, int(center_y - side / 2)),
                min(image.width, int(center_x + side / 2)),
                min(image.height, int(center_y + side / 2))))
            crop.thumbnail((size, size))
            stream = io.BytesIO()
            crop.save(stream, 'JPEG', quality=88)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f'Face crop failed for {face_id}: {exc}', file=sys.stderr, flush=True)
            return self.error_json(404, 'Не удалось вырезать кадр')
        body = stream.getvalue()
        self.send_headers(200, 'image/jpeg', len(body), 'private, max-age=3600')
        self.wfile.write(body)

    # Пороги ниже, чем у рейтинга: лишний размытый кусок не страшен, пропущенный — страшен.
    INTIMATE = {'FEMALE_GENITALIA_EXPOSED': .25, 'MALE_GENITALIA_EXPOSED': .25,
                'ANUS_EXPOSED': .25}
    BARE = {'FEMALE_BREAST_EXPOSED': .35, 'BUTTOCKS_EXPOSED': .35}

    @staticmethod
    def blur_regions(image, regions, mode):
        """Замыливает найденные области; возвращает, сколько закрыл."""
        from PIL import ImageFilter
        wanted = dict(Handler.INTIMATE)
        if mode == 'regions':
            wanted.update(Handler.BARE)
        covered = 0
        for attempt in (wanted, {**Handler.INTIMATE, **Handler.BARE}):
            for item in regions:
                limit = attempt.get(item.get('class'))
                if limit is None or float(item.get('score', 0)) < limit:
                    continue
                # NudeNet отдаёт рамку как [x, y, ширина, высота].
                x, y, width, height = (int(value) for value in item['box'])
                if width <= 0 or height <= 0:
                    continue
                pad = max(8, round(max(width, height) * .22))
                box = (max(0, x - pad), max(0, y - pad),
                       min(image.width, x + width + pad), min(image.height, y + height + pad))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                piece = image.crop(box)
                # Сначала пикселизация, потом размытие: так не остаётся читаемых краёв.
                small = piece.resize((max(1, piece.width // 24), max(1, piece.height // 24)))
                piece = small.resize(piece.size).filter(
                    ImageFilter.GaussianBlur(max(12, min(piece.size) / 3)))
                image.paste(piece, box)
                covered += 1
            if covered:
                break
        return covered

    # Лестница размеров: превью кэшируются, и одинаковые запросы попадают в одну копию.
    PREVIEW_STEPS = (240, 400, 640, 960, 1440, 2200)

    def preview_file(self, path, size, blur, stamp, moment=0):
        """Путь кэша уменьшенной копии; ключ помнит файл, время, размер, блюр и секунду."""
        tail = f'|{round(moment, 2)}' if moment else ''
        key = hashlib.sha1(f'{path}|{stamp}|{size}|{blur}{tail}'.encode('utf-8')).hexdigest()
        folder = self.app.store.folder / 'previews' / key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{key}.jpg'

    def render_preview(self, path, size, blur, raw_path, moment=0):
        """Готовит уменьшенную копию, при необходимости с замыленными областями."""
        from PIL import Image, ImageFilter, ImageOps
        is_video = video_media.is_video(path)
        if is_video:
            # У ролика обложка — кадр из начала; дальше всё как с фотографией,
            # включая блюр: рейтинг 18+ у видео теперь тоже считается.
            image = video_media.to_image(video_media.poster(path, moment))
        elif blur:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
        else:
            with Image.open(path) as source:
                # draft разбирает JPEG сразу в уменьшенном виде — это в разы быстрее.
                source.draft('RGB', (size * 2, size * 2))
                image = ImageOps.exif_transpose(source).convert('RGB')
        if blur:
            row = self.app.store.db.execute(
                'SELECT rating,regions_json FROM photo_adult_analysis '
                "WHERE path=? AND status='ok'", (raw_path,)).fetchone()
            rating = row[0] if row else 'unknown'
            regions = json.loads(row[1]) if row else []
            covered = 0 if blur == 'full' else self.blur_regions(image, regions, blur)
            # Кадр считается небезопасным, а закрывать нечего — прячем целиком:
            # детектор мелкий и промахивается, оставлять «как есть» нельзя.
            if blur == 'full' or (not covered and rating in {'explicit', 'nudity'}):
                image = image.filter(ImageFilter.GaussianBlur(max(18, min(image.size) / 24)))
        image.thumbnail((size, size))
        stream = io.BytesIO()
        image.save(stream, 'JPEG', quality=84, optimize=True, progressive=True)
        return stream.getvalue()

    def send_range(self, path, content_type, cache='private, max-age=3600'):
        """Отдаём файл кусками: без этого браузер не перематывает видео."""
        total = path.stat().st_size
        header = self.headers.get('Range', '')
        start, end = 0, total - 1
        partial = False
        if header.startswith('bytes='):
            first, _, last = header[6:].split(',')[0].partition('-')
            try:
                if first:
                    start = int(first)
                    end = int(last) if last else end
                else:
                    start = max(0, total - int(last))
                partial = True
            except ValueError:
                partial = False
        if partial and (start >= total or start > end):
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{total}')
            self.end_headers()
            return
        end = min(end, total - 1)
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(length))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', cache)
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

    def send_video(self, raw_path):
        row = self.app.store.db.execute(
            "SELECT path FROM photos WHERE path=? AND status='ok'", (raw_path,)).fetchone()
        if row is None:
            return self.error_json(404, 'Видео не найдено в каталоге')
        path = self.app.file_for(row[0]).resolve()
        if not path.is_file():
            return self.error_json(404, 'Файл не найден')
        content_type = mimetypes.guess_type(path.name)[0] or 'video/mp4'
        return self.send_range(path, content_type)

    def send_photo(self, raw_path, blur='', size=0, moment=0):
        row = self.app.store.db.execute(
            "SELECT path FROM photos WHERE path=? AND status='ok'", (raw_path,)).fetchone()
        if row is None:
            return self.error_json(404, 'Фотография не найдена в каталоге')
        path = self.app.file_for(row[0]).resolve()
        if not path.is_file():
            return self.error_json(404, 'Файл не найден')
        blur = blur if blur in {'explicit', 'regions', 'full'} else ''
        size = next((step for step in self.PREVIEW_STEPS if step >= size), self.PREVIEW_STEPS[-1]) \
            if size > 0 else (self.PREVIEW_STEPS[-1] if blur else 0)
        if video_media.is_video(path):
            # Целиком ролик сюда отдавать нельзя — это картинка обложки,
            # но блюр 18+ на ней работает так же, как у обычного фото.
            size = size or self.PREVIEW_STEPS[-2]
        if size or blur:
            try:
                stamp = path.stat().st_mtime_ns
                cache = self.preview_file(path, size, blur, stamp, moment)
                if not cache.is_file():
                    body = self.render_preview(path, size, blur, raw_path, moment)
                    temporary = cache.with_suffix('.tmp')
                    temporary.write_bytes(body)
                    os.replace(temporary, cache)
                else:
                    body = cache.read_bytes()
                # Ключ кэша включает время файла, а ссылка — параметр v, поэтому
                # браузеру можно разрешить держать копию долго.
                self.send_headers(200, 'image/jpeg', len(body), 'private, max-age=604800')
                self.wfile.write(body)
                return
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f'Preview failed for {raw_path}: {exc}', file=sys.stderr, flush=True)
        content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        self.send_headers(200, content_type, path.stat().st_size, 'private, max-age=3600')
        with path.open('rb') as stream:
            shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)

    def send_static(self, raw_path):
        if not WEB_ROOT.is_dir():
            return self.error_json(404, 'Этот процесс отдаёт только API и медиа; '
                                        'интерфейс живёт на Proxmox')
        relative = 'index.html' if raw_path in {'', '/'} else unquote(raw_path.lstrip('/'))
        path = (WEB_ROOT / relative).resolve()
        if WEB_ROOT not in path.parents or not path.is_file():
            return self.error_json(404, 'Страница не найдена')
        body = path.read_bytes()
        if path.name == 'index.html':
            body = body.replace(b'__LOCAL_TOKEN__', self.app.token.encode('ascii'))
        content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if content_type.startswith('text/') or content_type in {'application/javascript', 'application/json'}:
            content_type += '; charset=utf-8'
        self.send_headers(200, content_type, len(body))
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--min-cluster-size', type=int, default=8)
    parser.add_argument('--max-faces', type=int, default=0,
                        help='Потолок лиц в каталоге; 0 — без ограничения')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--token-file', type=Path,
                        help='файл с постоянным токеном; создаётся, если его нет')
    parser.add_argument('--device-id')
    parser.add_argument('--device-name')
    args = parser.parse_args()
    if args.host not in {'0.0.0.0', '127.0.0.1', 'localhost'}:
        parser.error('--host must be 0.0.0.0, 127.0.0.1 or localhost')
    token = None
    if args.token_file:
        if not args.token_file.exists():
            args.token_file.parent.mkdir(parents=True, exist_ok=True)
            args.token_file.write_text(secrets.token_urlsafe(32) + chr(10), encoding='ascii')
        token = args.token_file.read_text(encoding='ascii').strip()
        if not token:
            parser.error(f'Пустой токен в {args.token_file}')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.app = App(args.data, args.min_cluster_size, token,
                     args.device_id, args.device_name, args.max_faces)
    browser_host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    url = f'http://{browser_host}:{server.server_port}/'
    print(f'Local photo UI: {url}', flush=True)
    if args.host == '0.0.0.0':
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET) if not item[4][0].startswith('127.')})
        for address in addresses:
            print(f'LAN photo UI: http://{address}:{server.server_port}/', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.app.store.db.close()


if __name__ == '__main__':
    main()
