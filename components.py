"""Окружения и модели ядра: что стоит, установка, загрузка и копия с другого ядра.

Окружение (vision, audio, imgutils, OCR) ставится своим pip по
requirements-*.txt. Модель с Hugging Face качается snapshot_download в кэш
ядра, а если она уже есть на другом ядре — копируется оттуда по домашней сети:
хаб берёт у ядра-источника пропуск (ticket) только на чтение моделей и отдаёт
его ядру-получателю, так что токены ядер друг другу не достаются.

Одновременно идёт одна операция; её ход — в status()['operation'].
"""
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import envs

HERE = Path(__file__).resolve().parent
TICKET_SECONDS = 6 * 3600

VENVS = (
    {'id': 'vision', 'title': 'Визуальный поиск, описания, 18+', 'venv': 'vision-venv',
     'requirements': 'requirements-vision.txt', 'features': ['visual', 'caption', 'adult']},
    {'id': 'audio', 'title': 'Речь и голоса', 'venv': 'audio-venv',
     'requirements': 'requirements-audio.txt', 'features': ['speech', 'diarize']},
    {'id': 'imgutils', 'title': 'Рисованные лица', 'venv': 'imgutils-venv',
     'requirements': 'requirements-imgutils.txt', 'features': ['authenticity']},
    {'id': 'ocr', 'title': 'Текст на снимках (OCR)', 'venv': 'cv-ocr',
     'requirements': 'requirements-ocr.txt', 'features': ['ocr']},
)

# base: hf — кэш Hugging Face (repo и extra), cache — папка кэша моделей,
# models — папка models ядра.
MODELS = (
    {'id': 'faces', 'title': 'Лица: InsightFace buffalo_l', 'base': 'models',
     'paths': ['buffalo_l'], 'features': ['faces']},
    {'id': 'siglip2-224', 'title': 'Поиск: SigLIP 2 (224)', 'base': 'hf',
     'repo': 'google/siglip2-base-patch16-224', 'features': ['visual']},
    {'id': 'siglip2-256', 'title': 'Поиск: SigLIP 2 (256)', 'base': 'hf',
     'repo': 'google/siglip2-base-patch16-256', 'features': ['visual']},
    {'id': 'jina-clip-v2', 'title': 'Поиск: Jina CLIP v2', 'base': 'hf',
     'repo': 'jinaai/jina-clip-v2',
     'extra': ['jinaai/jina-clip-implementation', 'jinaai/jina-embeddings-v3',
               'jinaai/xlm-roberta-flash-implementation'], 'features': ['visual']},
    {'id': 'qwen3-vl', 'title': 'Описания: Qwen3-VL 2B', 'base': 'hf',
     'repo': 'Qwen/Qwen3-VL-2B-Instruct', 'features': ['caption']},
    {'id': 'wd-tagger', 'title': '18+: WD EVA02 tagger', 'base': 'hf',
     'repo': 'SmilingWolf/wd-eva02-large-tagger-v3', 'features': ['adult']},
    {'id': 'whisper', 'title': 'Речь: Whisper large-v3', 'base': 'hf',
     'repo': 'Systran/faster-whisper-large-v3', 'features': ['speech']},
    {'id': 'pyannote', 'title': 'Голоса: pyannote community-1', 'base': 'hf',
     'repo': 'pyannote/speaker-diarization-community-1', 'gated': True,
     'features': ['diarize']},
    {'id': 'anime-real', 'title': 'Рисованные лица: anime_real_cls', 'base': 'hf',
     'repo': 'deepghs/anime_real_cls', 'features': ['authenticity']},
    {'id': 'paddle', 'title': 'OCR: модели PaddleOCR', 'base': 'cache', 'paths': ['paddle'],
     'features': ['ocr'], 'note': 'скачаются сами при первом OCR'},
    {'id': 'ram-plus', 'title': 'Разметчик RAM++', 'base': 'cache', 'paths': ['ram-plus'],
     'features': [],
     'note': 'загрузка — setup-ram.ps1'},
)


def repo_dir(repo):
    return 'hub/models--' + repo.replace('/', '--')


def model_base(item, root=envs.ROOT):
    if item['base'] == 'hf':
        return envs.hf_home(root)
    if item['base'] == 'models':
        return envs.real(Path(root) / 'models')
    return envs.models_root(root)


def model_paths(item):
    """Папки модели относительно её base, в виде posix-строк."""
    if item['base'] == 'hf':
        return [repo_dir(repo) for repo in [item['repo'], *item.get('extra', [])]]
    return list(item['paths'])


def has_files(folder):
    try:
        return any(entry.is_file() for entry in Path(folder).rglob('*'))
    except OSError:
        return False


def venv_python(item, root=envs.ROOT):
    if item['id'] == 'ocr':
        return envs.ocr_python(root)
    return envs.python(item['venv'], root)


def base_python(root=envs.ROOT):
    """Python, от которого сделан .venv ядра: из него же создаём остальные окружения."""
    try:
        for line in (Path(root) / '.venv' / 'pyvenv.cfg').read_text(encoding='utf-8').splitlines():
            key, _, value = line.partition('=')
            if key.strip() == 'home':
                candidate = Path(value.strip()) / ('python.exe' if os.name == 'nt' else 'python3')
                if candidate.is_file():
                    return candidate
    except OSError:
        pass
    return Path(getattr(sys, '_base_executable', sys.executable))


def find(item_id):
    for item in (*VENVS, *MODELS):
        if item['id'] == item_id:
            return item
    raise KeyError(f'Нет такого окружения или модели: {item_id}')


def safe_relative(item, relative):
    """Путь файла модели из манифеста: только внутри папок этой модели."""
    path = PurePosixPath(str(relative).replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or not path.parts or ':' in path.parts[0]:
        raise ValueError(f'Недопустимый путь файла модели: {relative!r}')
    text = path.as_posix()
    if not any(text == folder or text.startswith(folder + '/') for folder in model_paths(item)):
        raise ValueError(f'Файл не из этой модели: {relative!r}')
    return text


class Components:
    def __init__(self, root=envs.ROOT, log_folder=None):
        self.root = Path(root)
        self.log_folder = Path(log_folder) if log_folder else self.root
        self.lock = threading.RLock()
        self.operation = None
        self.process = None
        self.stop_requested = False
        self.sink = None
        self.tickets = {}
        self.sizes = {}

    # ----- состояние -----

    def model_bytes(self, item):
        """Размер модели на диске; обходить гигабайты при каждом опросе дорого — кэш на 5 минут."""
        cached = self.sizes.get(item['id'])
        if cached and time.time() - cached[0] < 300:
            return cached[1]
        total = sum(entry['size'] for entry in self.manifest(item['id'])['files'])
        self.sizes[item['id']] = (time.time(), total)
        return total

    def status(self):
        venvs = [{'id': item['id'], 'kind': 'venv', 'title': item['title'],
                  'features': item['features'],
                  'installed': venv_python(item, self.root).is_file()} for item in VENVS]
        models = []
        for item in MODELS:
            base = model_base(item, self.root)
            installed = all(has_files(base / folder) for folder in model_paths(item))
            models.append({
                'id': item['id'], 'kind': 'model', 'title': item['title'],
                'features': item['features'], 'installed': installed,
                'bytes': self.model_bytes(item) if installed else 0,
                'downloadable': item['base'] == 'hf', 'gated': bool(item.get('gated')),
                'note': item.get('note', '')})
        with self.lock:
            operation = dict(self.operation) if self.operation else None
            if operation:
                operation['log'] = operation['log'][-60:]
        return {'venvs': venvs, 'models': models, 'operation': operation}

    def manifest(self, item_id):
        item = find(item_id)
        if item not in MODELS:
            raise ValueError('Копировать можно только модели')
        base = model_base(item, self.root)
        files = []
        for folder in model_paths(item):
            start = base / folder
            if not start.is_dir():
                continue
            for dirpath, _dirs, names in os.walk(start):
                for name in names:
                    if name.endswith('.part') or name.endswith('.lock'):
                        continue
                    path = Path(dirpath) / name
                    try:
                        size = path.stat().st_size
                    except OSError:
                        continue
                    files.append({'path': path.relative_to(base).as_posix(), 'size': size})
        return {'id': item_id, 'files': files}

    def file_path(self, item_id, relative):
        item = find(item_id)
        return model_base(item, self.root) / safe_relative(item, relative)

    # ----- пропуска для копии -----

    def issue_ticket(self):
        now = time.time()
        with self.lock:
            self.tickets = {key: until for key, until in self.tickets.items() if until > now}
            ticket = secrets.token_urlsafe(32)
            self.tickets[ticket] = now + TICKET_SECONDS
        return {'ticket': ticket, 'expires_in': TICKET_SECONDS}

    def ticket_ok(self, ticket):
        if not ticket:
            return False
        now = time.time()
        with self.lock:
            return any(secrets.compare_digest(key, ticket) and until > now
                       for key, until in self.tickets.items())

    # ----- операции -----

    def start(self, item_id, action, peer=None):
        item = find(item_id)
        if action not in {'install', 'download', 'copy'}:
            raise ValueError('Неизвестное действие')
        if action == 'install' and item not in VENVS:
            raise ValueError('Устанавливаются окружения; модели — скачать или скопировать')
        if action == 'download' and item.get('base') != 'hf':
            raise ValueError(f'«{item["title"]}» не качается отсюда: '
                             + (item.get('note') or 'скопируйте с другого ядра'))
        if action == 'copy':
            if item not in MODELS:
                raise ValueError('Копировать можно только модели')
            if not peer or not peer.get('url') or not peer.get('ticket'):
                raise ValueError('Не указано ядро, с которого копировать')
        with self.lock:
            if self.operation and self.operation['status'] == 'running':
                raise ValueError('Уже идёт установка: '
                                 + find(self.operation['id'])['title'])
            self.stop_requested = False
            self.operation = {'id': item_id, 'action': action, 'status': 'running',
                              'started_at': time.time(), 'finished_at': None, 'error': '',
                              'done': 0, 'total': 0, 'log': []}
            state = self.operation
        threading.Thread(target=self._run, args=(state, item, action, peer), daemon=True).start()
        return dict(state)

    def stop(self):
        with self.lock:
            self.stop_requested = True
            process = self.process
        if process and process.poll() is None:
            process.kill()
        return {'ok': True}

    def log(self, state, text):
        with self.lock:
            for line in str(text).splitlines():
                line = line.rstrip()
                if line:
                    state['log'].append(line[:400])
            del state['log'][:-300]

    def _run(self, state, item, action, peer):
        log_file = self.log_folder / f'component-{item["id"]}.log'
        try:
            self.log_folder.mkdir(parents=True, exist_ok=True)
            with log_file.open('w', encoding='utf-8', errors='replace') as sink:
                self.sink = sink
                if action == 'install':
                    self._install(state, item)
                elif action == 'download':
                    self._download(state, item)
                else:
                    self._copy(state, item, peer)
            if self.stop_requested:
                raise RuntimeError('Остановлено')
            self.sizes.pop(item['id'], None)
            with self.lock:
                state.update(status='completed', finished_at=time.time())
            self.log(state, 'Готово')
        except Exception as exc:
            self.log(state, f'Ошибка: {exc}')
            with self.lock:
                state.update(status='error', error=str(exc), finished_at=time.time())
        finally:
            self.sink = None
            self.process = None

    def _command(self, state, command, env=None):
        self.log(state, '> ' + ' '.join(str(part) for part in command[:6]))
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.Popen([str(part) for part in command], cwd=self.root,
                                   env={**os.environ, 'PYTHONUTF8': '1', **(env or {})},
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, creationflags=flags,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        with self.lock:
            self.process = process
        for line in process.stdout:
            if self.sink:
                self.sink.write(line)
            self.log(state, line)
        code = process.wait()
        if self.stop_requested:
            raise RuntimeError('Остановлено')
        if code:
            raise RuntimeError(f'Команда завершилась с кодом {code}')

    def _install(self, state, item):
        python = venv_python(item, self.root)
        if not python.is_file():
            target = python.parent.parent
            target.parent.mkdir(parents=True, exist_ok=True)
            base = base_python(self.root)
            self.log(state, f'Создаю окружение {target} от {base}')
            self._command(state, [base, '-m', 'venv', target])
        self._command(state, [python, '-m', 'pip', 'install', '--disable-pip-version-check',
                              '--upgrade', 'pip'])
        self._command(state, [python, '-m', 'pip', 'install', '--disable-pip-version-check',
                              '-r', HERE / item['requirements']])

    def hub_python(self):
        """Окружение, где есть huggingface_hub: им и качаем модели."""
        for item in VENVS:
            python = venv_python(item, self.root)
            if python.is_file() and subprocess.run(
                    [str(python), '-c', 'import huggingface_hub'], capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            ).returncode == 0:
                return python
        raise RuntimeError('Для загрузки моделей нужно окружение с huggingface_hub — '
                           'сначала установите «Визуальный поиск»')

    def _download(self, state, item):
        python = self.hub_python()
        token_file = self.root / 'hf-token.txt'
        token = token_file.read_text(encoding='utf-8').strip() if token_file.is_file() else ''
        if item.get('gated') and not token:
            raise RuntimeError('Модель закрыта лицензией Hugging Face: нужен hf-token.txt '
                               'в папке ядра (или скопируйте её с другого ядра)')
        cache = envs.hf_home(self.root) / 'hub'
        cache.mkdir(parents=True, exist_ok=True)
        script = ('import sys\nfrom huggingface_hub import snapshot_download\n'
                  'for repo in sys.argv[2:]:\n'
                  '    print("Качаю", repo, flush=True)\n'
                  '    snapshot_download(repo, cache_dir=sys.argv[1], token=__import__("os")'
                  '.environ.get("HF_TOKEN") or None)\n')
        env = {'HF_HUB_DISABLE_XET': '1', 'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
               'HF_HUB_OFFLINE': '0'}
        if token:
            env['HF_TOKEN'] = token
        self._command(state, [python, '-c', script, cache, item['repo'],
                              *item.get('extra', [])], env=env)

    def _copy(self, state, item, peer):
        url = peer['url'].rstrip('/')
        headers = {'X-Core-Ticket': peer['ticket']}
        request = Request(f'{url}/api/core/components/manifest?id={quote(item["id"])}',
                          headers=headers)
        with urlopen(request, timeout=120) as response:
            files = json.load(response)['files']
        if not files:
            raise RuntimeError('На том ядре этой модели нет')
        base = model_base(item, self.root)
        total = sum(entry['size'] for entry in files)
        with self.lock:
            state['total'] = total
        self.log(state, f'Копирую {len(files)} файлов, {total / 1e9:.2f} ГБ с {url}')
        done = 0
        for entry in files:
            if self.stop_requested:
                raise RuntimeError('Остановлено')
            relative = safe_relative(item, entry['path'])
            target = base / relative
            if target.is_file() and target.stat().st_size == entry['size']:
                done += entry['size']
                with self.lock:
                    state['done'] = done
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.part')
            request = Request(f'{url}/api/core/components/file?id={quote(item["id"])}'
                              f'&path={quote(relative)}', headers=headers)
            with urlopen(request, timeout=120) as response, partial.open('wb') as output:
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    if self.stop_requested:
                        break
                    output.write(chunk)
                    done += len(chunk)
                    with self.lock:
                        state['done'] = done
            if self.stop_requested:
                partial.unlink(missing_ok=True)
                raise RuntimeError('Остановлено')
            if partial.stat().st_size != entry['size']:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f'Файл пришёл не целиком: {relative}')
            os.replace(partial, target)
        self.log(state, f'Скопировано {done / 1e9:.2f} ГБ')
