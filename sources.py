"""Источники фотографий: где лежат оригиналы и как до них добраться.

Источник — это место с самими снимками и роликами: диск компьютера с ядром,
сетевая папка SMB (например, флешка в роутере), сервер SSH/SFTP, FTP или
WebDAV. В каталоге снимок записан ключом `источник:путь` (см. pathkeys.py),
а всё, что про него посчитано — лица, превью, описания, — живёт на хабе. В
самом источнике ничего не создаётся.

Здесь три слоя:

* записи об источниках (Registry, живёт на хабе в sources.json вместе с
  паролями; наружу — только public());
* драйверы протоколов с одним набором операций: listdir, stat, open (файл с
  seek — для Range и PIL), remove, rename, makedirs, roots;
* `local(key)` — путь к файлу на этой машине для этапов обработки: у своего
  диска это сам путь, у SMB на Windows — UNC-путь, у остального — копия во
  временной папке ядра, которую готовит служба ядра.

Сторонние библиотеки (paramiko, smbprotocol) подключаются только там, где
нужен их протокол: этапы в отдельных окружениях импортируют модуль ради
`local()`, и им хватает стандартной библиотеки.
"""
from dataclasses import dataclass
import ftplib
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import pathkeys

TYPES = {
    'device': 'Диск устройства с ядром',
    'smb': 'Сетевая папка (SMB)',
    'sftp': 'SSH / SFTP',
    'ftp': 'FTP',
    'webdav': 'WebDAV',
    'local': 'Папка на хабе',
}
DEFAULT_PORTS = {'smb': 445, 'sftp': 22, 'ftp': 21, 'webdav': 443}
SECRET_FIELDS = ('password',)


class SourceError(OSError):
    pass


@dataclass
class Entry:
    name: str
    is_dir: bool
    size: int = 0
    mtime_ns: int = 0


# ---------- записи ----------

def validate(value, current=None):
    """Проверяет запись источника из интерфейса. Пустой пароль — «оставить как был»."""
    current = current or {}
    source_id = str(value.get('id') or current.get('id') or '').strip().lower()
    if not pathkeys.SOURCE_ID.match(source_id):
        raise ValueError('Id источника: латиница, цифры, «-» и «_», от 2 до 32 символов')
    kind = str(value.get('type') or current.get('type') or '')
    if kind not in TYPES:
        raise ValueError('Неизвестный тип источника')
    name = str(value.get('name') or '').strip() or current.get('name') or source_id
    if len(name) > 80:
        raise ValueError('Слишком длинное имя источника')
    record = {
        'id': source_id, 'name': name, 'type': kind,
        'host': str(value.get('host') or '').strip(),
        'port': int(value.get('port') or 0) or DEFAULT_PORTS.get(kind, 0),
        'user': str(value.get('user') or '').strip(),
        'share': str(value.get('share') or '').strip().strip('\\/'),
        'path': str(value.get('path') or '').strip(),
        'device': str(value.get('device') or '').strip().lower(),
        'secure': bool(value.get('secure')),
        'enabled': value.get('enabled', current.get('enabled', True)) is not False,
        'roots': [str(item) for item in (value.get('roots', current.get('roots')) or [])
                  if str(item).strip()],
        'created_at': current.get('created_at') or time.time(),
    }
    password = value.get('password')
    if value.get('clearPassword'):
        record['password'] = ''
    elif password:
        record['password'] = str(password)
    else:
        record['password'] = current.get('password', '')
    if kind == 'device' and not record['device']:
        raise ValueError('Укажите устройство, на диске которого лежат снимки')
    if kind in {'smb', 'sftp', 'ftp', 'webdav'} and not record['host']:
        raise ValueError('Укажите адрес сервера')
    if kind == 'local' and not record['path']:
        raise ValueError('Укажите папку на хабе')
    if not 0 < record['port'] < 65536 and kind in DEFAULT_PORTS:
        raise ValueError('Некорректный порт')
    if re.search(r'[\r\n]', json.dumps(record, ensure_ascii=False)):
        raise ValueError('Переносы строк в полях источника недопустимы')
    return record


def public(record):
    """Запись для браузера: без пароля, только признак, что он задан."""
    result = {key: value for key, value in record.items() if key not in SECRET_FIELDS}
    result['hasPassword'] = bool(record.get('password'))
    result['typeName'] = TYPES.get(record.get('type'), record.get('type'))
    return result


def root_key(record):
    """Ключ корня источника: с него начинается дерево в галерее и в выборе папок."""
    kind = record['type']
    if kind == 'smb':
        return pathkeys.make(record['id'], '/' + record['share'] if record.get('share') else '/')
    if kind == 'device':
        return pathkeys.make(record['id'], '')
    return pathkeys.make(record['id'], record.get('path') or '/')


class Registry:
    """sources.json хаба. Файл с паролями: права 600, пишется целиком."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def load(self):
        with self.lock:
            try:
                rows = json.loads(self.path.read_text(encoding='utf-8'))
            except FileNotFoundError:
                return []
            except (OSError, json.JSONDecodeError) as exc:
                raise SourceError(f'Не читается {self.path}: {exc}') from exc
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

    def get(self, source_id):
        return next((row for row in self.load() if row['id'] == source_id), None)

    def save(self, value):
        with self.lock:
            rows = self.load()
            wanted = str(value.get('id') or '').strip().lower()
            index = next((i for i, row in enumerate(rows) if row['id'] == wanted), -1)
            record = validate(value, rows[index] if index >= 0 else None)
            if index >= 0:
                rows[index] = record
            else:
                rows.append(record)
            self.save_all(rows)
            return record

    def remove(self, source_id):
        with self.lock:
            rows = self.load()
            left = [row for row in rows if row['id'] != source_id]
            if len(left) == len(rows):
                raise KeyError('Источник не найден')
            self.save_all(left)


# ---------- файлы с перемоткой ----------

class RangeReader(io.RawIOBase):
    """Файл, у которого есть только «прочитать кусок с такого-то байта»."""

    def __init__(self, size, fetch, name=''):
        self.size = int(size)
        self._fetch = fetch
        self.position = 0
        self.name = name

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def tell(self):
        return self.position

    def readinto(self, buffer):
        if self.position >= self.size:
            return 0
        want = min(len(buffer), self.size - self.position)
        data = self._fetch(self.position, want)
        count = len(data)
        buffer[:count] = data
        self.position += count
        return count


def buffered(raw, size=1024 * 1024):
    return io.BufferedReader(raw, buffer_size=size)


# ---------- драйверы ----------

class Driver:
    """Общий вид доступа к источнику. Пути — «родные» пути внутри источника."""
    kind = ''
    separator = '/'

    def __init__(self, record):
        self.record = record

    def close(self):
        pass

    def test(self):
        self.roots()
        return True

    def roots(self):
        return [Entry(self.record.get('path') or '/', True)]

    def join(self, folder, name):
        return folder.rstrip('\\/') + self.separator + name if folder not in {'', '/'} \
            else (folder or '') + (self.separator if folder != '/' else '') + name

    def read_bytes(self, native, limit=None):
        with self.open(native) as stream:
            data = stream.read(limit if limit else -1)
        return data

    def local_path(self, native):
        return None

    def remove(self, native):
        raise SourceError('Удаление в этом источнике не поддерживается')

    def rename(self, old, new):
        raise SourceError('Перемещение в этом источнике не поддерживается')

    def makedirs(self, native):
        raise SourceError('Создание папок в этом источнике не поддерживается')

    def exists(self, native):
        try:
            self.stat(native)
            return True
        except (FileNotFoundError, SourceError):
            return False


class LocalDriver(Driver):
    """Обычная файловая система этой машины: свой диск ядра или папка хаба."""
    kind = 'local'

    def __init__(self, record=None):
        super().__init__(record or {})
        self.separator = os.sep

    def roots(self):
        if os.name == 'nt':
            import ctypes
            mask = ctypes.windll.kernel32.GetLogicalDrives()
            found = []
            for index in range(26):
                if mask & (1 << index):
                    path = f'{chr(65 + index)}:\\'
                    try:
                        shutil.disk_usage(path)
                    except OSError:
                        continue
                    found.append(Entry(path, True))
            return found
        base = self.record.get('path') or '/'
        return [Entry(base, True)]

    def join(self, folder, name):
        return os.path.join(folder, name)

    def listdir(self, native):
        result = []
        try:
            with os.scandir(native) as items:
                for item in items:
                    try:
                        if item.is_symlink():
                            continue
                        is_dir = item.is_dir()
                        info = item.stat()
                        result.append(Entry(item.name, is_dir, 0 if is_dir else info.st_size,
                                            info.st_mtime_ns))
                    except OSError:
                        continue
        except FileNotFoundError:
            raise
        except PermissionError as exc:
            raise SourceError(f'Нет доступа к папке: {native}') from exc
        return result

    def stat(self, native):
        info = os.stat(native)
        return Entry(os.path.basename(native.rstrip('\\/')) or native,
                     os.path.isdir(native), info.st_size, info.st_mtime_ns)

    def open(self, native):
        return open(native, 'rb')

    def local_path(self, native):
        return native

    def remove(self, native):
        try:
            from send2trash import send2trash
            send2trash(native)
        except ImportError:
            os.remove(native)

    def rename(self, old, new):
        os.makedirs(os.path.dirname(new), exist_ok=True)
        if os.path.exists(new):
            raise SourceError('Файл с таким именем уже есть')
        shutil.move(old, new)

    def makedirs(self, native):
        os.makedirs(native, exist_ok=True)


class SmbDriver(Driver):
    """SMB 2/3 через smbprotocol. Путь: /Шара/папка/файл."""
    kind = 'smb'
    _registered = set()
    _net_used = set()
    _lock = threading.Lock()

    def __init__(self, record):
        super().__init__(record)
        self.host = record['host']
        self.port = int(record.get('port') or 445)

    def _client(self):
        try:
            import smbclient
        except ImportError as exc:
            raise SourceError('Для SMB нужен пакет smbprotocol') from exc
        python_ntlm()
        key = (self.host, self.port, self.record.get('user'), self.record.get('password'))
        with self._lock:
            if key not in self._registered:
                try:
                    smbclient.register_session(
                        self.host, username=self.record.get('user') or None,
                        password=self.record.get('password') or None, port=self.port,
                        auth_protocol='ntlm', connection_timeout=10)
                except Exception as exc:
                    raise SourceError(f'SMB {self.host}: {exc}') from exc
                self._registered.add(key)
        return smbclient

    def unc(self, native):
        parts = [part for part in re.split(r'[\\/]', native) if part]
        return '\\\\' + self.host + (('\\' + '\\'.join(parts)) if parts else '')

    def roots(self):
        shares = list_smb_shares(self.record)
        if self.record.get('share'):
            return [Entry('/' + self.record['share'], True)]
        return [Entry('/' + name, True) for name in shares]

    def test(self):
        if self.record.get('share'):
            self.listdir('/' + self.record['share'])
        else:
            self.roots()
        return True

    def listdir(self, native):
        client = self._client()
        result = []
        try:
            for item in client.scandir(self.unc(native)):
                try:
                    is_dir = item.is_dir()
                    info = item.stat()
                    result.append(Entry(item.name, is_dir, 0 if is_dir else info.st_size,
                                        int(info.st_mtime_ns)))
                except Exception:
                    continue
        except FileNotFoundError:
            raise
        except Exception as exc:
            if 'STATUS_OBJECT_NAME_NOT_FOUND' in str(exc) or 'STATUS_OBJECT_PATH_NOT_FOUND' in str(exc):
                raise FileNotFoundError(native) from exc
            raise SourceError(f'SMB {self.host}: {exc}') from exc
        return result

    def stat(self, native):
        client = self._client()
        try:
            info = client.stat(self.unc(native))
        except Exception as exc:
            if 'NOT_FOUND' in str(exc) or isinstance(exc, FileNotFoundError):
                raise FileNotFoundError(native) from exc
            raise SourceError(f'SMB {self.host}: {exc}') from exc
        import stat as stat_module
        return Entry(pathkeys.name('x:' + native) if native.strip('/') else '/',
                     stat_module.S_ISDIR(info.st_mode), info.st_size, int(info.st_mtime_ns))

    def open(self, native):
        client = self._client()
        try:
            return client.open_file(self.unc(native), mode='rb', share_access='rw')
        except Exception as exc:
            if 'NOT_FOUND' in str(exc):
                raise FileNotFoundError(native) from exc
            raise SourceError(f'SMB {self.host}: {exc}') from exc

    def local_path(self, native):
        """На Windows ядро открывает SMB как обычный путь \\\\хост\\шара\\…"""
        if os.name != 'nt':
            return None
        share = next((part for part in re.split(r'[\\/]', native) if part), '')
        if not share:
            return None
        key = (self.host, share, self.record.get('user'))
        with self._lock:
            if key not in self._net_used:
                if not net_use(self.host, share, self.record.get('user'),
                               self.record.get('password')):
                    return None
                self._net_used.add(key)
        return self.unc(native)

    def remove(self, native):
        self._client().remove(self.unc(native))

    def rename(self, old, new):
        client = self._client()
        folder = new.rsplit('/', 1)[0]
        if folder:
            client.makedirs(self.unc(folder), exist_ok=True)
        client.rename(self.unc(old), self.unc(new))

    def makedirs(self, native):
        self._client().makedirs(self.unc(native), exist_ok=True)


def python_ntlm():
    """NTLM для SMB — из pyspnego, а не из SSPI Windows.

    Ядро запускается из сеанса SSH, и там SSPI не принимает явно переданные
    логин и пароль (SEC_E_UNKNOWN_CREDENTIALS). Встроенная реализация NTLM
    работает одинаково везде: на ядрах и на хабе в контейнере.
    """
    import spnego
    if getattr(spnego.client, 'homecloud_ntlm', False):
        return
    original = spnego.client

    def client(*args, **kwargs):
        kwargs['options'] = kwargs.get('options', spnego.NegotiateOptions.none) \
            | spnego.NegotiateOptions.use_ntlm
        return original(*args, **kwargs)
    client.homecloud_ntlm = True
    spnego.client = client


def net_use(host, share, user, password):
    """Регистрирует учётные данные SMB в сеансе Windows, чтобы открывался UNC-путь."""
    target = f'\\\\{host}\\{share}'
    command = ['net', 'use', target]
    if password:
        command.append(password)
    if user:
        command.append(f'/user:{user}')
    command.append('/persistent:no')
    try:
        result = subprocess.run(command, capture_output=True, timeout=30,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        return False
    output = console_text(result.stdout + result.stderr)
    # 1219 — к этому серверу уже подключены под этим же именем: так тоже годится.
    return result.returncode == 0 or '1219' in output or os.path.isdir(target)


def console_text(raw):
    """Вывод консольных команд Windows: кодовая страница OEM, а не UTF-8."""
    for encoding in ('utf-8', 'cp866', 'cp1251'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', 'replace')


def list_smb_shares(record):
    """Список шар сервера: smbclient -L на Linux, net view на Windows."""
    host = record['host']
    user = record.get('user') or ''
    password = record.get('password') or ''
    if record.get('share'):
        return [record['share']]
    try:
        if os.name == 'nt':
            net_use(host, 'IPC$', user, password)
            output = console_text(subprocess.run(
                ['net', 'view', f'\\\\{host}'], capture_output=True, timeout=30,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout)
            shares, started = [], False
            for line in output.splitlines():
                if line.startswith('---'):
                    started = True
                    continue
                if started and line.strip():
                    match = re.match(r'^(\S(?:.*?\S)?)\s{2,}(Disk|Диск)\b', line)
                    if match:
                        shares.append(match.group(1))
            return shares
        env = {**os.environ, 'PASSWD': password}
        command = ['smbclient', '-g', '-L', f'//{host}', '-p', str(record.get('port') or 445)]
        command += ['-U', user] if user else ['-N']
        output = subprocess.run(command, capture_output=True, text=True, timeout=30, env=env).stdout
        return [line.split('|')[1] for line in output.splitlines()
                if line.startswith('Disk|') and not line.split('|')[1].endswith('$')]
    except (OSError, subprocess.SubprocessError, IndexError):
        return []


class SftpDriver(Driver):
    """SSH/SFTP через paramiko. У Windows OpenSSH диски видны как /D:/папка."""
    kind = 'sftp'

    def __init__(self, record, key_file=None):
        super().__init__(record)
        self.key_file = key_file
        self._client = None
        self._sftp = None
        self._lock = threading.RLock()

    def _connect(self):
        with self._lock:
            if self._sftp is not None:
                transport = self._sftp.get_channel().get_transport()
                if transport is not None and transport.is_active():
                    return self._sftp
                self.close()
            try:
                import paramiko
            except ImportError as exc:
                raise SourceError('Для SSH/SFTP нужен пакет paramiko') from exc
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    self.record['host'], port=int(self.record.get('port') or 22),
                    username=self.record.get('user') or None,
                    password=self.record.get('password') or None,
                    key_filename=self.key_file if self.key_file and Path(self.key_file).is_file()
                    else None,
                    look_for_keys=False, allow_agent=False, timeout=10, banner_timeout=15,
                    auth_timeout=15)
            except Exception as exc:
                raise SourceError(f'SSH {self.record["host"]}: {exc}') from exc
            self._client = client
            self._sftp = client.open_sftp()
            self._sftp.get_channel().settimeout(60)
            return self._sftp

    def close(self):
        with self._lock:
            for item in (self._sftp, self._client):
                try:
                    if item is not None:
                        item.close()
                except Exception:
                    pass
            self._sftp = self._client = None

    def remote(self, native):
        return native or '/'

    def roots(self):
        base = self.record.get('path') or '/'
        return [Entry(base, True)]

    def listdir(self, native):
        import stat as stat_module
        sftp = self._connect()
        try:
            items = sftp.listdir_attr(self.remote(native))
        except FileNotFoundError:
            raise
        except IOError as exc:
            if getattr(exc, 'errno', None) == 2:
                raise FileNotFoundError(native) from exc
            raise SourceError(f'SFTP: {exc}') from exc
        result = []
        for item in items:
            mode = item.st_mode or 0
            if stat_module.S_ISLNK(mode):
                continue
            is_dir = stat_module.S_ISDIR(mode)
            result.append(Entry(item.filename, is_dir, 0 if is_dir else int(item.st_size or 0),
                                int(item.st_mtime or 0) * 1_000_000_000))
        return result

    def stat(self, native):
        import stat as stat_module
        sftp = self._connect()
        try:
            item = sftp.stat(self.remote(native))
        except IOError as exc:
            raise FileNotFoundError(native) from exc
        return Entry(posixpath.basename(self.remote(native)), stat_module.S_ISDIR(item.st_mode or 0),
                     int(item.st_size or 0), int(item.st_mtime or 0) * 1_000_000_000)

    def open(self, native):
        sftp = self._connect()
        try:
            handle = sftp.open(self.remote(native), 'rb')
        except IOError as exc:
            raise FileNotFoundError(native) from exc
        handle.set_pipelined(True)
        return handle

    def remove(self, native):
        self._connect().remove(self.remote(native))

    def makedirs(self, native):
        sftp = self._connect()
        current = ''
        for part in [item for item in self.remote(native).split('/') if item]:
            current += '/' + part
            try:
                sftp.stat(current)
            except IOError:
                sftp.mkdir(current)

    def rename(self, old, new):
        self.makedirs(posixpath.dirname(self.remote(new)))
        self._connect().rename(self.remote(old), self.remote(new))


class DeviceSftpDriver(SftpDriver):
    """Диск Windows-устройства по SSH: D:\\Фото ↔ /D:/Фото."""
    kind = 'device'
    separator = '\\'

    def remote(self, native):
        text = native.replace('\\', '/')
        if re.match(r'^[A-Za-z]:', text):
            return '/' + text
        return text or '/'

    def roots(self):
        found = []
        try:
            for item in self._connect().listdir_attr('/'):
                if re.match(r'^[A-Za-z]:$', item.filename):
                    found.append(Entry(item.filename + '\\', True))
        except IOError as exc:
            raise SourceError(f'SFTP: {exc}') from exc
        return found

    def join(self, folder, name):
        return folder.rstrip('\\/') + '\\' + name if folder else name


class DeviceAgentDriver(Driver):
    """Диск устройства через службу ядра на нём (если SSH не настроен)."""
    kind = 'device'
    separator = '\\'

    def __init__(self, record, url, token):
        super().__init__(record)
        self.url = url.rstrip('/')
        self.token = token

    def _get(self, path, timeout=20):
        request = Request(self.url + path, headers={'X-Local-Token': self.token})
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception as exc:
            if getattr(exc, 'code', 0) == 404:
                raise FileNotFoundError(path) from exc
            raise SourceError(f'Ядро {self.url} недоступно: {exc}') from exc

    def roots(self):
        return [Entry(item['path'], True) for item in self._get('/api/device/browse').get(
            'directories', [])]

    def listdir(self, native):
        data = self._get('/api/core/list?path=' + quote(native, safe=''))
        return [Entry(item['name'], item['dir'], item.get('size', 0), item.get('mtime_ns', 0))
                for item in data.get('entries', [])]

    def stat(self, native):
        item = self._get('/api/core/stat?path=' + quote(native, safe=''))
        return Entry(item['name'], item['dir'], item.get('size', 0), item.get('mtime_ns', 0))

    def open(self, native):
        info = self.stat(native)

        def fetch(start, length):
            request = Request(self.url + '/api/core/read?path=' + quote(native, safe=''),
                              headers={'X-Local-Token': self.token,
                                       'Range': f'bytes={start}-{start + length - 1}'})
            with urlopen(request, timeout=60) as response:
                return response.read()
        return buffered(RangeReader(info.size, fetch, native))

    def join(self, folder, name):
        return folder.rstrip('\\/') + '\\' + name if folder else name


class FtpDriver(Driver):
    """FTP/FTPS через ftplib. На каждый поток — своё соединение."""
    kind = 'ftp'

    def __init__(self, record):
        super().__init__(record)
        self.local = threading.local()

    def _ftp(self):
        ftp = getattr(self.local, 'ftp', None)
        if ftp is not None:
            try:
                ftp.voidcmd('NOOP')
                return ftp
            except (ftplib.all_errors):
                self.local.ftp = None
        ftp = ftplib.FTP_TLS(timeout=30) if self.record.get('secure') else ftplib.FTP(timeout=30)
        ftp.encoding = 'utf-8'
        try:
            ftp.connect(self.record['host'], int(self.record.get('port') or 21))
            ftp.login(self.record.get('user') or 'anonymous', self.record.get('password') or '')
            if self.record.get('secure'):
                ftp.prot_p()
        except ftplib.all_errors as exc:
            raise SourceError(f'FTP {self.record["host"]}: {exc}') from exc
        self.local.ftp = ftp
        return ftp

    def close(self):
        ftp = getattr(self.local, 'ftp', None)
        if ftp is not None:
            try:
                ftp.quit()
            except ftplib.all_errors:
                pass
            self.local.ftp = None

    @staticmethod
    def _stamp(value):
        try:
            parsed = time.strptime(value[:14], '%Y%m%d%H%M%S')
            import calendar
            return calendar.timegm(parsed) * 1_000_000_000
        except (ValueError, TypeError):
            return 0

    def listdir(self, native):
        ftp = self._ftp()
        result = []
        try:
            for name, facts in ftp.mlsd(native or '/', facts=['type', 'size', 'modify']):
                if name in {'.', '..'} or facts.get('type') in {'cdir', 'pdir'}:
                    continue
                is_dir = facts.get('type') == 'dir'
                result.append(Entry(name, is_dir, int(facts.get('size') or 0),
                                    self._stamp(facts.get('modify'))))
            return result
        except ftplib.error_perm as exc:
            if str(exc).startswith('550'):
                raise FileNotFoundError(native) from exc
            if not str(exc).startswith('500'):
                raise SourceError(f'FTP: {exc}') from exc
        # Сервер без MLSD: имена через NLST, дальше SIZE/MDTM по каждому.
        for name in ftp.nlst(native or '/'):
            name = posixpath.basename(name.rstrip('/'))
            if name in {'.', '..'}:
                continue
            path = posixpath.join(native or '/', name)
            try:
                size = ftp.size(path)
                modified = self._stamp(ftp.voidcmd('MDTM ' + path)[4:].strip())
                result.append(Entry(name, False, int(size or 0), modified))
            except ftplib.error_perm:
                result.append(Entry(name, True))
        return result

    def stat(self, native):
        ftp = self._ftp()
        name = posixpath.basename(native.rstrip('/')) or '/'
        try:
            # MLST отвечает сразу всем: тип, размер, время.
            response = ftp.sendcmd('MLST ' + native)
            facts = {}
            for line in response.splitlines()[1:-1]:
                for part in line.strip().split(' ', 1)[0].split(';'):
                    if '=' in part:
                        key, value = part.split('=', 1)
                        facts[key.lower()] = value
            if facts:
                return Entry(name, facts.get('type', '').lower() in {'dir', 'cdir', 'pdir'},
                             int(facts.get('size') or 0), self._stamp(facts.get('modify')))
        except ftplib.error_perm as exc:
            if str(exc).startswith('550'):
                raise FileNotFoundError(native) from exc
        try:
            # SIZE многие серверы отдают только в двоичном режиме.
            ftp.voidcmd('TYPE I')
            size = ftp.size(native)
            modified = self._stamp(ftp.voidcmd('MDTM ' + native)[4:].strip())
            return Entry(name, False, int(size or 0), modified)
        except ftplib.error_perm:
            try:
                ftp.cwd(native)
                return Entry(posixpath.basename(native.rstrip('/')) or '/', True)
            except ftplib.error_perm as exc:
                raise FileNotFoundError(native) from exc

    def open(self, native):
        info = self.stat(native)

        def fetch(start, length):
            ftp = self._ftp()
            ftp.voidcmd('TYPE I')
            connection = ftp.transfercmd('RETR ' + native, rest=start or None)
            chunks, left = [], length
            try:
                while left > 0:
                    chunk = connection.recv(min(left, 1024 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    left -= len(chunk)
            finally:
                connection.close()
                try:
                    ftp.voidresp()
                except ftplib.all_errors:
                    # Прерванная передача — ответ 426/451; соединение лучше открыть заново.
                    self.local.ftp = None
            return b''.join(chunks)
        return buffered(RangeReader(info.size, fetch, native), 4 * 1024 * 1024)

    def remove(self, native):
        self._ftp().delete(native)

    def makedirs(self, native):
        ftp = self._ftp()
        current = ''
        for part in [item for item in native.split('/') if item]:
            current += '/' + part
            try:
                ftp.mkd(current)
            except ftplib.error_perm:
                pass

    def rename(self, old, new):
        self.makedirs(posixpath.dirname(new))
        self._ftp().rename(old, new)


class WebdavDriver(Driver):
    """WebDAV: PROPFIND для списков, GET с Range для чтения."""
    kind = 'webdav'

    def __init__(self, record):
        super().__init__(record)
        self.https = bool(record.get('secure')) or int(record.get('port') or 443) == 443

    def _connection(self):
        factory = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return factory(self.record['host'], int(self.record.get('port') or (443 if self.https else 80)),
                       timeout=60)

    def _headers(self, extra=None):
        headers = dict(extra or {})
        if self.record.get('user'):
            import base64
            token = base64.b64encode(
                f'{self.record["user"]}:{self.record.get("password") or ""}'.encode()).decode()
            headers['Authorization'] = 'Basic ' + token
        return headers

    @staticmethod
    def _href(native):
        return quote(native or '/', safe='/')

    def _request(self, method, native, headers=None, body=None):
        connection = self._connection()
        try:
            connection.request(method, self._href(native), body=body, headers=self._headers(headers))
            response = connection.getresponse()
            data = response.read()
            return response.status, response, data
        except OSError as exc:
            raise SourceError(f'WebDAV {self.record["host"]}: {exc}') from exc
        finally:
            connection.close()

    def _propfind(self, native, depth):
        status, _response, data = self._request(
            'PROPFIND', native, {'Depth': str(depth), 'Content-Type': 'application/xml'},
            b'<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:resourcetype/>'
            b'<d:getcontentlength/><d:getlastmodified/></d:prop></d:propfind>')
        if status == 404:
            raise FileNotFoundError(native)
        if status >= 400:
            raise SourceError(f'WebDAV ответил {status}')
        from email.utils import parsedate_to_datetime
        found = []
        for response in ElementTree.fromstring(data).findall('{DAV:}response'):
            href = unquote(urlsplit(response.findtext('{DAV:}href') or '').path)
            prop = response.find('.//{DAV:}prop')
            if prop is None:
                continue
            is_dir = prop.find('{DAV:}resourcetype/{DAV:}collection') is not None
            size = int(prop.findtext('{DAV:}getcontentlength') or 0)
            try:
                stamp = int(parsedate_to_datetime(prop.findtext('{DAV:}getlastmodified')).timestamp()
                            * 1_000_000_000)
            except (TypeError, ValueError):
                stamp = 0
            found.append((href, Entry(posixpath.basename(href.rstrip('/')) or '/', is_dir, size,
                                      stamp)))
        return found

    def listdir(self, native):
        wanted = (native or '/').rstrip('/')
        return [entry for href, entry in self._propfind(native, 1)
                if href.rstrip('/') != wanted]

    def stat(self, native):
        found = self._propfind(native, 0)
        if not found:
            raise FileNotFoundError(native)
        return found[0][1]

    def open(self, native):
        info = self.stat(native)

        def fetch(start, length):
            status, _response, data = self._request(
                'GET', native, {'Range': f'bytes={start}-{start + length - 1}'})
            if status == 200:
                return data[start:start + length]
            if status != 206:
                raise SourceError(f'WebDAV ответил {status}')
            return data
        return buffered(RangeReader(info.size, fetch, native), 4 * 1024 * 1024)

    def remove(self, native):
        status, _response, _data = self._request('DELETE', native)
        if status >= 400:
            raise SourceError(f'WebDAV ответил {status}')

    def makedirs(self, native):
        current = ''
        for part in [item for item in native.split('/') if item]:
            current += '/' + part
            self._request('MKCOL', current)

    def rename(self, old, new):
        self.makedirs(posixpath.dirname(new))
        scheme = 'https' if self.https else 'http'
        destination = f'{scheme}://{self.record["host"]}:{self.record.get("port")}{self._href(new)}'
        status, _response, _data = self._request('MOVE', old, {'Destination': destination,
                                                               'Overwrite': 'F'})
        if status >= 400:
            raise SourceError(f'WebDAV ответил {status}')


# ---------- выбор драйвера ----------

class Access:
    """Как этой машине добраться до источников.

    `device_id` — id ядра, на котором работаем (у хаба пусто): свой диск
    читается напрямую. `device_route(id)` — для хаба: как дойти до чужого
    диска ({'ssh': {...}, 'agent': (url, token)}). Драйверы кэшируются: у SFTP
    и SMB дорогое подключение.
    """

    def __init__(self, records=(), device_id='', device_route=None, ssh_key=None):
        self.records = {row['id']: row for row in records}
        self.device_id = device_id
        self.device_route = device_route
        self.ssh_key = ssh_key
        self.drivers = {}
        self.lock = threading.RLock()

    def update(self, records):
        with self.lock:
            fresh = {row['id']: row for row in records}
            for source_id, driver in list(self.drivers.items()):
                if fresh.get(source_id) != self.records.get(source_id):
                    driver.close()
                    self.drivers.pop(source_id, None)
            self.records = fresh

    def record(self, source_id):
        record = self.records.get(source_id)
        if record is None:
            raise SourceError(f'Источник «{source_id}» не найден')
        return record

    def driver(self, source_id):
        with self.lock:
            driver = self.drivers.get(source_id)
            if driver is None:
                driver = self.drivers[source_id] = self._make(self.record(source_id))
            return driver

    def _make(self, record):
        kind = record['type']
        if kind == 'local':
            return LocalDriver(record)
        if kind == 'device':
            if record.get('device') == self.device_id:
                return LocalDriver(record)
            route = self.device_route(record['device']) if self.device_route else None
            if not route:
                raise SourceError(f'Устройство «{record.get("device")}» не подключено к хабу')
            if route.get('ssh'):
                ssh = route['ssh']
                return DeviceSftpDriver({**record, 'host': ssh['host'], 'port': ssh.get('port', 22),
                                         'user': ssh['user'], 'password': ssh.get('password', '')},
                                        key_file=self.ssh_key)
            if route.get('agent'):
                return DeviceAgentDriver(record, *route['agent'])
            raise SourceError('Диск устройства недоступен: нет ни SSH, ни ядра в сети')
        if kind == 'smb':
            return SmbDriver(record)
        if kind == 'sftp':
            return SftpDriver(record, key_file=self.ssh_key)
        if kind == 'ftp':
            return FtpDriver(record)
        if kind == 'webdav':
            return WebdavDriver(record)
        raise SourceError(f'Неизвестный тип источника: {kind}')

    def resolve(self, key):
        source, native = pathkeys.split(key)
        if not source:
            return LocalDriver(), native
        return self.driver(source), native

    def close(self):
        with self.lock:
            for driver in self.drivers.values():
                driver.close()
            self.drivers.clear()


# ---------- путь к файлу для этапов обработки ----------

_resolver = None
_env_cache = {}


def set_resolver(function):
    """Служба ядра подставляет свой способ: копия во временной папке и т. п."""
    global _resolver
    _resolver = function


def _core_config():
    folder = os.environ.get('HOMECLOUD_CORE_DIR')
    if not folder:
        return None
    path = Path(folder) / 'sources.json'
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    cached = _env_cache.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    _env_cache[str(path)] = (stamp, value)
    return value


_core_access = None


def core_access():
    """Драйверы источников в процессе этапа: из sources.json, что оставило ядро."""
    global _core_access
    config = _core_config()
    if config is None:
        raise SourceError('Ядро не передало настройки источников (HOMECLOUD_CORE_DIR)')
    if _core_access is None:
        _core_access = Access(config.get('sources', []), device_id=config.get('core', ''))
    else:
        _core_access.update(config.get('sources', []))
    return _core_access


def local(key):
    """Путь к файлу на этой машине. Для старых путей без источника — сам путь."""
    key = str(key)
    source, native = pathkeys.split(key)
    if not source:
        return key
    if _resolver is not None:
        return _resolver(key)
    config = _core_config()
    if config is None:
        raise SourceError(f'Не знаю, где взять {key}: ядро не передало настройки источников')
    record = next((row for row in config.get('sources', []) if row['id'] == source), None)
    if record is None:
        raise SourceError(f'Источник «{source}» не найден')
    if record['type'] == 'device' and record.get('device') == config.get('core'):
        return native
    if record['type'] == 'local' and not config.get('core'):
        return native
    if record['type'] == 'smb' and os.name == 'nt' and config.get('unc', {}).get(source):
        return SmbDriver(record).unc(native)
    agent = config.get('agent')
    if not agent:
        raise SourceError(f'Нет службы ядра, чтобы забрать {key}')
    request = Request(agent['url'].rstrip('/') + '/api/core/stage?key=' + quote(key, safe=''),
                      headers={'X-Local-Token': agent['token']})
    try:
        with urlopen(request, timeout=3600) as response:
            return json.loads(response.read())['path']
    except Exception as exc:
        raise SourceError(f'Не удалось забрать {key}: {exc}') from exc


class Stage:
    """Временные копии файлов с удалённых источников — для этапов обработки.

    Копия живёт, пока место под папку не кончится: старые удаляются первыми.
    """

    def __init__(self, folder, access, limit_bytes=20 * 1024 ** 3):
        self.folder = Path(folder)
        self.access = access
        self.limit = limit_bytes
        self.lock = threading.Lock()
        self.busy = {}

    def path(self, key):
        source, native = pathkeys.split(key)
        driver, native = self.access.resolve(key)
        direct = driver.local_path(native)
        if direct:
            return direct
        digest = hashlib.sha1(key.encode('utf-8')).hexdigest()
        target = self.folder / digest[:2] / (digest + pathkeys.suffix(key))
        info = driver.stat(native)
        with self.lock:
            event = self.busy.get(digest)
            if event is None:
                event = self.busy[digest] = threading.Event()
                owner = True
            else:
                owner = False
        if not owner:
            event.wait(3600)
            return str(target)
        try:
            try:
                current = target.stat()
                if current.st_size == info.size:
                    os.utime(target, None)
                    return str(target)
            except OSError:
                pass
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.part')
            with driver.open(native) as stream, open(temporary, 'wb') as output:
                shutil.copyfileobj(stream, output, 4 * 1024 * 1024)
            os.replace(temporary, target)
            self.trim()
            return str(target)
        finally:
            with self.lock:
                self.busy.pop(digest, None)
            event.set()

    def trim(self):
        files = []
        total = 0
        for path in self.folder.rglob('*'):
            if path.is_file() and not path.name.endswith('.part'):
                info = path.stat()
                files.append((info.st_atime, info.st_size, path))
                total += info.st_size
        files.sort()
        for _atime, size, path in files:
            if total <= self.limit:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                pass

    def clear(self):
        shutil.rmtree(self.folder, ignore_errors=True)
