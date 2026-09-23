"""Связь ядра с хабом: адрес, токен и HTTP-запросы с keep-alive.

Ядро (core) — компьютер с видеокартой: сам каталог у него не хранится, он
живёт на хабе (VM homecloud). В рабочей папке ядра лежит `hub.json`:

    {"url": "http://192.168.99.20:18401", "token": "…", "core": "pc-x"}

Всё, что этапы пишут в каталог, и всё, что читают из него, идёт через этот
адрес. Модуль — только стандартная библиотека: его импортируют окружения
этапов (vision-venv, audio-venv…), в которых нет ничего лишнего.
"""
import http.client
import json
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import quote, urlsplit

CONFIG_NAME = 'hub.json'
_configs = {}
_links = {}
_lock = threading.Lock()


class HubError(RuntimeError):
    def __init__(self, message, status=0):
        super().__init__(message)
        self.status = status


def config(folder):
    """Настройки связи из рабочей папки ядра; None — каталог локальный."""
    if folder is None:
        return None
    path = Path(folder) / CONFIG_NAME
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    cached = _configs.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not value.get('url') or not value.get('token'):
        return None
    _configs[str(path)] = (stamp, value)
    return value


def is_remote(folder):
    return config(folder) is not None


def link_for(folder):
    value = config(folder)
    if value is None:
        return None
    key = (value['url'], value['token'])
    with _lock:
        link = _links.get(key)
        if link is None:
            link = _links[key] = HubLink(value['url'], value['token'], value.get('core', ''))
        return link


class HubLink:
    """HTTP к порту связи хаба. Соединение своё у каждого потока и живёт долго."""

    def __init__(self, url, token, core=''):
        parts = urlsplit(url)
        if parts.scheme != 'http' or not parts.hostname:
            raise ValueError(f'Адрес хаба должен быть http://хост:порт, а не {url!r}')
        self.url = url.rstrip('/')
        self.host = parts.hostname
        self.port = parts.port or 80
        self.token = token
        self.core = core
        self.local = threading.local()

    def _connection(self, timeout):
        connection = getattr(self.local, 'connection', None)
        if connection is None:
            connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
            self.local.connection = connection
            self.local.fresh = True
        else:
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
            self.local.fresh = False
        return connection

    def _drop(self):
        connection = getattr(self.local, 'connection', None)
        self.local.connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def request(self, method, path, body=None, headers=None, timeout=120):
        """(статус, заголовки, тело). Сорванное простаивавшее соединение — ещё раз."""
        headers = {'X-Core-Token': self.token, 'X-Core-Id': self.core, **(headers or {})}
        if body is not None and not isinstance(body, (bytes, bytearray)):
            body = bytes(body)
        for attempt in range(3):
            connection = self._connection(timeout)
            reused = not self.local.fresh
            try:
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                data = response.read()
                if response.getheader('Connection', '').lower() == 'close':
                    self._drop()
                return response.status, {k.lower(): v for k, v in response.getheaders()}, data
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError,
                    http.client.CannotSendRequest, http.client.ResponseNotReady) as exc:
                self._drop()
                # Сервер закрыл простаивавшее соединение раньше, чем мы его взяли:
                # запрос до него не дошёл, повторять безопасно. Свежее соединение
                # так не рвётся — значит, хаб действительно недоступен.
                if not reused:
                    raise HubError(f'Хаб недоступен: {exc}') from exc
            except (socket.timeout, TimeoutError) as exc:
                self._drop()
                raise HubError(f'Хаб не ответил за {timeout} с') from exc
            except OSError as exc:
                self._drop()
                if attempt == 2:
                    raise HubError(f'Хаб недоступен ({self.host}:{self.port}): {exc}') from exc
                time.sleep(1 + attempt * 2)
        raise HubError('Хаб недоступен')

    def json(self, method, path, payload=None, timeout=120):
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        status, _headers, data = self.request(method, path, body, headers, timeout)
        try:
            value = json.loads(data or b'{}')
        except json.JSONDecodeError:
            value = {'error': data[:300].decode('utf-8', 'replace')}
        if status >= 400:
            raise HubError(value.get('error') or f'Хаб ответил {status}', status)
        return value

    def put_file(self, relative, data, timeout=120):
        status, _headers, body = self.request(
            'PUT', '/files/' + quote(relative.replace('\\', '/')), data,
            {'Content-Type': 'application/octet-stream'}, timeout)
        if status >= 400:
            raise HubError(body[:300].decode('utf-8', 'replace') or f'Хаб ответил {status}', status)

    def get_file(self, relative, timeout=120):
        status, _headers, body = self.request(
            'GET', '/files/' + quote(relative.replace('\\', '/')), None, None, timeout)
        if status == 404:
            raise FileNotFoundError(relative)
        if status >= 400:
            raise HubError(body[:300].decode('utf-8', 'replace') or f'Хаб ответил {status}', status)
        return body

    def delete_file(self, relative, timeout=60):
        self.request('DELETE', '/files/' + quote(relative.replace('\\', '/')), None, None, timeout)


def environment_folder():
    """Рабочая папка ядра, которую задание передаёт этапам через окружение."""
    value = os.environ.get('HOMECLOUD_CORE_DIR')
    return Path(value) if value else None
