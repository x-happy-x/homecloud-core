"""Соединение с каталогом: свой SQLite или каталог хаба по сети.

Этапы обработки и службы открывают каталог через `connect(folder)`. Если в
папке лежит `hub.json` (см. hublink.py), соединение удалённое: каждый вызов
уходит на хаб, где стоит настоящий SQLite, и выполняется там на отдельном
соединении этого клиента. Поэтому смысл транзакций, блокировок и ошибок —
ровно тот же, что у sqlite3: хаб просто держит соединение за клиента.

Повторяется та часть sqlite3.Connection, что нужна скриптам: execute,
executemany, executescript, курсоры (fetchone/fetchmany/fetchall, перебор),
commit/rollback, `with db:`, lastrowid/rowcount/description, in_transaction.
Ошибки приходят теми же классами sqlite3, так что `except sqlite3.Error`
работает как раньше.

Здесь же — серверная половина (Sessions), чтобы протокол жил в одном месте.
Только стандартная библиотека.
"""
import base64
from collections import deque
from collections.abc import Mapping
import itertools
import math
from pathlib import Path
import sqlite3
import threading
import time

import hublink

FIRST_PAGE = 1000
PAGE = 5000
MANY_CHUNK = 1000


# ---------- кодирование значений ----------

def encode_value(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'$b': base64.b64encode(bytes(value)).decode('ascii')}
    if isinstance(value, float) and not math.isfinite(value):
        return {'$f': repr(value)}
    return value


def decode_value(value):
    if isinstance(value, dict):
        if '$b' in value:
            return base64.b64decode(value['$b'])
        if '$f' in value:
            return float(value['$f'])
    return value


def _plain(value):
    """Как sqlite3: bool/int/float-наследники приводятся, чужие типы — ошибка."""
    if value is None or isinstance(value, (str, bytes, bytearray, memoryview)):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    raise sqlite3.ProgrammingError(
        f'Error binding parameter - unsupported type {type(value).__name__}')


def encode_params(params):
    if params is None:
        return []
    if isinstance(params, Mapping):
        return {'$map': {str(key): encode_value(_plain(value)) for key, value in params.items()}}
    return [encode_value(_plain(value)) for value in params]


def decode_params(params):
    if isinstance(params, dict) and '$map' in params:
        return {key: decode_value(value) for key, value in params['$map'].items()}
    return [decode_value(value) for value in params or []]


def encode_rows(rows):
    return [[encode_value(value) for value in row] for row in rows]


def decode_rows(rows):
    return [tuple(decode_value(value) for value in row) for row in rows]


def _raise(error):
    kind = getattr(sqlite3, str(error.get('type') or ''), None)
    if not (isinstance(kind, type) and issubclass(kind, (sqlite3.Error, sqlite3.Warning))):
        kind = sqlite3.OperationalError
    raise kind(error.get('message') or 'Ошибка каталога на хабе')


# ---------- клиент ----------

def connect(folder, timeout=30, check_same_thread=True, **_ignored):
    """sqlite3.Connection к folder/catalog.sqlite или удалённое соединение с хабом."""
    folder = Path(folder)
    link = hublink.link_for(folder)
    if link is not None:
        return RemoteConnection(link, timeout)
    return sqlite3.connect(folder / 'catalog.sqlite', timeout=timeout,
                           check_same_thread=check_same_thread)


def is_remote(folder):
    return hublink.is_remote(folder)


class RemoteCursor:
    arraysize = 1

    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self.rowcount = -1
        self.lastrowid = None
        self._rows = deque()
        self._cursor = None
        self._done = True

    def _load(self, reply):
        names = reply.get('description')
        self.description = (tuple((name, None, None, None, None, None, None) for name in names)
                            if names is not None else None)
        self.rowcount = reply.get('rowcount', -1)
        self.lastrowid = reply.get('lastrowid')
        self._rows = deque(decode_rows(reply.get('rows') or []))
        self._cursor = reply.get('cursor')
        self._done = bool(reply.get('done', True))

    def execute(self, sql, params=()):
        self._release()
        self._load(self.connection._call({'op': 'execute', 'sql': sql,
                                          'params': encode_params(params),
                                          'first': FIRST_PAGE}))
        return self

    def executemany(self, sql, seq_of_params):
        self._release()
        total = 0
        last = None
        iterator = iter(seq_of_params)
        while True:
            chunk = [encode_params(params) for params in itertools.islice(iterator, MANY_CHUNK)]
            if not chunk:
                break
            reply = self.connection._call({'op': 'executemany', 'sql': sql, 'rows': chunk})
            total += max(0, reply.get('rowcount') or 0)
            last = reply.get('lastrowid', last)
        self.description, self.rowcount, self.lastrowid = None, total, last
        self._rows, self._cursor, self._done = deque(), None, True
        return self

    def executescript(self, sql):
        self._release()
        self.connection._call({'op': 'executescript', 'sql': sql})
        self.description, self._rows, self._cursor, self._done = None, deque(), None, True
        return self

    def _more(self, size):
        if self._done or self._cursor is None:
            return False
        reply = self.connection._call({'op': 'fetch', 'cursor': self._cursor, 'size': size})
        self._rows.extend(decode_rows(reply.get('rows') or []))
        self._done = bool(reply.get('done', True))
        if self._done:
            self._cursor = None
        return True

    def fetchone(self):
        if not self._rows:
            self._more(PAGE)
        return self._rows.popleft() if self._rows else None

    def fetchmany(self, size=None):
        size = self.arraysize if size is None else size
        while len(self._rows) < size and self._more(max(size, PAGE)):
            pass
        return [self._rows.popleft() for _ in range(min(size, len(self._rows)))]

    def fetchall(self):
        while self._more(PAGE):
            pass
        result, self._rows = list(self._rows), deque()
        return result

    def __iter__(self):
        return self

    def __next__(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

    def _release(self):
        if self._cursor is not None:
            self.connection._forget.append(self._cursor)
            self._cursor = None
        self._rows, self._done = deque(), True

    def close(self):
        self._release()

    def __del__(self):
        try:
            self._release()
        except Exception:
            pass


class RemoteConnection:
    """Соединение с каталогом на хабе. Поведение — как у sqlite3.Connection."""

    def __init__(self, link, timeout=30):
        self.link = link
        self.timeout = float(timeout or 30)
        self.row_factory = None
        self.isolation_level = ''
        self.in_transaction = False
        self.total_changes = 0
        self._forget = []
        self._lock = threading.RLock()
        self._session = None
        self._closed = False
        self._open()

    def _open(self):
        reply = self._send({'op': 'open', 'timeout': self.timeout})
        self._session = reply['session']

    def _send(self, payload):
        reply = self.link.json('POST', '/db', payload, timeout=self.timeout + 300)
        if reply.get('error'):
            _raise(reply['error'])
        return reply

    def _call(self, payload):
        with self._lock:
            if self._closed:
                raise sqlite3.ProgrammingError('Cannot operate on a closed database.')
            if self.row_factory is not None:
                raise sqlite3.NotSupportedError('row_factory у каталога хаба не поддерживается')
            payload = {**payload, 'session': self._session}
            if self._forget:
                payload['forget'], self._forget = self._forget, []
            try:
                reply = self._send(payload)
            except sqlite3.OperationalError as exc:
                # Хаб перезапустился: сессии нет. Вне транзакции её можно
                # просто открыть заново — никакое состояние не теряется.
                if 'session-lost' not in str(exc) or self.in_transaction:
                    raise
                self._open()
                payload['session'] = self._session
                payload.pop('forget', None)
                reply = self._send(payload)
            self.in_transaction = bool(reply.get('tx', False))
            self.total_changes = int(reply.get('changes', self.total_changes) or 0)
            return reply

    def cursor(self):
        return RemoteCursor(self)

    def execute(self, sql, params=()):
        return RemoteCursor(self).execute(sql, params)

    def executemany(self, sql, seq_of_params):
        return RemoteCursor(self).executemany(sql, seq_of_params)

    def executescript(self, sql):
        return RemoteCursor(self).executescript(sql)

    def commit(self):
        self._call({'op': 'commit'})

    def rollback(self):
        self._call({'op': 'rollback'})

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                self._send({'op': 'close', 'session': self._session})
            except Exception:
                pass
            self._closed = True

    def create_function(self, *args, **kwargs):
        raise sqlite3.NotSupportedError('Свои функции SQL у каталога хаба не поддерживаются')

    def backup(self, *args, **kwargs):
        raise sqlite3.NotSupportedError('Резервную копию каталога делает сам хаб')

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.commit()
        else:
            self.rollback()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------- сервер (живёт в хабе) ----------

class _Session:
    def __init__(self, connection, owner):
        self.connection = connection
        self.owner = owner
        self.cursors = {}
        self.lock = threading.Lock()
        self.used = time.monotonic()
        self.counter = itertools.count(1)


class Sessions:
    """Соединения клиентов с каталогом хаба: у каждого своё, как у процесса.

    Простаивающая сессия закрывается через `idle` секунд, а забытая открытая
    транзакция откатывается через `tx_idle` — иначе упавшее посреди записи
    ядро держало бы каталог заблокированным.
    """

    def __init__(self, database, idle=6 * 3600, tx_idle=300):
        self.database = Path(database)
        self.idle = idle
        self.tx_idle = tx_idle
        self.sessions = {}
        self.lock = threading.Lock()
        self.counter = itertools.count(1)
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()

    def _reap_loop(self):
        while True:
            time.sleep(30)
            try:
                self.reap()
            except Exception:
                pass

    def reap(self):
        now = time.monotonic()
        with self.lock:
            items = list(self.sessions.items())
        for key, session in items:
            if not session.lock.acquire(blocking=False):
                continue
            try:
                idle = now - session.used
                if idle > self.idle:
                    self._close(key, session)
                elif session.connection.in_transaction and idle > self.tx_idle:
                    session.connection.rollback()
                    session.cursors.clear()
            finally:
                session.lock.release()

    def _close(self, key, session):
        session.cursors.clear()
        try:
            session.connection.close()
        except sqlite3.Error:
            pass
        with self.lock:
            self.sessions.pop(key, None)

    def close_owner(self, owner):
        """Ядро перезапустилось — его прежние соединения больше не нужны."""
        with self.lock:
            items = [(key, session) for key, session in self.sessions.items()
                     if session.owner == owner]
        for key, session in items:
            with session.lock:
                if session.connection.in_transaction:
                    session.connection.rollback()
                self._close(key, session)

    def stats(self):
        with self.lock:
            return {'sessions': len(self.sessions),
                    'owners': sorted({session.owner for session in self.sessions.values()})}

    def handle(self, payload, owner=''):
        try:
            return self._handle(payload, owner)
        except (sqlite3.Error, sqlite3.Warning) as exc:
            return {'error': {'type': type(exc).__name__, 'message': str(exc)}}
        except (ValueError, TypeError, OverflowError) as exc:
            return {'error': {'type': 'ProgrammingError', 'message': str(exc)}}

    def _handle(self, payload, owner):
        op = payload.get('op')
        if op == 'open':
            timeout = min(max(float(payload.get('timeout') or 30), 1.0), 600.0)
            connection = sqlite3.connect(self.database, timeout=timeout,
                                         check_same_thread=False)
            key = f'{owner}:{next(self.counter)}:{time.time_ns()}'
            with self.lock:
                self.sessions[key] = _Session(connection, owner)
            return {'session': key}
        with self.lock:
            session = self.sessions.get(payload.get('session'))
        if session is None:
            return {'error': {'type': 'OperationalError',
                              'message': 'session-lost: соединение с каталогом хаба закрыто'}}
        with session.lock:
            session.used = time.monotonic()
            for cursor_id in payload.get('forget') or ():
                session.cursors.pop(cursor_id, None)
            connection = session.connection
            reply = self._run(session, connection, op, payload, owner)
            if op != 'close':
                reply['tx'] = connection.in_transaction
                reply['changes'] = connection.total_changes
            return reply

    def _run(self, session, connection, op, payload, owner):
        if op == 'execute':
            cursor = connection.cursor()
            cursor.execute(payload.get('sql') or '', decode_params(payload.get('params')))
            first = min(max(int(payload.get('first') or FIRST_PAGE), 1), 20000)
            rows = cursor.fetchmany(first) if cursor.description is not None else []
            done = cursor.description is None or len(rows) < first
            cursor_id = None
            if not done:
                cursor_id = next(session.counter)
                session.cursors[cursor_id] = cursor
                if len(session.cursors) > 64:
                    session.cursors.pop(min(session.cursors))
            return {'description': ([item[0] for item in cursor.description]
                                    if cursor.description is not None else None),
                    'rowcount': cursor.rowcount, 'lastrowid': cursor.lastrowid,
                    'rows': encode_rows(rows), 'cursor': cursor_id, 'done': done}
        if op == 'fetch':
            cursor = session.cursors.get(payload.get('cursor'))
            if cursor is None:
                return {'rows': [], 'done': True}
            size = min(max(int(payload.get('size') or PAGE), 1), 20000)
            rows = cursor.fetchmany(size)
            done = len(rows) < size
            if done:
                session.cursors.pop(payload.get('cursor'), None)
            return {'rows': encode_rows(rows), 'done': done}
        if op == 'executemany':
            cursor = connection.executemany(
                payload.get('sql') or '', [decode_params(row) for row in payload.get('rows') or []])
            return {'rowcount': cursor.rowcount, 'lastrowid': cursor.lastrowid}
        if op == 'executescript':
            connection.executescript(payload.get('sql') or '')
            return {}
        if op == 'commit':
            connection.commit()
            return {}
        if op == 'rollback':
            connection.rollback()
            session.cursors.clear()
            return {}
        if op == 'close':
            key = payload.get('session')
            session.cursors.clear()
            try:
                connection.close()
            finally:
                with self.lock:
                    self.sessions.pop(key, None)
            return {'ok': True}
        raise ValueError(f'Неизвестная операция каталога: {op!r}')
