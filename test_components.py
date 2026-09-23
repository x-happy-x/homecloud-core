"""Окружения и модели ядра: состояние, пропуск и копия модели с другого ядра."""
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import components
import core
import envs

QWEN = 'huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct'


def make_root(base, with_model):
    root = Path(base)
    (root / 'models' / 'cv-models').mkdir(parents=True)
    if with_model:
        snapshot = root / 'models' / 'cv-models' / QWEN / 'snapshots' / 'abc'
        snapshot.mkdir(parents=True)
        (snapshot / 'model.safetensors').write_bytes(b'w' * 5000)
        (snapshot / 'config.json').write_text('{}', encoding='utf-8')
    return root


class ComponentsTest(unittest.TestCase):
    def setUp(self):
        # Общий C:/cv-models этой машины в тестах не трогаем.
        patch = mock.patch.object(envs, 'SHARED_MODELS', Path(tempfile.gettempdir()) / 'no-such-cv-models')
        patch.start()
        self.addCleanup(patch.stop)

    def test_status_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            state = components.Components(make_root(temp, True))
            models = {item['id']: item for item in state.status()['models']}
            self.assertTrue(models['qwen3-vl']['installed'])
            self.assertEqual(models['qwen3-vl']['bytes'], 5002)
            self.assertFalse(models['whisper']['installed'])
            paths = sorted(entry['path'] for entry in state.manifest('qwen3-vl')['files'])
            self.assertEqual(paths, [f'hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/abc/{name}'
                                     for name in ('config.json', 'model.safetensors')])

    def test_paths_stay_inside_model(self):
        item = components.find('qwen3-vl')
        for bad in ('../x', '/etc/passwd', 'C:/x', 'hub/models--other/x',
                    'hub/models--Qwen--Qwen3-VL-2B-Instruct/../../x'):
            with self.assertRaises(ValueError, msg=bad):
                components.safe_relative(item, bad)

    def test_ticket(self):
        state = components.Components()
        ticket = state.issue_ticket()['ticket']
        self.assertTrue(state.ticket_ok(ticket))
        self.assertFalse(state.ticket_ok('nope'))
        self.assertFalse(state.ticket_ok(''))

    def test_copy_from_peer(self):
        with tempfile.TemporaryDirectory() as source_dir, \
                tempfile.TemporaryDirectory() as target_dir:
            source = components.Components(make_root(source_dir, True))
            server = ThreadingHTTPServer(('127.0.0.1', 0), core.CoreHandler)
            server.app = SimpleNamespace(token='core-token', components=source)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            url = f'http://127.0.0.1:{server.server_port}'

            target = components.Components(make_root(target_dir, False))
            # Без пропуска манифест не отдаётся.
            with self.assertRaises(ValueError):
                target.start('qwen3-vl', 'copy', {'url': url, 'ticket': ''})
            target.start('qwen3-vl', 'copy', {'url': url, 'ticket': 'wrong'})
            operation = self.wait(target)
            self.assertEqual(operation['status'], 'error')

            ticket = source.issue_ticket()['ticket']
            target.start('qwen3-vl', 'copy', {'url': url, 'ticket': ticket})
            operation = self.wait(target)
            self.assertEqual(operation['status'], 'completed', operation['log'])
            copied = Path(target_dir) / 'models' / 'cv-models' / QWEN / 'snapshots' / 'abc'
            self.assertEqual((copied / 'model.safetensors').read_bytes(), b'w' * 5000)
            self.assertEqual(operation['done'], 5002)
            models = {item['id']: item for item in target.status()['models']}
            self.assertTrue(models['qwen3-vl']['installed'])

    def test_venv_models_cannot_be_downloaded_wrongly(self):
        state = components.Components()
        with self.assertRaises(ValueError):
            state.start('vision', 'download')
        with self.assertRaises(ValueError):
            state.start('paddle', 'download')
        with self.assertRaises(ValueError):
            state.start('qwen3-vl', 'install')

    @staticmethod
    def wait(state):
        deadline = time.time() + 30
        while time.time() < deadline:
            operation = state.status()['operation']
            if operation and operation['status'] != 'running':
                return operation
            time.sleep(0.05)
        raise AssertionError('операция не закончилась')


if __name__ == '__main__':
    unittest.main()
