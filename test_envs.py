"""Окружения ядра: work как обычная папка и как ссылка на другой диск."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import envs


class WorkerRootTest(unittest.TestCase):
    def test_plain_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'work').mkdir()
            self.assertEqual(envs.worker_root(root), root / 'work')
            self.assertEqual(envs.python('vision-venv', root),
                             root / 'work' / 'vision-venv' / 'Scripts' / 'python.exe')

    def test_missing_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(envs.worker_root(temp), Path(temp) / 'work')

    @unittest.skipUnless(os.name == 'nt', 'junction есть только в Windows')
    def test_junction_is_resolved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = root / 'elsewhere' / 'work'
            real.mkdir(parents=True)
            (root / 'core').mkdir()
            made = subprocess.run(['cmd', '/c', 'mklink', '/J', str(root / 'core' / 'work'), str(real)],
                                  capture_output=True)
            if made.returncode:
                self.skipTest('mklink /J недоступен')
            self.assertEqual(os.path.normcase(envs.worker_root(root / 'core')),
                             os.path.normcase(real))


if __name__ == '__main__':
    unittest.main()
