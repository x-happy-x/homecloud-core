"""Копии файлов: ключи, выбор оригинала, этапы без копий и перенос результатов."""
from pathlib import Path
import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace

import catalog_index
import filekeys
import pathkeys


class FileKeysTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = self.root / 'catalog'
        self.catalog.mkdir()
        photos = self.root / 'photos'
        (photos / 'a').mkdir(parents=True)
        (photos / 'b').mkdir()
        same = os.urandom(200_000)
        self.files = {
            'first': photos / 'a' / 'one.jpg',
            'copy': photos / 'b' / 'one-copy.jpg',
            # Тот же размер, другое содержимое — не копия.
            'twin': photos / 'b' / 'other.jpg',
            'alone': photos / 'b' / 'alone.jpg',
        }
        self.files['first'].write_bytes(same)
        self.files['copy'].write_bytes(same)
        self.files['twin'].write_bytes(os.urandom(200_000))
        self.files['alone'].write_bytes(os.urandom(1234))
        db = sqlite3.connect(self.catalog / 'catalog.sqlite')
        catalog_index.ensure_schema(db)
        with db:
            for path in self.files.values():
                info = path.stat()
                db.execute("INSERT INTO photos(path,size,modified,status) VALUES(?,?,?,'ok')",
                           (str(path), info.st_size, info.st_mtime_ns))
            db.execute('CREATE TABLE photo_analysis (path TEXT PRIMARY KEY, caption TEXT)')
            db.execute('INSERT INTO photo_analysis VALUES(?,?)', (str(self.files['first']), 'кот'))
        db.close()

    def tearDown(self):
        self.temp.cleanup()

    def run_keys(self):
        filekeys.keys(SimpleNamespace(catalog=self.catalog, root=[], path=[], force=False,
                                      progress_file=None, stop_file=None))

    def db(self):
        return sqlite3.connect(self.catalog / 'catalog.sqlite')

    def test_copy_is_found_skipped_and_filled(self):
        self.run_keys()
        db = self.db()
        try:
            self.assertEqual(db.execute('SELECT path,original FROM photo_copies').fetchall(),
                             [(str(self.files['copy']), str(self.files['first']))])
            # Файл уникального размера ключ даже не считает.
            self.assertIsNone(db.execute('SELECT 1 FROM photo_keys WHERE path=?',
                                         (str(self.files['alone']),)).fetchone())
            where, values = pathkeys.analysis_scope_sql((), (), 'photos.path')
            chosen = {row[0] for row in db.execute(f'SELECT path FROM photos WHERE 1{where}', values)}
            self.assertNotIn(str(self.files['copy']), chosen)
            self.assertIn(str(self.files['first']), chosen)
            self.assertEqual(filekeys.propagate(db)['photo_analysis'], 1)
            self.assertEqual(filekeys.propagate(db)['photo_analysis'], 0, 'повтор без дублей')
            self.assertEqual(db.execute('SELECT caption FROM photo_analysis WHERE path=?',
                                        (str(self.files['copy']),)).fetchone(), ('кот',))
        finally:
            db.close()

    def test_changed_file_gets_a_new_key(self):
        self.run_keys()
        self.files['copy'].write_bytes(os.urandom(200_000))
        info = self.files['copy'].stat()
        db = self.db()
        with db:
            db.execute('UPDATE photos SET modified=? WHERE path=?',
                       (info.st_mtime_ns, str(self.files['copy'])))
        db.close()
        self.run_keys()
        db = self.db()
        try:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM photo_copies').fetchone(), (0,))
        finally:
            db.close()


if __name__ == '__main__':
    unittest.main()
