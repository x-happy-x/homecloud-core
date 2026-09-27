"""Ручная отметка 18+: копии, возврат автоматической оценки, этап её не трогает."""
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace

import adult_manual
import adult_photos
import catalog_index


class AdultManualTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = sqlite3.connect(Path(self.temp.name) / 'catalog.sqlite')
        catalog_index.ensure_schema(self.db)
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS photo_adult_analysis (
            path TEXT PRIMARY KEY, size INTEGER NOT NULL, modified INTEGER NOT NULL,
            rating TEXT NOT NULL DEFAULT 'unknown', adult_score REAL NOT NULL DEFAULT 0,
            tags_json TEXT NOT NULL DEFAULT '[]', regions_json TEXT NOT NULL DEFAULT '[]',
            description TEXT NOT NULL DEFAULT '', detector_model TEXT, tagger_model TEXT,
            status TEXT NOT NULL, error TEXT, analyzed_at TEXT NOT NULL);
        ''')
        with self.db:
            for path in ('a.jpg', 'copy.jpg', 'other.jpg'):
                self.db.execute("INSERT INTO photos(path,size,modified,status) VALUES(?,10,20,'ok')", (path,))
            self.db.execute("INSERT INTO photo_copies VALUES('copy.jpg','a.jpg')")
            for path in ('a.jpg', 'copy.jpg'):
                self.db.execute(
                    "INSERT INTO photo_adult_analysis(path,size,modified,rating,adult_score,regions_json,"
                    "detector_model,status,analyzed_at) VALUES(?,10,20,'explicit',.9,'[{\"box\":[1,2,3,4]}]',"
                    "'nudenet','ok','t')", (path,))

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def rating(self, path):
        row = self.db.execute('SELECT rating,regions_json FROM photo_adult_analysis WHERE path=?',
                              (path,)).fetchone()
        return row and row[0]

    def test_safe_mark_covers_copies_and_can_be_undone(self):
        changed = adult_manual.mark(self.db, ['copy.jpg'], 'safe', 'anna')
        self.assertEqual(changed, ['a.jpg', 'copy.jpg'])
        self.assertEqual(self.rating('a.jpg'), 'safe')
        self.assertEqual(self.rating('copy.jpg'), 'safe')
        self.assertEqual(adult_manual.manual_ratings(self.db, ['a.jpg', 'other.jpg']), {'a.jpg': 'safe'})
        # Передумали: отметили 18+ — исходная оценка всё равно помнится первая.
        adult_manual.mark(self.db, ['a.jpg'], 'explicit', 'anna')
        self.assertEqual(self.rating('a.jpg'), 'explicit')
        adult_manual.mark(self.db, ['a.jpg'], None)
        row = self.db.execute('SELECT rating,regions_json,detector_model FROM photo_adult_analysis '
                              'WHERE path=?', ('a.jpg',)).fetchone()
        self.assertEqual(row, ('explicit', '[{"box":[1,2,3,4]}]', 'nudenet'))
        self.assertEqual(adult_manual.manual_ratings(self.db, ['a.jpg', 'copy.jpg']), {})

    def test_undo_without_previous_removes_the_row(self):
        adult_manual.mark(self.db, ['other.jpg'], 'safe')
        self.assertEqual(self.rating('other.jpg'), 'safe')
        adult_manual.mark(self.db, ['other.jpg'], None)
        self.assertIsNone(self.rating('other.jpg'))

    def test_stage_skips_manual_marks(self):
        adult_manual.mark(self.db, ['other.jpg'], 'safe')
        args = SimpleNamespace(root=[], path=[], force=True, limit=100, kinds='all')
        # Колонка blocked есть в настоящем каталоге.
        self.db.execute('ALTER TABLE photos ADD COLUMN blocked INTEGER')
        paths = [row[0] for row in adult_photos.scoped_rows(self.db, args)]
        self.assertNotIn('other.jpg', paths)

    def test_unknown_rating_is_rejected(self):
        with self.assertRaises(ValueError):
            adult_manual.mark(self.db, ['a.jpg'], 'maybe')


class SimilarPhotosTest(unittest.TestCase):
    def test_similar_skips_itself_and_copies_in_score_order(self):
        import threading
        import web_server
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        db = sqlite3.connect(Path(temp.name) / 'catalog.sqlite', check_same_thread=False)
        self.addCleanup(db.close)
        catalog_index.ensure_schema(db)
        db.executescript('''CREATE TABLE photo_embeddings (path TEXT, model TEXT, size INTEGER,
            modified INTEGER, embedding BLOB, dims INTEGER, analyzed_at TEXT, PRIMARY KEY(path,model));
            CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);''')
        db.execute("INSERT INTO photo_copies VALUES('copy.jpg','a.jpg')")
        import settings as catalog_settings
        model = catalog_settings.read(db)['visual_model']
        db.execute("INSERT INTO photo_embeddings VALUES('a.jpg',?,1,1,?,2,'t')", (model, b'\0' * 8))
        asked = {}

        class Semantic:
            def query(self, text, top=500, vector=''):
                asked['vector'] = vector
                return [{'path': 'a.jpg', 'score': 1}, {'path': 'copy.jpg', 'score': .99},
                        {'path': 'far.jpg', 'score': .2}, {'path': 'near.jpg', 'score': .8}]

        def payloads(**kwargs):
            return [{'path': path} for path in kwargs['wanted']][::-1], 0

        app = SimpleNamespace(lock=threading.RLock(), store=SimpleNamespace(db=db),
                              semantic=Semantic(), photo_payloads=payloads)
        result = web_server.App.similar_photos(app, 'a.jpg', 5)
        self.assertTrue(asked['vector'])
        self.assertEqual([card['path'] for card in result['photos']], ['near.jpg', 'far.jpg'])
        self.assertEqual(web_server.App.similar_photos(app, 'missing.jpg', 5),
                         {'photos': [], 'ready': False})


if __name__ == '__main__':
    unittest.main()
