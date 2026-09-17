import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

sys.modules.setdefault('people_gui', SimpleNamespace(CatalogStore=object))

spec = importlib.util.spec_from_file_location('web_server', Path(__file__).with_name('web_server.py'))
web_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web_server)
people_albums = web_server.people_albums


class HiddenPeopleTests(unittest.TestCase):
    """Человек из скрытого альбома не всплывает в подсказках — ни у кого."""

    def test_suggestions_skip_hidden_groups_and_hidden_people(self):
        with tempfile.TemporaryDirectory() as temp:
            db = sqlite3.connect(Path(temp) / 'catalog.sqlite')
            people_albums.ensure_schema(db)
            album = people_albums.create(db, 'Личное')
            people_albums.set_members(db, album, add=['person:7', 'auto:3'])
            people_albums.set_hidden(db, album, True)
            suggestions = [
                {'key': 'auto:1', 'person_id': 5, 'name': 'Мама', 'score': .9, 'faces': 4},
                {'key': 'auto:2', 'person_id': 7, 'name': 'Скрытый', 'score': .9, 'faces': 4},
                {'key': 'auto:3', 'person_id': 5, 'name': 'Мама', 'score': .9, 'faces': 4},
            ]
            app = SimpleNamespace(
                catalog_folder=Path(temp), lock=threading.Lock(),
                store=SimpleNamespace(db=db, suggest_people=lambda threshold: list(suggestions)))
            result = web_server.App.face_suggestions(app)
            db.close()
        self.assertEqual([item['key'] for item in result['suggestions']], ['auto:1'])


if __name__ == '__main__':
    unittest.main()
