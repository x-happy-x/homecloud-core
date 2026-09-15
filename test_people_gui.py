import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location('people_gui', Path(__file__).with_name('people_gui.py'))
people_gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(people_gui)


class PeopleGuiStoreTests(unittest.TestCase):
    def make_store(self, folder):
        db = people_gui.database(Path(folder))
        vectors = [
            [1.0, 0.01], [1.0, -0.01], [0.99, 0.0],
            [0.01, 1.0], [-0.01, 1.0], [0.0, 0.99],
        ]
        with db:
            for index, vector in enumerate(vectors):
                path = str(Path(folder) / f'photo-{index}.jpg')
                db.execute('INSERT INTO photos VALUES(?,?,?,?,?,?)',
                           (path, 1, index, 'test-model', 'ok', None))
                db.execute('INSERT INTO faces(path,box,embedding,thumbnail) VALUES(?,?,?,?)',
                           (path, '[0,0,100,100]', np.asarray(vector, dtype='<f4').tobytes(),
                            f'thumb-{index}.jpg'))
        db.close()
        return people_gui.CatalogStore(folder, min_cluster_size=2)

    def test_assign_merge_exclude_and_undo(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            try:
                automatic = [group for group in store.groups() if group['kind'] == 'auto']
                self.assertEqual(len(automatic), 2)
                first, second = automatic
                store.assign(first['face_ids'], 'Мама')
                store.assign(second['face_ids'], 'Мама')
                named = [group for group in store.groups() if group['kind'] == 'person']
                self.assertEqual(len(named), 1)
                self.assertEqual(len(named[0]['face_ids']), 6)

                face_id = named[0]['face_ids'][0]
                store.exclude([face_id])
                excluded = [group for group in store.groups() if group['kind'] == 'excluded']
                self.assertEqual(len(excluded[0]['face_ids']), 1)
                self.assertEqual(store.undo(), 'Лица исключены из автоматических групп')
                named = [group for group in store.groups() if group['kind'] == 'person']
                self.assertEqual(len(named[0]['face_ids']), 6)
            finally:
                store.db.close()


if __name__ == '__main__':
    unittest.main()
