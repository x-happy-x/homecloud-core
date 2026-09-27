import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location('people_gui', Path(__file__).with_name('people_gui.py'))
people_gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(people_gui)


def clump(degrees, count=3, spread=1.5):
    """Тесная горстка векторов вокруг заданного угла — одна будущая группа."""
    return [[float(np.cos(np.radians(degrees + offset * spread))),
             float(np.sin(np.radians(degrees + offset * spread)))]
            for offset in range(count)]


class PeopleGuiStoreTests(unittest.TestCase):
    def make_store(self, folder, vectors=None, labels=None):
        db = people_gui.database(Path(folder))
        if vectors is None:
            vectors = [
                [1.0, 0.01], [1.0, -0.01], [0.99, 0.0],
                [0.01, 1.0], [-0.01, 1.0], [0.0, 0.99],
            ]
        with db:
            for index, vector in enumerate(vectors):
                path = str(Path(folder) / f'photo-{index}.jpg')
                # Колонки перечислены поимённо: в photos их со временем
                # прибавлялось (kind, duration), и позиционная вставка ломалась.
                db.execute('INSERT INTO photos(path,size,modified,model,status,error) '
                           'VALUES(?,?,?,?,?,?)',
                           (path, 1, index, 'test-model', 'ok', None))
                db.execute('INSERT INTO faces(path,box,embedding,thumbnail) VALUES(?,?,?,?)',
                           (path, '[0,0,100,100]', np.asarray(vector, dtype='<f4').tobytes(),
                            f'thumb-{index}.jpg'))
                if labels is not None:
                    # Готовые метки — как у каталога, где человек разъехался
                    # по группам в разных проходах сборки.
                    db.execute('INSERT INTO face_clusters(face_id,label,probability,method,'
                               'computed_at) VALUES(last_insert_rowid(),?,1,?,?)',
                               (labels[index], 'test', 'now'))
        db.close()
        return people_gui.CatalogStore(folder, min_cluster_size=2)

    def test_rejected_candidate_is_not_offered_again(self):
        with tempfile.TemporaryDirectory() as temp:
            vectors = [*clump(0, 4), *clump(90, 3), [0.995, 0.02], [0.99, -0.03]]
            store = self.make_store(temp, vectors)
            store.assign([1, 2, 3, 4], 'Анна', None)
            person = store.db.execute("SELECT id FROM people WHERE name='Анна'").fetchone()[0]
            offered = {item['face_id'] for item in store.person_candidates(person)}
            self.assertTrue({8, 9} <= offered, offered)
            self.assertEqual(store.reject_candidates(person, [8]), 1)
            store.reject_candidates(person, [8])
            offered = {item['face_id'] for item in store.person_candidates(person)}
            self.assertNotIn(8, offered)
            self.assertIn(9, offered)
            with self.assertRaises(KeyError):
                store.reject_candidates(person + 100, [9])
            store.db.close()

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
                self.assertGreater(store.db.execute(
                    'SELECT count(*) FROM face_identity_conflicts').fetchone()[0], 0)
                excluded = [group for group in store.groups() if group['kind'] == 'excluded']
                self.assertEqual(len(excluded[0]['face_ids']), 1)
                self.assertEqual(store.undo(), 'Лица исключены из автоматических групп')
                self.assertEqual(store.db.execute(
                    'SELECT count(*) FROM face_identity_conflicts').fetchone()[0], 0)
                named = [group for group in store.groups() if group['kind'] == 'person']
                self.assertEqual(len(named[0]['face_ids']), 6)
            finally:
                store.db.close()

    def test_assign_does_not_request_full_recluster(self):
        """Ручное имя применяется сразу и не запускает тяжёлую пересборку."""
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            try:
                with store.db:
                    store.db.execute(
                        "INSERT OR REPLACE INTO identity_state VALUES('dirty','0')")
                automatic = next(group for group in store.groups() if group['kind'] == 'auto')
                store.assign(automatic['face_ids'], 'Мама')
                dirty = store.db.execute(
                    "SELECT value FROM identity_state WHERE key='dirty'").fetchone()
                self.assertEqual(dirty[0], '0')
            finally:
                store.db.close()

    def test_suggestions_point_at_the_named_person(self):
        """Один человек разъехался на две группы: назвали одну — вторая должна
        её напомнить, а посторонняя группа рядом — остаться без догадки."""
        with tempfile.TemporaryDirectory() as temp:
            # Две горстки в 25 градусах друг от друга — это похожесть 0.91,
            # то есть один человек; третья в 80 градусах — посторонний.
            # Метки заданы заранее: средняя связь такие горстки сразу склеила
            # бы, а человек разъезжается по группам между проходами сборки.
            store = self.make_store(temp, clump(0) + clump(25) + clump(80),
                                    labels=[0, 0, 0, 1, 1, 1, 2, 2, 2])
            try:
                automatic = [group for group in store.groups() if group['kind'] == 'auto']
                self.assertEqual(len(automatic), 3)
                store.assign(automatic[0]['face_ids'], 'Мама')
                found = store.suggest_people(threshold=0.6)
                self.assertEqual(len(found), 1, 'подсказка должна быть ровно одна')
                self.assertEqual(found[0]['name'], 'Мама')
                self.assertGreater(found[0]['score'], 0.85)
                self.assertEqual(found[0]['faces'], 3)
                # Догадка досталась похожей группе, а не посторонней.
                rest = {group['key']: group for group in store.groups()
                        if group['kind'] == 'auto'}
                self.assertIn(found[0]['key'], rest)
            finally:
                store.db.close()

    def test_suggestions_are_silent_when_nobody_is_named(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            try:
                self.assertEqual(store.suggest_people(threshold=0.6), [])
            finally:
                store.db.close()

    def test_threshold_decides_whether_a_guess_is_shown(self):
        """Группы здесь ортогональны, похожесть около нуля: при высоком пороге
        подсказки быть не должно, при нулевом — появится."""
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp)
            try:
                automatic = [group for group in store.groups() if group['kind'] == 'auto']
                store.assign(automatic[0]['face_ids'], 'Мама')
                self.assertEqual(store.suggest_people(threshold=0.95), [])
                self.assertEqual(len(store.suggest_people(threshold=0.0)), 1)
            finally:
                store.db.close()


if __name__ == '__main__':
    unittest.main()
