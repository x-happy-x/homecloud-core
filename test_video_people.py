"""Подсказка «сколько людей на видео» и массовое исключение лиц файла/папки."""
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


class VideoPeopleTests(unittest.TestCase):
    def make_store(self, folder, entries):
        """entries: список (path, dir, vector)."""
        db = people_gui.database(Path(folder))
        db.execute('ALTER TABLE photos ADD COLUMN dir TEXT')
        with db:
            for index, (path, directory, vector) in enumerate(entries):
                db.execute(
                    'INSERT INTO photos(path,size,modified,model,status,error,dir) '
                    'VALUES(?,?,?,?,?,?,?) ON CONFLICT(path) DO NOTHING',
                    (path, 1, index, 'test-model', 'ok', None, directory))
                db.execute('INSERT INTO faces(path,box,embedding,thumbnail) VALUES(?,?,?,?)',
                           (path, '[0,0,100,100]', np.asarray(vector, dtype='<f4').tobytes(),
                            f'thumb-{index}.jpg'))
        db.close()
        return people_gui.CatalogStore(folder, min_cluster_size=2)

    def test_video_hint_merges_spurious_groups_of_one_video(self):
        """Одно видео случайно разъехалось на три группы — подсказка «двое»
        сводит его лица ровно к двум меткам, не трогая другое видео."""
        with tempfile.TemporaryDirectory() as temp:
            video = str(Path(temp) / 'clip.mp4')
            other = str(Path(temp) / 'other.mp4')
            vectors = clump(0) + clump(90) + clump(180)
            entries = [(video, temp, vector) for vector in vectors]
            entries.append((other, temp, [0.0, 1.0]))
            store = self.make_store(temp, entries)
            try:
                before = {row[0] for row in store.db.execute(
                    'SELECT DISTINCT label FROM face_clusters JOIN faces ON faces.id=face_clusters.face_id '
                    "WHERE faces.path=?", (video,))}
                self.assertEqual(len(before), 3)

                store.set_video_people_hint(video, 2)

                after = {row[0] for row in store.db.execute(
                    'SELECT DISTINCT label FROM face_clusters JOIN faces ON faces.id=face_clusters.face_id '
                    "WHERE faces.path=?", (video,))}
                self.assertEqual(len(after), 2)

                other_labels = {row[0] for row in store.db.execute(
                    'SELECT label FROM face_clusters JOIN faces ON faces.id=face_clusters.face_id '
                    "WHERE faces.path=?", (other,))}
                self.assertEqual(len(other_labels), 1)
                self.assertEqual(store.video_people_hint(video), 2)
            finally:
                store.db.close()

    def test_named_faces_are_not_touched_by_the_hint(self):
        with tempfile.TemporaryDirectory() as temp:
            video = str(Path(temp) / 'clip.mp4')
            vectors = clump(0) + clump(90)
            store = self.make_store(temp, [(video, temp, vector) for vector in vectors])
            try:
                named_face = store.rows[0][0]
                store.assign([named_face], 'Мама')
                store.set_video_people_hint(video, 1)
                person = [group for group in store.groups() if group['kind'] == 'person'][0]
                self.assertIn(named_face, person['face_ids'])
            finally:
                store.db.close()

    def test_exclude_path_excludes_whole_file(self):
        with tempfile.TemporaryDirectory() as temp:
            store = self.make_store(temp, [
                (str(Path(temp) / 'a.jpg'), temp, [1.0, 0.0]),
                (str(Path(temp) / 'a.jpg'), temp, [1.0, 0.01]),
                (str(Path(temp) / 'b.jpg'), temp, [0.0, 1.0]),
            ])
            try:
                count = store.exclude_path(str(Path(temp) / 'a.jpg'))
                self.assertEqual(count, 2)
                excluded = [group for group in store.groups() if group['kind'] == 'excluded']
                self.assertEqual(len(excluded[0]['face_ids']), 2)
            finally:
                store.db.close()

    def test_exclude_path_excludes_whole_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            sub = Path(temp) / 'private'
            sub.mkdir()
            store = self.make_store(temp, [
                (str(sub / 'a.jpg'), str(sub), [1.0, 0.0]),
                (str(sub / 'b.jpg'), str(sub), [1.0, 0.01]),
                (str(Path(temp) / 'c.jpg'), temp, [0.0, 1.0]),
            ])
            try:
                count = store.exclude_path(str(sub), folder=True)
                self.assertEqual(count, 2)
                excluded = [group for group in store.groups() if group['kind'] == 'excluded']
                self.assertEqual(len(excluded[0]['face_ids']), 2)
                remaining_auto = [group for group in store.groups() if group['kind'] == 'auto']
                self.assertEqual(sum(len(group['face_ids']) for group in remaining_auto), 0)
            finally:
                store.db.close()


if __name__ == '__main__':
    unittest.main()
