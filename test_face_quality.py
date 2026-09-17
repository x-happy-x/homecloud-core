from pathlib import Path
import sqlite3
import tempfile
import unittest

import cv2
import numpy as np

import face_quality
import face_stacks
from prototype import cluster_embeddings


def face_like(size=160, seed=0):
    """Резкая «картинка» с мелкими деталями: шум поверх градиента."""
    rng = np.random.default_rng(seed)
    base = np.linspace(40, 200, size, dtype=np.float32)[None, :].repeat(size, axis=0)
    detail = rng.normal(0, 35, (size, size)).astype(np.float32)
    return np.clip(base + detail, 0, 255).astype(np.uint8)


class BlurScoreTests(unittest.TestCase):
    def test_blurred_image_scores_higher_than_sharp(self):
        sharp = face_like()
        soft = cv2.GaussianBlur(sharp, (0, 0), 4)
        self.assertLess(face_quality.face_blur(sharp), 0.5)
        self.assertGreater(face_quality.face_blur(soft), face_quality.BLUR_THRESHOLD)

    def test_tiny_face_looks_blurry_after_upscaling(self):
        tiny = cv2.resize(face_like(), (14, 14), interpolation=cv2.INTER_AREA)
        self.assertGreater(face_quality.face_blur(tiny), face_quality.face_blur(face_like()))

    def test_flat_image_is_blurry_not_an_error(self):
        self.assertEqual(face_quality.blur_score(np.full((96, 96), 128.0)), 1.0)

    def test_unknown_score_is_not_blurry(self):
        self.assertFalse(face_quality.is_blurry(None, None))
        self.assertTrue(face_quality.is_blurry(None, 12))
        self.assertTrue(face_quality.is_blurry(0.9, 300))
        self.assertFalse(face_quality.is_blurry(0.4, 300))


class MeasureTests(unittest.TestCase):
    def test_measure_reads_thumbnails_and_keeps_human_decision(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'thumbnails').mkdir()
            cv2.imwrite(str(folder / 'thumbnails' / 'sharp.jpg'), face_like())
            cv2.imwrite(str(folder / 'thumbnails' / 'soft.jpg'),
                        cv2.GaussianBlur(face_like(), (0, 0), 5))
            db = sqlite3.connect(folder / 'catalog.sqlite')
            db.execute('CREATE TABLE faces (id INTEGER PRIMARY KEY, path TEXT, box TEXT, '
                       'embedding BLOB, thumbnail TEXT)')
            db.executemany('INSERT INTO faces(id,path,box,thumbnail) VALUES(?,?,?,?)', [
                (1, 'a.jpg', '[0,0,120,140]', 'thumbnails/sharp.jpg'),
                (2, 'b.jpg', '[0,0,120,140]', 'thumbnails/soft.jpg'),
                (3, 'c.jpg', '[0,0,12,12]', 'thumbnails/sharp.jpg'),
                (4, 'd.jpg', '[0,0,120,140]', 'thumbnails/missing.jpg'),
            ])
            face_quality.ensure_schema(db)
            self.assertEqual(face_quality.measure(db, folder), 4)
            self.assertEqual(face_quality.measure(db, folder), 0, 'второй раз считать нечего')
            self.assertEqual(face_quality.blurry_ids(db), {2, 3})
            face_quality.keep(db, [2])
            face_quality.measure(db, folder, [2])
            self.assertEqual(face_quality.blurry_ids(db), {3})
            db.close()


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class StackTests(unittest.TestCase):
    def face(self, face_id, **extra):
        base = {'id': face_id, 'path': f'/p/{face_id}.jpg', 'kind': 'photo', 'folder': '/p',
                'taken': None, 'dhash': None, 'vector': None, 'blur': None}
        return {**base, **extra}

    def test_video_tracks_of_one_file_share_a_stack_topped_by_sharpest(self):
        faces = [
            self.face(1, path='/v/clip.mp4', kind='video', blur=0.6),
            self.face(2, path='/v/clip.mp4', kind='video', blur=0.3),
            self.face(3, path='/v/other.mp4', kind='video'),
        ]
        stacks = face_stacks.stack_faces(faces)
        self.assertEqual(stacks, {1: 2, 2: 2, 3: 3})

    def test_copies_stack_across_folders_but_distant_hashes_do_not(self):
        faces = [
            self.face(1, folder='/a', dhash='ffff0000ffff0000'),
            self.face(2, folder='/b', dhash='ffff0000ffff0001'),
            self.face(3, folder='/c', dhash='ffff0000ffffff00'),
            self.face(4, folder='/a', dhash='ffff0000ffff003f'),
        ]
        stacks = face_stacks.stack_faces(faces)
        self.assertEqual(stacks[1], stacks[2])
        # 8 бит разницы в чужой папке — уже не копия; 6 бит в своей — серия.
        self.assertNotEqual(stacks[1], stacks[3])
        self.assertEqual(stacks[1], stacks[4])

    def test_series_needs_close_time_and_similar_faces(self):
        same = unit([1, 0.05])
        stranger = unit([0, 1])
        faces = [
            self.face(1, taken=100.0, vector=same),
            self.face(2, taken=101.5, vector=same),
            self.face(3, taken=102.0, vector=stranger),
            self.face(4, taken=160.0, vector=same),
        ]
        stacks = face_stacks.stack_faces(faces)
        self.assertEqual(stacks[1], stacks[2])
        self.assertEqual(stacks[3], 3)
        self.assertEqual(stacks[4], 4)


class AverageLinkageTests(unittest.TestCase):
    def test_two_people_bridged_by_a_chain_stay_apart(self):
        """Промежуточные лица между двумя людьми не склеивают их в одну группу:
        средняя связь смотрит на все пары лиц сразу, а не на ближайших соседей."""
        rng = np.random.default_rng(3)
        dims = 64
        first, second = unit(rng.normal(size=dims)), unit(rng.normal(size=dims))
        people = []
        for centre in (first, second):
            people += [unit(centre + rng.normal(0, 0.25, dims) / np.sqrt(dims) * 4)
                       for _ in range(20)]
        bridge = [unit(first * (1 - t) + second * t) for t in np.linspace(0.2, 0.8, 5)]
        labels, probabilities = cluster_embeddings(np.stack(people + bridge),
                                                   min_cluster_size=5)
        self.assertEqual(len(set(labels[:20])), 1)
        self.assertEqual(len(set(labels[20:40])), 1)
        self.assertNotEqual(labels[0], labels[20])
        self.assertGreaterEqual(labels[0], 0)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_small_groups_become_noise(self):
        rng = np.random.default_rng(1)
        lonely = np.stack([unit(rng.normal(size=32)) for _ in range(6)])
        labels, _ = cluster_embeddings(lonely, min_cluster_size=3)
        self.assertTrue(np.all(labels == -1))


if __name__ == '__main__':
    unittest.main()
