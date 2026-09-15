import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('prototype', Path(__file__).with_name('prototype.py'))
prototype = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prototype)


class CatalogTests(unittest.TestCase):
    def test_hdbscan_separates_clear_face_groups(self):
        vectors = np.array([
            [1.0, 0.01], [1.0, -0.01], [0.99, 0.0],
            [0.01, 1.0], [-0.01, 1.0], [0.0, 0.99],
        ], dtype=np.float32)
        labels, probabilities = prototype.cluster_embeddings(
            vectors, algorithm='hdbscan', min_cluster_size=2)
        self.assertTrue(np.all(labels[:3] == labels[0]))
        self.assertTrue(np.all(labels[3:] == labels[3]))
        self.assertNotEqual(labels[0], labels[3])
        self.assertTrue(np.all(probabilities > 0))

    def test_failed_replacement_rolls_back_existing_faces(self):
        with tempfile.TemporaryDirectory() as temp:
            db = prototype.database(Path(temp))
            with db:
                db.execute('INSERT INTO photos VALUES(?,?,?,?,?,?)', ('photo.jpg', 1, 2, 'model', 'ok', None))
                db.execute('INSERT INTO faces(path,box,embedding,thumbnail) VALUES(?,?,?,?)', ('photo.jpg', '[]', b'old', 'old.jpg'))
            try:
                with db:
                    db.execute('DELETE FROM faces WHERE path=?', ('photo.jpg',))
                    raise RuntimeError('Interrupted replacement')
            except RuntimeError:
                pass
            self.assertEqual(db.execute('SELECT embedding FROM faces').fetchone()[0], b'old')
            db.close()


if __name__ == '__main__':
    unittest.main()
