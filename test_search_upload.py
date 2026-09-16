import base64
import importlib.util
import io
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from PIL import Image

sys.modules.setdefault('people_gui', SimpleNamespace(CatalogStore=object))

spec = importlib.util.spec_from_file_location('web_server', Path(__file__).with_name('web_server.py'))
web_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web_server)


def jpeg_base64(color=(220, 40, 60)):
    stream = io.BytesIO()
    Image.new('RGB', (32, 24), color).save(stream, 'JPEG')
    return base64.b64encode(stream.getvalue()).decode('ascii')


class SearchUploadTests(unittest.TestCase):
    def test_video_frame_upload_uses_current_frame_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            media = root / 'clip.mp4'
            media.write_bytes(b'not a real video; only existence is checked for supplied frames')
            db = sqlite3.connect(':memory:')
            db.execute('CREATE TABLE photos(path TEXT PRIMARY KEY, status TEXT, kind TEXT)')
            db.execute("INSERT INTO photos VALUES('clip.mp4', 'ok', 'video')")
            app = SimpleNamespace(
                store=SimpleNamespace(db=db),
                file_for=lambda raw: root / raw,
            )

            with mock.patch.object(web_server.reverse_search, 'upload', return_value='https://tmp/frame.jpg') as upload:
                url = web_server._upload_search_image(app, 'clip.mp4', jpeg_base64())

            self.assertEqual(url, 'https://tmp/frame.jpg')
            payload, name = upload.call_args.args
            self.assertEqual(name, 'clip-frame.jpg')
            self.assertLessEqual(len(payload), web_server.SEARCH_FRAME_MAX_BYTES)
            with Image.open(io.BytesIO(payload)) as image:
                self.assertEqual(image.format, 'JPEG')

    def test_frame_payload_is_rejected_for_photos(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            media = root / 'photo.jpg'
            Image.new('RGB', (8, 8), 'white').save(media)
            db = sqlite3.connect(':memory:')
            db.execute('CREATE TABLE photos(path TEXT PRIMARY KEY, status TEXT, kind TEXT)')
            db.execute("INSERT INTO photos VALUES('photo.jpg', 'ok', 'photo')")
            app = SimpleNamespace(
                store=SimpleNamespace(db=db),
                file_for=lambda raw: root / raw,
            )

            with self.assertRaises(ValueError):
                web_server._upload_search_image(app, 'photo.jpg', jpeg_base64())


if __name__ == '__main__':
    unittest.main()
