import io
import unittest
from unittest import mock

import reverse_search


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class ReverseSearchTests(unittest.TestCase):
    def test_upload_falls_back_to_uguu_when_litterbox_fails(self):
        calls = []

        def fake_urlopen(request, **_kwargs):
            calls.append(request.full_url)
            if request.full_url == reverse_search.UPLOAD_URL:
                raise OSError('tls failed')
            return Response(
                b'{"success":true,"files":[{"url":"https:\\/\\/d.uguu.se\\/image.jpg"}]}')

        with mock.patch.object(reverse_search, 'urlopen', side_effect=fake_urlopen):
            url = reverse_search.upload(b'jpeg', 'photo.jpg', timeout=1)

        self.assertEqual(url, 'https://d.uguu.se/image.jpg')
        self.assertEqual(calls, [reverse_search.UPLOAD_URL, reverse_search.UGUU_UPLOAD_URL])


if __name__ == '__main__':
    unittest.main()
