# -*- coding: utf-8 -*-
"""Метаданные для просмотрщика: EXIF с камеры читается и подписывается по-человечески."""
from pathlib import Path
import tempfile
import unittest

from PIL import ExifTags, Image, TiffImagePlugin

import media_metadata


def rational(value):
    return TiffImagePlugin.IFDRational(value)


class PhotoMetadataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / 'IMG_1.jpg'
        exif = Image.Exif()
        exif[ExifTags.Base.Make] = 'Xiaomi'
        exif[ExifTags.Base.Model] = 'MI 8'
        exif[ExifTags.Base.Orientation] = 6
        sub = exif.get_ifd(ExifTags.IFD.Exif)
        sub[ExifTags.Base.DateTimeOriginal] = '2019:11:12 17:36:09'
        sub[ExifTags.Base.ExposureTime] = rational(1 / 125)
        sub[ExifTags.Base.FNumber] = rational(1.8)
        sub[ExifTags.Base.ISOSpeedRatings] = 400
        sub[ExifTags.Base.Flash] = 16
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps[1], gps[2] = 'N', (rational(47), rational(13), rational(55.57))
        gps[3], gps[4] = 'E', (rational(39), rational(42), rational(55.19))
        gps[6] = rational(103.7)
        Image.new('RGB', (64, 48), 'teal').save(self.path, exif=exif)

    def tearDown(self):
        self.folder.cleanup()

    def values(self):
        result = media_metadata.read(self.path)
        return result, {item['label']: item['value']
                        for group in result['groups'] for item in group['items']}

    def test_camera_fields_are_human_readable(self):
        result, values = self.values()
        self.assertEqual(result['kind'], 'photo')
        self.assertEqual(values['Производитель'], 'Xiaomi')
        self.assertEqual(values['Снято'], '12.11.2019 17:36:09')
        self.assertEqual(values['Выдержка'], '1/125 с')
        self.assertEqual(values['Диафрагма'], 'f/1.8')
        self.assertEqual(values['ISO'], '400')
        self.assertEqual(values['Вспышка'], 'не сработала (выключена)')
        self.assertEqual(values['Ориентация'], 'Повёрнута на 90° вправо')
        self.assertEqual(values['Размер в пикселях'], '64×48')

    def test_gps_becomes_coordinates(self):
        result, values = self.values()
        self.assertAlmostEqual(result['coords']['latitude'], 47.232103, places=4)
        self.assertAlmostEqual(result['coords']['longitude'], 39.715331, places=4)
        self.assertEqual(values['Высота'], '103.7 м')
        self.assertEqual([group['id'] for group in result['groups']][:3], ['shot', 'camera', 'place'])

    def test_integers_keep_their_zeros(self):
        self.assertEqual(media_metadata._trim(90.0, 0), '90')
        self.assertEqual(media_metadata._trim(30.0), '30')

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            media_metadata.read(Path(self.folder.name) / 'nope.jpg')


if __name__ == '__main__':
    unittest.main()
