import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'job_features', Path(__file__).with_name('job_features.py'))
job_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job_features)

ALL = {name: True for name in job_features.FEATURES}


class JobFeatureTests(unittest.TestCase):
    def test_single_set_goes_to_photos_and_videos(self):
        """Прежний вызов без второго набора работает как раньше."""
        selected, kinds = job_features.resolve({'faces': True}, None, ALL)
        self.assertTrue(selected['faces'])
        self.assertEqual(kinds['faces'], 'all')

    def test_sets_are_independent(self):
        selected, kinds = job_features.resolve({'faces': True}, {'ocr': True}, ALL)
        self.assertEqual(kinds['faces'], 'photos')
        self.assertEqual(kinds['ocr'], 'videos')
        # OCR тянет за собой визуальный индекс — и только для роликов.
        self.assertTrue(selected['visual'])
        self.assertEqual(kinds['visual'], 'videos')

    def test_same_feature_on_both_sides_is_all(self):
        _, kinds = job_features.resolve({'faces': True}, {'faces': True}, ALL)
        self.assertEqual(kinds['faces'], 'all')

    def test_caption_pulls_visual_and_adult_within_its_kind(self):
        _, kinds = job_features.resolve({'caption': True}, {}, ALL)
        self.assertEqual(kinds['caption'], 'photos')
        self.assertEqual(kinds['visual'], 'photos')
        self.assertEqual(kinds['adult'], 'photos')

    def test_adult_dependency_skipped_when_device_cannot(self):
        selected, _ = job_features.resolve({'caption': True}, {}, {**ALL, 'adult': False})
        self.assertFalse(selected['adult'])

    def test_speech_never_lands_on_photos(self):
        selected, kinds = job_features.resolve({'speech': True}, {'speech': True}, ALL)
        self.assertTrue(selected['speech'])
        self.assertEqual(kinds['speech'], 'videos')

    def test_inventory_has_no_kind_and_is_shared(self):
        selected, kinds = job_features.resolve({}, {'inventory': True}, ALL)
        self.assertTrue(selected['inventory'])
        self.assertNotIn('inventory', kinds)

    def test_empty_selection(self):
        selected, kinds = job_features.resolve({}, {}, ALL)
        self.assertFalse(any(selected.values()))
        self.assertEqual(kinds, {})


if __name__ == '__main__':
    unittest.main()
