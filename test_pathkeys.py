import unittest

import pathkeys


class PathKeysTest(unittest.TestCase):
    def test_windows_device_key(self):
        key = 'pc-x:D:\\trash\\a.jpg'
        self.assertEqual(pathkeys.split(key), ('pc-x', 'D:\\trash\\a.jpg'))
        self.assertEqual(pathkeys.name(key), 'a.jpg')
        self.assertEqual(pathkeys.parent(key), 'pc-x:D:\\trash')
        self.assertEqual(pathkeys.parent('pc-x:D:\\trash'), 'pc-x:D:\\')
        self.assertEqual(pathkeys.parent('pc-x:D:\\'), 'pc-x:D:\\')
        self.assertTrue(pathkeys.is_root('pc-x:D:\\'))
        self.assertEqual(pathkeys.trim('pc-x:D:'), 'pc-x:D:\\')
        self.assertEqual(pathkeys.join('pc-x:D:\\', 'x', 'y.jpg'), 'pc-x:D:\\x\\y.jpg')
        self.assertEqual(pathkeys.chain(key),
                         ['pc-x:D:\\', 'pc-x:D:\\trash', 'pc-x:D:\\trash\\a.jpg'])

    def test_posix_source_key(self):
        key = 'netcraze:/Photos/2020/a.JPG'
        self.assertEqual(pathkeys.name(key), 'a.JPG')
        self.assertEqual(pathkeys.suffix(key), '.jpg')
        self.assertEqual(pathkeys.stem(key), 'a')
        self.assertEqual(pathkeys.parent(key), 'netcraze:/Photos/2020')
        self.assertEqual(pathkeys.parent('netcraze:/Photos'), 'netcraze:/')
        self.assertEqual(pathkeys.parent('netcraze:/'), 'netcraze:/')
        self.assertEqual(pathkeys.trim('netcraze:/Photos/'), 'netcraze:/Photos')
        self.assertEqual(pathkeys.join('netcraze:/', 'Photos'), 'netcraze:/Photos')
        self.assertEqual(pathkeys.join('netcraze:/Photos', 'a.jpg'), 'netcraze:/Photos/a.jpg')

    def test_legacy_path_has_no_source(self):
        self.assertEqual(pathkeys.split('D:\\x\\y.jpg'), ('', 'D:\\x\\y.jpg'))
        self.assertEqual(pathkeys.parent('D:\\x\\y.jpg'), 'D:\\x')
        self.assertFalse(pathkeys.is_key('d:\\x'))

    def test_inside_and_bounds(self):
        self.assertTrue(pathkeys.inside('pc-x:D:\\trash\\a.jpg', 'pc-x:D:\\trash'))
        self.assertFalse(pathkeys.inside('pc-x:D:\\trash2\\a.jpg', 'pc-x:D:\\trash'))
        self.assertTrue(pathkeys.inside('pc-x:D:\\a.jpg', 'pc-x:D:\\'))
        low, high = pathkeys.bounds('netcraze:/Photos')
        self.assertEqual(low, 'netcraze:/Photos/')
        self.assertTrue(low <= 'netcraze:/Photos/x.jpg' < high)

    def test_scope_sql(self):
        where, values = pathkeys.scope_sql(['netcraze:/Photos'], ['pc-x:D:\\a.jpg'])
        self.assertIn('photos.path=?', where)
        self.assertEqual(values, ['netcraze:/Photos', 'netcraze:/Photos/',
                                  'netcraze:/Photos/\uffff', 'pc-x:D:\\a.jpg'])


if __name__ == '__main__':
    unittest.main()
