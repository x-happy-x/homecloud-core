# -*- coding: utf-8 -*-
"""Отбор групп копий: фильтр по одной папке и то, что в ней удалится."""
import unittest

import duplicates


def build(*groups):
    """Группы из путей: shapes задаём размером, чтобы «оставить» был предсказуем."""
    found = [{'key': f'g{index}', 'kind': 'exact', 'paths': list(paths)}
             for index, paths in enumerate(groups)]
    shapes = {}
    for paths in groups:
        for order, path in enumerate(paths):
            # Первый путь группы — самый крупный кадр, он и станет keep.
            side = 100 - order
            shapes[path] = (1000 - order, side, side)
    return duplicates.index(found, shapes)


class FolderScopeTests(unittest.TestCase):
    def test_summary_counts_all_copies_including_keeper(self):
        _, summary = build(
            (r'D:\album\one.jpg', r'D:\dump\one.jpg', r'D:\dump\sub\one.jpg'),
            (r'D:\album\two.jpg', r'D:\dump\two.jpg'),
        )
        counts = {item['folder']: item['copies'] for item in summary['top_folders']}
        self.assertEqual(counts[r'D:\dump'], 2)
        # Вложенная папка считается отдельно; папку с оригиналом тоже можно выбрать.
        self.assertEqual(counts[r'D:\dump\sub'], 1)
        self.assertEqual(counts[r'D:\album'], 2)

    def test_filter_returns_exactly_what_the_summary_promised(self):
        groups, summary = build(
            (r'D:\album\one.jpg', r'D:\dump\one.jpg'),
            (r'D:\album\two.jpg', r'D:\dump\two.jpg'),
            (r'D:\album\three.jpg', r'D:\other\three.jpg'),
        )
        chosen = duplicates.select(groups, folder=r'D:\dump')
        copies = sum(len(duplicates.in_folder(group, r'D:\dump')) for group in chosen)
        promised = {item['folder']: item['copies'] for item in summary['top_folders']}
        self.assertEqual(copies, promised[r'D:\dump'])
        self.assertEqual(len(chosen), 2)

    def test_nested_folders_are_not_swept_along(self):
        groups, _ = build((r'D:\album\one.jpg', r'D:\dump\sub\one.jpg'))
        self.assertEqual(duplicates.select(groups, folder=r'D:\dump'), [])
        self.assertEqual(len(duplicates.select(groups, folder=r'D:\dump\sub')), 1)

    def test_scope_includes_keeper(self):
        # Фильтр находит все файлы папки, включая исходный keep.
        groups, _ = build((r'D:\dump\big.jpg', r'D:\dump\small.jpg', r'D:\album\small.jpg'))
        chosen = duplicates.select(groups, folder=r'D:\dump')
        doomed = duplicates.in_folder(chosen[0], r'D:\dump')
        self.assertEqual(doomed, [r'D:\dump\big.jpg', r'D:\dump\small.jpg'])

    def test_folder_filter_stacks_with_the_others(self):
        groups, _ = build(
            (r'D:\album\one.jpg', r'D:\dump\one.jpg'),
            (r'D:\album\two.jpg', r'D:\dump\two.jpg'),
        )
        groups[0]['kind'] = 'similar'
        chosen = duplicates.select(groups, kind='exact', folder=r'D:\dump')
        self.assertEqual([group['key'] for group in chosen], [groups[1]['key']])

    def test_scoped_summary_and_keeper(self):
        groups, _ = build(
            (r'D:\album\one.jpg', r'D:\dump\one.jpg'),
            (r'D:\album\two.jpg', r'D:\other\two.jpg'),
        )
        shapes = {path: (100 if duplicates.parent(path) == r'D:\dump' else 200, 10, 10)
                  for group in groups for path in group['paths']}
        scoped, summary = duplicates.index(
            duplicates.select(groups, folder=r'D:\dump'), shapes, folder=r'D:\dump')
        self.assertEqual(scoped[0]['keep'], r'D:\dump\one.jpg')
        self.assertEqual(summary['groups'], 1)
        self.assertEqual(summary['exact'], 1)
        self.assertEqual(summary['files'], 2)
        self.assertEqual(summary['extra_files'], 1)
        self.assertEqual(summary['extra_bytes'], 200)
        self.assertEqual(summary['small_groups'], 1)
        self.assertEqual(len(duplicates.select(groups, folder=r'D:\album')), 2)


if __name__ == '__main__':
    unittest.main()
