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
    def test_summary_counts_copies_beside_the_keeper(self):
        _, summary = build(
            (r'D:\album\one.jpg', r'D:\dump\one.jpg', r'D:\dump\sub\one.jpg'),
            (r'D:\album\two.jpg', r'D:\dump\two.jpg'),
        )
        counts = {item['folder']: item['copies'] for item in summary['top_folders']}
        self.assertEqual(counts[r'D:\dump'], 2)
        # Вложенная папка считается отдельно, а у папки с оригиналом лишнего нет.
        self.assertEqual(counts[r'D:\dump\sub'], 1)
        self.assertNotIn(r'D:\album', counts)

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

    def test_scope_never_offers_the_keeper_for_deletion(self):
        # Здесь keep лежит в самой папке: удалить можно только вторую копию.
        groups, _ = build((r'D:\dump\big.jpg', r'D:\dump\small.jpg', r'D:\album\small.jpg'))
        chosen = duplicates.select(groups, folder=r'D:\dump')
        doomed = duplicates.in_folder(chosen[0], r'D:\dump')
        self.assertEqual(doomed, [r'D:\dump\small.jpg'])
        self.assertNotIn(chosen[0]['keep'], doomed)

    def test_folder_filter_stacks_with_the_others(self):
        groups, _ = build(
            (r'D:\album\one.jpg', r'D:\dump\one.jpg'),
            (r'D:\album\two.jpg', r'D:\dump\two.jpg'),
        )
        groups[0]['kind'] = 'similar'
        chosen = duplicates.select(groups, kind='exact', folder=r'D:\dump')
        self.assertEqual([group['key'] for group in chosen], [groups[1]['key']])


if __name__ == '__main__':
    unittest.main()
