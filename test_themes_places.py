# -*- coding: utf-8 -*-
"""Темы и места в подборках: отнесение к теме, ближайший город, поездки."""
from datetime import datetime, timedelta
import unittest

import numpy as np

import highlight_generator as hg
import highlight_themes
import places
from test_highlights import MODEL, candidate, unit

DIMS = 32


def theme_matrix(axis):
    """Описания темы смотрят вдоль одной оси пространства."""
    vector = np.zeros(DIMS, dtype=np.float32)
    vector[axis] = 1.0
    return np.stack([vector])


class ThemeAssignTests(unittest.TestCase):
    def test_outliers_go_to_their_theme_and_the_rest_stay_plain(self):
        rng = np.random.default_rng(1)
        items = []
        start = datetime(2023, 5, 1, 12)
        for index in range(200):
            item = candidate(f'p{index}.jpg', start + timedelta(hours=index), scene=index % 5 + 10)
            item.vector = unit(item.vector + 0.1 * rng.normal(size=DIMS))
            items.append(item)
        # Десять «котиков» — заметно ближе к оси темы 0.
        for item in items[:10]:
            item.vector = unit(item.vector * 0.3 + np.eye(DIMS, dtype=np.float32)[0])
        counts = highlight_themes.assign(items, {MODEL: {'cats': theme_matrix(0), 'sea': theme_matrix(1)}})
        self.assertEqual(counts.get('cats'), 10)
        self.assertEqual({item.theme for item in items[:10]}, {'cats'})
        # Остальные — не «котики»; случайный выброс в другую тему статистически возможен.
        self.assertTrue(all(item.theme != 'cats' for item in items[10:]))
        self.assertLessEqual(sum(item.theme is not None for item in items[10:]), 4)

    def test_pictures_like_wallpapers_stay_out_of_themes(self):
        rng = np.random.default_rng(2)
        items = []
        for index in range(200):
            item = candidate(f'p{index}.jpg', datetime(2023, 5, 1) + timedelta(hours=index), scene=index % 5 + 10)
            item.vector = unit(item.vector + 0.1 * rng.normal(size=DIMS))
            items.append(item)
        # Пять «котиков-обоев»: похожи и на тему, и на анти-тему.
        for item in items[:5]:
            item.vector = unit(item.vector * 0.3 + np.eye(DIMS, dtype=np.float32)[0] + np.eye(DIMS, dtype=np.float32)[2])
        counts = highlight_themes.assign(items, {MODEL: {'cats': theme_matrix(0), 'not-stock': theme_matrix(2)}})
        self.assertEqual(counts, {})

    def test_small_pool_gets_no_themes(self):
        items = [candidate(f'p{index}.jpg', datetime(2023, 5, 1) + timedelta(hours=index)) for index in range(10)]
        self.assertEqual(highlight_themes.assign(items, {MODEL: {'cats': theme_matrix(0)}}), {})

    def test_prompt_names_are_unique(self):
        names = [name for name, _ in highlight_themes.prompt_rows()]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith('theme:') for name in names))


class PlacesTests(unittest.TestCase):
    def test_nearest_city_by_coordinates(self):
        self.assertEqual(places.nearest(42.9849, 47.5047).name, 'Махачкала')
        self.assertEqual(places.nearest(42.0578, 48.2896).name, 'Дербент')
        self.assertEqual(places.nearest(29.9792, 31.1342).country_name, 'Египет')
        self.assertIsNone(places.nearest(43.0, 50.5), 'посреди Каспия города нет')


class TripTests(unittest.TestCase):
    def test_far_event_is_a_trip_named_by_place(self):
        home = places.nearest(42.9849, 47.5047)
        kaspiysk = places.nearest(42.8816, 47.6389)
        cairo = places.nearest(30.0444, 31.2357)
        self.assertEqual(cairo.name, 'Каир')
        start = datetime(2022, 8, 2, 10)
        trip = [candidate(f'c{index}.jpg', start + timedelta(minutes=10 * index), score=0.8, scene=index,
                          seed=index) for index in range(8)]
        for item in trip:
            item.place = cairo
        near = [candidate(f'd{index}.jpg', start + timedelta(days=5, minutes=10 * index), score=0.8,
                          scene=20 + index, seed=index) for index in range(8)]
        for item in near:
            item.place, item.theme = kaspiysk, 'sea'
        events = hg.detect_events(trip + near)
        groups = hg.event_groups(events, dict(hg.PARAMS), home=home)
        kinds = {group['kind']: group for group in groups}
        self.assertEqual(kinds['trip']['title'].split(' · ')[0], cairo.name)
        self.assertEqual(kinds['trip']['subtitle'], 'Египет')
        self.assertTrue(kinds['trip']['key'].startswith('trip:'))
        # Каспийск рядом с домом: обычное событие, названное по теме.
        self.assertTrue(kinds['event']['title'].startswith('Море · '))


if __name__ == '__main__':
    unittest.main()
