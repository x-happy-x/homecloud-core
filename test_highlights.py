# -*- coding: utf-8 -*-
"""Оценка снимков и автоматические подборки на маленьком искусственном каталоге."""
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

import device_job
import highlight_generator as hg
import job_features
import photo_curation as pc

MODEL = 'google/siglip2-base-patch16-224'


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def scene_vector(scene, jitter=0.0, seed=0, dims=32):
    """Вектор «сцены»: кадры одной сцены близки, разных — почти ортогональны."""
    base = np.zeros(dims, dtype=np.float32)
    base[scene % dims] = 1.0
    noise = unit(np.random.default_rng(seed).normal(size=dims))
    return unit(base + jitter * noise)


def candidate(path, moment, score=0.7, scene=0, jitter=0.05, seed=0, dhash=None, lat=None, lon=None):
    item = hg.Candidate()
    item.path, item.taken, item.ts = path, moment, pc.wall_ts(moment)
    item.source, item.trust = 'exif', 1.0
    item.base = item.score = score
    item.visual, item.technical, item.personal = score, score, 0.0
    item.dhash = dhash
    item.model, item.vector = MODEL, scene_vector(scene, jitter, seed)
    item.lat, item.lon = lat, lon
    item.people, item.faces, item.event, item.series = [], 0, None, []
    return item


class TimeFromNameTests(unittest.TestCase):
    def test_camera_and_messenger_names(self):
        cases = {
            'IMG_20191218_131104.jpg': (datetime(2019, 12, 18, 13, 11, 4), 'filename'),
            'PXL_20230101_123456789.jpg': (datetime(2023, 1, 1, 12, 34, 56), 'filename'),
            'WhatsApp Image 2021-12-27 at 18.16.46.jpeg': (datetime(2021, 12, 27, 18, 16, 46), 'filename'),
            'Изображение WhatsApp 2024-02-21 в 17.56.22_60e48927.jpg':
                (datetime(2024, 2, 21, 17, 56, 22), 'filename'),
            'IMG-20210101-WA0001.jpg': (datetime(2021, 1, 1, 12, 0), 'filename-date'),
        }
        for name, expected in cases.items():
            self.assertEqual(pc.time_from_name(name), expected, name)

    def test_earliest_date_wins_over_resave_stamp(self):
        moment, source = pc.time_from_name('IMG-20180525-WA0008_1572848274767.jpg')
        self.assertEqual((moment.date(), source), (date(2018, 5, 25), 'filename-date'))

    def test_hashes_and_numbers_are_not_dates(self):
        self.assertEqual(pc.time_from_name('96b6785df7971e339c45a5c221b715fc.jpg'), (None, None))
        self.assertEqual(pc.time_from_name('00033-768846290.png'), (None, None))


class ScoreTests(unittest.TestCase):
    def row(self, faces=(), content='photo', rating='safe'):
        return {'width': 4000, 'height': 3000, 'size': 3_000_000, 'blur_score': 400.0,
                'analysis_model': MODEL, 'content_type': content, 'content_confidence': 0.4, 'rating': rating,
                'faces': list(faces)}

    FILES = {'brightness': 0.45, 'contrast': 0.18, 'clipped': 0.01, 'taken_ts': 1.0,
             'taken_source': 'exif'}

    @staticmethod
    def face(side, person=None, label=-1):
        return {'id': 1, 'box': [0, 0, side, side], 'label': label, 'person_id': person,
                'excluded': False, 'drawn': False}

    def test_face_does_not_raise_visual_quality(self):
        plain = pc.score(self.row(), self.FILES, {}, None)
        with_face = pc.score(self.row([self.face(1500, person=1)]), self.FILES, {}, None)
        self.assertAlmostEqual(plain['base_score'], with_face['base_score'])
        self.assertGreater(with_face['personal_score'], 0.5)
        self.assertEqual(plain['personal_score'], 0.0)

    def test_tiny_background_face_counts_less_than_portrait(self):
        tiny = pc.score(self.row([self.face(60, person=1)]), self.FILES, {}, None)
        portrait = pc.score(self.row([self.face(1200, person=1)]), self.FILES, {}, None)
        stranger = pc.score(self.row([self.face(1200)]), self.FILES, {}, None)
        self.assertLess(tiny['personal_score'], 0.05)
        self.assertGreater(portrait['personal_score'], stranger['personal_score'])

    def test_rejections_explain_why(self):
        result = pc.score(self.row(content='screenshot', rating='explicit'), self.FILES, {}, None)
        self.assertFalse(result['eligible'])
        self.assertEqual(json.loads(result['rejections_json']), ['content:screenshot', 'adult:explicit'])

    def test_misclassified_camera_photo_can_pass(self):
        row = self.row(content='game')
        self.assertTrue(pc.score(row, {**self.FILES, 'camera': 'Xiaomi MI 8'}, {}, None)['eligible'])
        self.assertFalse(pc.score(row, self.FILES, {}, None)['eligible'])

    def test_unreliable_content_model_needs_camera(self):
        row = {**self.row(content='photo'), 'analysis_model': 'jinaai/jina-clip-v2'}
        rejected = pc.score(row, self.FILES, {}, None)
        self.assertEqual(json.loads(rejected['rejections_json']), ['content:unverified'])
        self.assertTrue(pc.score(row, {**self.FILES, 'camera': 'Apple iPhone'}, {}, None)['eligible'])

    def test_uncalibrated_model_gives_no_visual_score(self):
        prompts = {'good': (1.0, unit([1, 0])), 'bad': (-1.0, unit([0, 1]))}
        jina = 'jinaai/jina-clip-v2'
        result = pc.score(self.row(), self.FILES, {jina: prompts}, (jina, unit([1, 0])))
        self.assertIsNone(result['visual_score'])
        self.assertAlmostEqual(result['base_score'],
                               0.55 * pc.VISUAL_PRIOR + 0.45 * result['technical_score'])


class EventTests(unittest.TestCase):
    def test_long_gap_splits(self):
        items = [candidate('a', datetime(2024, 7, 1, 10)), candidate('b', datetime(2024, 7, 1, 11)),
                 candidate('c', datetime(2024, 7, 1, 20))]
        self.assertEqual([len(event) for event in hg.detect_events(items)], [2, 1])

    def test_medium_gap_splits_only_on_scene_change(self):
        same = [candidate('a', datetime(2024, 7, 1, 10), scene=1),
                candidate('b', datetime(2024, 7, 1, 12), scene=1, seed=1)]
        other = [candidate('a', datetime(2024, 7, 1, 10), scene=1),
                 candidate('b', datetime(2024, 7, 1, 12), scene=2)]
        self.assertEqual(len(hg.detect_events(same)), 1)
        self.assertEqual(len(hg.detect_events(other)), 2)

    def test_place_change_splits_when_coordinates_exist(self):
        items = [candidate('a', datetime(2024, 7, 1, 10), lat=47.2, lon=39.7),
                 candidate('b', datetime(2024, 7, 1, 10, 40), lat=45.0, lon=38.9)]
        self.assertEqual(len(hg.detect_events(items)), 2)


class SelectionTests(unittest.TestCase):
    def test_near_duplicates_collapse_into_best(self):
        params = hg.PARAMS
        items = [candidate('best', datetime(2024, 7, 1, 10), score=0.9, dhash=0xFF00),
                 candidate('copy', datetime(2024, 7, 1, 12), score=0.8, scene=5, dhash=0xFF01),
                 candidate('burst', datetime(2024, 7, 1, 10, 0, 5), score=0.7, jitter=0.2, seed=3),
                 candidate('other', datetime(2024, 7, 1, 11), score=0.6, scene=9)]
        trace = {}
        kept = hg.collapse_series(items, params, trace)
        self.assertEqual([item.path for item in kept], ['best', 'other'])
        self.assertEqual(sorted(kept[0].series), ['burst', 'copy'])
        self.assertEqual(trace['copy']['decision'], 'duplicate')

    def test_long_series_of_one_scene_does_not_flood_selection(self):
        """30 похожих кадров одной сцены лучше остальных — но в подборку идут один-два."""
        items = []
        for number in range(30):
            items.append(candidate(f'scene-{number:02}', datetime(2024, 7, 1, 10, number),
                                   score=0.95 - number * 0.001, scene=1, jitter=0.3, seed=number))
        for number in range(8):
            items.append(candidate(f'day-{number}', datetime(2024, 7, 2 + number * 2, 12),
                                   score=0.7, scene=10 + number))
        hg.detect_events(sorted(items, key=lambda item: item.ts))
        trace = {}
        group = hg.build_group('month', 'month:2024-07', 'Июль 2024', '', items, (2.2, 6, 30),
                               hg.PARAMS, trace)
        picked = [photo['path'] for photo in group['photos']]
        self.assertLessEqual(sum(path.startswith('scene-') for path in picked), 2)
        self.assertEqual(sum(path.startswith('day-') for path in picked), 8)
        self.assertIn(trace['scene-05']['decision'], ('duplicate', 'same_scene'))

    def test_events_are_balanced(self):
        """Большое событие не забирает все места у маленьких."""
        items = []
        for number in range(40):
            items.append(candidate(f'big-{number:02}', datetime(2024, 7, 1, 8) + timedelta(minutes=10 * number),
                                   score=0.85, scene=number, seed=number))
        for number in range(3):
            items.append(candidate(f'small-{number}', datetime(2024, 7, 10 + number * 5, 12),
                                   score=0.75, scene=50 + number))
        items.sort(key=lambda item: item.ts)
        hg.detect_events(items)
        selected = hg.diversify(items, 10, hg.BUCKET_WEIGHTS['month'], hg.PARAMS)
        paths = [item.path for item, _ in selected]
        self.assertEqual(sum(path.startswith('small-') for path in paths), 3)


class CatalogTests(unittest.TestCase):
    """Сквозная проверка: curation → highlights → чтение, на временном каталоге."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.catalog = Path(self.folder.name)
        self.photos = self.catalog / 'photos'
        self.photos.mkdir()
        db = sqlite3.connect(self.catalog / 'catalog.sqlite')
        db.executescript('''
            CREATE TABLE photos (path TEXT PRIMARY KEY, size INTEGER, modified INTEGER, model TEXT,
              status TEXT, error TEXT, kind TEXT NOT NULL DEFAULT 'photo', blocked INTEGER DEFAULT 0);
            CREATE TABLE photo_analysis (path TEXT PRIMARY KEY, size INTEGER, modified INTEGER,
              width INTEGER, height INTEGER, blur_score REAL, content_type TEXT,
              content_confidence REAL, status TEXT, analyzed_at TEXT, embedding_model TEXT);
            CREATE TABLE photo_embeddings (path TEXT, model TEXT, size INTEGER, modified INTEGER,
              embedding BLOB, dims INTEGER, analyzed_at TEXT, PRIMARY KEY(path, model));
            CREATE TABLE photo_adult_analysis (path TEXT PRIMARY KEY, rating TEXT, status TEXT,
              analyzed_at TEXT);
            CREATE TABLE faces (id INTEGER PRIMARY KEY, path TEXT, box TEXT, frame_time REAL);
            CREATE TABLE face_people (face_id INTEGER, person_id INTEGER);
            CREATE TABLE hidden_photos (path TEXT PRIMARY KEY, owner TEXT, stored TEXT, hidden_at REAL);
            CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, changed REAL);
        ''')
        from PIL import Image
        rng = np.random.default_rng(1)
        # Три дня июля: по 12 кадров, у каждого дня своя сцена, внутри — разные ракурсы.
        self.paths = []
        for day in range(3):
            for number in range(12):
                moment = datetime(2024, 7, 3 + day * 7, 10) + timedelta(minutes=10 * number)
                path = self.photos / f'IMG_{moment:%Y%m%d_%H%M%S}.jpg'
                pixels = (rng.random((600, 800, 3)) * 255).astype('uint8')
                Image.fromarray(pixels).save(path, quality=90)
                stat = path.stat()
                key = str(path)
                self.paths.append(key)
                vector = scene_vector(day * 3 + number % 3, 0.2, seed=day * 100 + number)
                db.execute("INSERT INTO photos(path,size,modified,status) VALUES(?,?,?,'ok')",
                           (key, stat.st_size, stat.st_mtime_ns))
                db.execute("INSERT INTO photo_analysis VALUES(?,?,?,800,600,?,'photo',0.5,'ok','t1',?)",
                           (key, stat.st_size, stat.st_mtime_ns, 300.0 + number * 20, MODEL))
                db.execute('INSERT INTO photo_embeddings VALUES(?,?,?,?,?,?,?)',
                           (key, MODEL, stat.st_size, stat.st_mtime_ns,
                            vector.astype('<f4').tobytes(), len(vector), 't1'))
                db.execute("INSERT INTO photo_adult_analysis VALUES(?,'safe','ok','t1')", (key,))
        db.execute('INSERT INTO faces(id,path,box,frame_time) VALUES(1,?,?,NULL)',
                   (self.paths[0], json.dumps([100, 100, 400, 450])))
        db.commit()
        db.close()

    def tearDown(self):
        self.folder.cleanup()

    def curate(self, **extra):
        return pc.curate(self.catalog, log=lambda message: None, workers=2, **extra)

    def test_curation_is_incremental(self):
        first = self.curate()
        self.assertEqual((first['total'], first['read']), (36, 36))
        self.assertEqual(self.curate()['total'], 0)
        # Человека назвали — переоценка без чтения файла.
        db = sqlite3.connect(self.catalog / 'catalog.sqlite')
        db.execute('INSERT INTO face_people VALUES(1, 7)')
        db.commit()
        db.close()
        renamed = self.curate()
        self.assertEqual((renamed['total'], renamed['read']), (1, 0))
        record = pc.explain(self.catalog, self.paths[0])
        self.assertEqual(record['known_people'], 1)
        self.assertEqual(record['taken_source'], 'filename')
        forced = self.curate(force=True)
        self.assertEqual(forced['read'], 36)

    def test_generation_saves_replaces_and_explains(self):
        self.curate()
        db = hg.connect(self.catalog)
        try:
            stats = hg.regenerate(self.catalog, today=date(2026, 7, 10), db=db,
                                  params={'min_base_score': 0.0, 'quality_floor': 0.0, 'min_event_score': 0.0,
                                          'min_gain': -1})
            self.assertEqual(stats['candidates'], 36)
            groups, _ = hg.list_groups(db)
            keys = {group['key'] for group in groups}
            self.assertIn('month:2024-07', keys)
            self.assertIn('year:2024', keys)
            self.assertIn('on-this-day:07-10:2024', keys)
            self.assertEqual(sum(key.startswith('event:') for key in keys), 3)
            month = hg.get_group(db, 'month:2024-07')
            paths = [photo['path'] for photo in month['photos']]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertTrue(all('reasons' in photo for photo in month['photos']))
            self.assertEqual([photo['position'] for photo in month['photos']],
                             list(range(len(paths))))
            # Все три события месяца представлены.
            events = {photo['reasons']['event'] for photo in month['photos']}
            self.assertEqual(len(events), 3)

            # Пересборка сохраняет id, а пропавшие подборки удаляет.
            old_id = month['id']
            hg.regenerate(self.catalog, today=date(2026, 1, 1), db=db,
                          params={'min_base_score': 0.0, 'quality_floor': 0.0, 'min_gain': -1,
                                          'min_event_score': 0.0})
            self.assertEqual(hg.get_group(db, 'month:2024-07')['id'], old_id)
            self.assertIsNone(hg.get_group(db, 'on-this-day:07-10:2024'))

            # Пустой результат (например, не досчитана проверка 18+) старое не стирает.
            before = hg.list_groups(db)[1]
            db.execute('DELETE FROM photo_adult_analysis')
            db.commit()
            empty = hg.regenerate(self.catalog, db=db)
            self.assertTrue(empty['kept_previous'])
            self.assertEqual(hg.list_groups(db)[1], before)

            # Остановленная пересборка старое не трогает.
            before = hg.list_groups(db)[1]
            stopped = hg.regenerate(self.catalog, db=db, stop=lambda: True)
            self.assertTrue(stopped.get('stopped'))
            self.assertEqual(hg.list_groups(db)[1], before)

            report = hg.explain(db, paths[0], params={'min_base_score': 0.0, 'quality_floor': 0.0, 'min_event_score': 0.0,
                                                     'require_adult_check': False,
                                                     'min_gain': -1})
            self.assertEqual(report['decisions']['month']['decision'], 'selected')
        finally:
            db.close()

    def test_hidden_and_unchecked_photos_are_left_out(self):
        self.curate()
        db = hg.connect(self.catalog)
        try:
            db.execute("INSERT INTO hidden_photos VALUES(?,'me','x',0)", (self.paths[0],))
            db.execute('DELETE FROM photo_adult_analysis WHERE path=?', (self.paths[1],))
            db.commit()
            items, stats = hg.load_candidates(db, {'min_base_score': 0.0})
            self.assertEqual((stats['hidden'], stats['adult_unchecked']), (1, 1))
            self.assertNotIn(self.paths[0], {item.path for item in items})
            items, stats = hg.load_candidates(db, {'min_base_score': 0.0,
                                                   'require_adult_check': False})
            self.assertIn(self.paths[1], {item.path for item in items})
        finally:
            db.close()


class PipelineTests(unittest.TestCase):
    def test_highlights_pull_curation_and_visual_for_photos_only(self):
        supported = {name: True for name in job_features.FEATURES}
        selected, kinds = job_features.resolve({'highlights': True}, None, supported)
        self.assertTrue(selected['curation'] and selected['visual'])
        self.assertEqual(kinds['highlights'], 'photos')
        self.assertEqual(kinds['visual'], 'photos')

    def test_phase_order(self):
        plan = device_job.planned_phases({'visual': True, 'adult': True, 'curation': True,
                                          'highlights': True})
        self.assertEqual(plan, ['inventory', 'visual', 'adult', 'curation', 'highlights'])
        self.assertEqual(device_job.planned_phases({'highlights': True}), ['highlights'])


if __name__ == '__main__':
    unittest.main()
