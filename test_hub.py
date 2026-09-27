"""Хаб и ядро: удалённый каталог, перенос каталогов, данные по источникам."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest

import catalogdb
import hub
import hublink
import migrate
import pathkeys
import sources


def link_server(root):
    """Хаб с одним ядром и его рабочая папка с hub.json — для удалённого каталога."""
    catalog = root / 'hub-catalog'
    catalog.mkdir()
    sqlite3.connect(catalog / 'catalog.sqlite').close()
    state = hub.Hub(catalog, root / 'hub-data', link_url='')
    core = state.cores.save({'id': 'pc-t', 'name': 'PC-T', 'host': '127.0.0.1'})
    server = hub.serve_link(state, '127.0.0.1', 0)
    work = root / 'core'
    work.mkdir()
    (work / 'hub.json').write_text(json.dumps({
        'url': f'http://127.0.0.1:{server.server_port}', 'token': core['link_token'],
        'core': 'pc-t'}), encoding='utf-8')
    return state, server, work


class RemoteCatalogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.hub, self.server, self.work = link_server(self.root)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def test_same_behaviour_as_sqlite(self):
        db = catalogdb.connect(self.work, timeout=5)
        self.assertIsInstance(db, catalogdb.RemoteConnection)
        db.executescript('CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, blob BLOB);')
        cursor = db.execute('INSERT INTO t(name,blob) VALUES(?,?)', ('а', b'\x00\x01'))
        self.assertEqual(cursor.lastrowid, 1)
        self.assertTrue(db.in_transaction)
        db.commit()
        self.assertFalse(db.in_transaction)
        with db:
            db.executemany('INSERT INTO t(name) VALUES(?)', ((f'n{index}',) for index in range(3000)))
        self.assertEqual(db.execute('SELECT COUNT(*) FROM t').fetchone()[0], 3001)
        rows = list(db.execute('SELECT id,name FROM t ORDER BY id'))
        self.assertEqual(len(rows), 3001)
        self.assertEqual(rows[0], (1, 'а'))
        self.assertEqual(db.execute('SELECT blob FROM t WHERE id=1').fetchone()[0], b'\x00\x01')
        cursor = db.execute('SELECT id FROM t ORDER BY id')
        self.assertEqual(cursor.fetchmany(5), [(1,), (2,), (3,), (4,), (5,)])
        self.assertEqual(cursor.description[0][0], 'id')
        with self.assertRaises(sqlite3.IntegrityError):
            db.execute('INSERT INTO t(id,name) VALUES(1,?)', ('dup',))
        with self.assertRaises(sqlite3.OperationalError):
            db.execute('SELECT nope FROM t')
        try:
            with db:
                db.execute('DELETE FROM t')
                raise RuntimeError('откат')
        except RuntimeError:
            pass
        self.assertEqual(db.execute('SELECT COUNT(*) FROM t').fetchone()[0], 3001)
        named = db.execute('SELECT name FROM t WHERE id=:id', {'id': 2}).fetchone()
        self.assertEqual(named, ('n0',))
        db.close()
        local = sqlite3.connect(self.hub.database)
        self.assertEqual(local.execute('SELECT COUNT(*) FROM t').fetchone()[0], 3001)
        local.close()

    def test_heartbeat_keeps_open_transaction(self):
        # Heartbeat ядра раз в 5 минут не должен рвать транзакцию задания,
        # а запуск ядра — закрывает прежние соединения.
        db = catalogdb.connect(self.work, timeout=5)
        db.executescript('CREATE TABLE t (id INTEGER PRIMARY KEY);')
        db.execute('INSERT INTO t(id) VALUES(1)')
        link = hublink.link_for(self.work)
        link.json('POST', '/hello', {'version': 'x', 'startup': False})
        db.commit()
        self.assertEqual(db.execute('SELECT COUNT(*) FROM t').fetchone()[0], 1)
        db.execute('INSERT INTO t(id) VALUES(2)')
        link.json('POST', '/hello', {'version': 'x', 'startup': True})
        with self.assertRaises(sqlite3.OperationalError):
            db.commit()
        db.close()

    def test_unknown_core_is_refused(self):
        (self.work / 'hub.json').write_text(json.dumps({
            'url': f'http://127.0.0.1:{self.server.server_port}', 'token': 'bad'}),
            encoding='utf-8')
        with self.assertRaises(hublink.HubError):
            catalogdb.connect(self.work)

    def test_catalog_files_and_thumbs(self):
        import base64
        import catalogfiles
        target = catalogfiles.write_bytes(self.work, 'thumbnails/ab.jpg', b'jpeg')
        self.assertTrue(target.is_file())
        self.assertEqual((self.hub.catalog / 'thumbnails' / 'ab.jpg').read_bytes(), b'jpeg')
        target.unlink()
        self.assertEqual(catalogfiles.read_bytes(self.work, 'thumbnails/ab.jpg'), b'jpeg')
        link = hublink.link_for(self.work)
        link.json('POST', '/thumbs', {'items': [{
            'path': 'nas:/a.jpg', 'size': 3, 'modified': 7, 'width': 10, 'height': 20,
            'thumb_width': 5, 'thumb_height': 10, 'data': base64.b64encode(b'xyz').decode(),
            'metadata': {'kind': 'photo'}}]})
        self.assertEqual(self.hub.thumb_file('nas:/a.jpg').read_bytes(), b'xyz')


class MigrateTest(unittest.TestCase):
    def catalog(self, path, face_ids, person, label):
        db = sqlite3.connect(path)
        db.executescript('''
            CREATE TABLE photos (path TEXT PRIMARY KEY, size INTEGER, modified INTEGER,
              model TEXT, status TEXT, error TEXT, dir TEXT);
            CREATE TABLE faces (id INTEGER PRIMARY KEY, path TEXT, box TEXT, embedding BLOB,
              thumbnail TEXT);
            CREATE TABLE face_clusters (face_id INTEGER PRIMARY KEY, label INTEGER NOT NULL,
              probability REAL NOT NULL DEFAULT 0, method TEXT NOT NULL DEFAULT 'x',
              computed_at TEXT NOT NULL);
            CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
              created_at TEXT NOT NULL, bigfam_id TEXT);
            CREATE TABLE face_people (face_id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL,
              source TEXT NOT NULL DEFAULT 'human', score REAL, version INTEGER);
            CREATE TABLE people_albums (id INTEGER PRIMARY KEY, parent_id INTEGER NOT NULL
              DEFAULT 0, title TEXT NOT NULL, created REAL NOT NULL, hidden INTEGER NOT NULL
              DEFAULT 0, UNIQUE(parent_id,title));
            CREATE TABLE people_album_members (album_id INTEGER NOT NULL, group_key TEXT NOT NULL,
              added REAL NOT NULL, PRIMARY KEY(album_id,group_key));
            CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, changed REAL);
            CREATE TABLE scan_runs (id INTEGER PRIMARY KEY, roots_json TEXT NOT NULL,
              paths_json TEXT NOT NULL, first_run_at TEXT NOT NULL, last_run_at TEXT NOT NULL,
              runs INTEGER NOT NULL DEFAULT 1, last_features TEXT NOT NULL,
              done_features TEXT NOT NULL DEFAULT '[]', last_status TEXT NOT NULL DEFAULT 'x');
        ''')
        db.execute("INSERT INTO people VALUES(1,?,'t',NULL)", (person,))
        for face_id in face_ids:
            path = f'D:\\Фото\\{face_id}.jpg'
            db.execute("INSERT OR IGNORE INTO photos VALUES(?,1,1,'m','ok',NULL,'D:\\Фото')",
                       (path,))
            db.execute('INSERT INTO faces VALUES(?,?,?,?,?)', (face_id, path, '[0,0,1,1]', b'v',
                                                               f'thumbnails/{face_id}.jpg'))
            db.execute("INSERT INTO face_clusters VALUES(?,?,1,'x','t')", (face_id, label))
        db.execute('INSERT INTO face_people VALUES(?,1,?,NULL,NULL)', (face_ids[0], 'human'))
        db.execute("INSERT INTO people_albums VALUES(1,0,'Семья',1,0)")
        db.execute("INSERT INTO people_album_members VALUES(1,'person:1',1)")
        db.execute("INSERT INTO people_album_members VALUES(1,?,1)", (f'auto:{label}',))
        db.execute('INSERT INTO settings VALUES(?,?,0)', ('block_paths', json.dumps('D:\\Игры')))
        db.execute("INSERT INTO scan_runs(roots_json,paths_json,first_run_at,last_run_at,"
                   "last_features) VALUES(?,?,'t','t','[]')", (json.dumps(['D:\\']), '[]'))
        db.commit()
        db.close()

    def test_prefix_then_merge(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.catalog(root / 'x.sqlite', [1, 2], 'Анна', 0)
            self.catalog(root / 'a.sqlite', [1, 5], 'Анна', 0)
            hub_catalog = root / 'hub.sqlite'
            self.assertEqual(migrate.merge(hub_catalog, root / 'x.sqlite', 'pc-x'),
                             {'created': True})
            result = migrate.merge(hub_catalog, root / 'a.sqlite', 'pc-a')
            self.assertEqual(result['offsets']['faces'], 2)
            db = sqlite3.connect(hub_catalog)
            paths = sorted(row[0] for row in db.execute('SELECT path FROM faces'))
            self.assertEqual(paths, ['pc-a:D:\\Фото\\1.jpg', 'pc-a:D:\\Фото\\5.jpg',
                                     'pc-x:D:\\Фото\\1.jpg', 'pc-x:D:\\Фото\\2.jpg'])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM people').fetchone()[0], 1)
            self.assertEqual(sorted(row[0] for row in db.execute('SELECT face_id FROM face_people')),
                             [1, 3])
            labels = dict(db.execute('SELECT face_id,label FROM face_clusters'))
            self.assertEqual(labels, {1: 0, 2: 0, 3: 1, 7: 1})
            members = sorted(row[0] for row in db.execute(
                'SELECT group_key FROM people_album_members'))
            self.assertEqual(members, ['auto:0', 'auto:1', 'person:1'])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM people_albums').fetchone()[0], 1)
            rules = json.loads(db.execute(
                "SELECT value FROM settings WHERE key='block_paths'").fetchone()[0])
            self.assertEqual(rules.split('\n'), ['pc-x:D:\\Игры', 'pc-a:D:\\Игры'])
            roots = [json.loads(row[0]) for row in db.execute('SELECT roots_json FROM scan_runs')]
            self.assertEqual(roots, [['pc-x:D:\\'], ['pc-a:D:\\']])
            self.assertEqual(db.execute("SELECT dir FROM photos WHERE path LIKE 'pc-a:%' LIMIT 1")
                             .fetchone()[0], 'pc-a:D:\\Фото')
            db.close()


class StorageTest(unittest.TestCase):
    def test_stats_and_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / 'catalog'
            catalog.mkdir()
            db = sqlite3.connect(catalog / 'catalog.sqlite')
            db.executescript('''
                CREATE TABLE photos (path TEXT PRIMARY KEY, size INTEGER, modified INTEGER,
                  model TEXT, status TEXT, kind TEXT, dir TEXT);
                CREATE TABLE faces (id INTEGER PRIMARY KEY, path TEXT, thumbnail TEXT);
                CREATE TABLE face_people (face_id INTEGER PRIMARY KEY, person_id INTEGER);
                CREATE TABLE photo_analysis (path TEXT PRIMARY KEY, caption_status TEXT);
            ''')
            db.execute("INSERT INTO photos VALUES('nas:/a.jpg',1,1,'m','ok','photo','nas:/')")
            db.execute("INSERT INTO photos VALUES('pc-x:D:\\b.jpg',1,1,'m','ok','photo','pc-x:D:\\')")
            db.execute("INSERT INTO faces VALUES(1,'nas:/a.jpg','thumbnails/1.jpg')")
            db.execute("INSERT INTO face_people VALUES(1,1)")
            db.execute("INSERT INTO photo_analysis VALUES('nas:/a.jpg','ok')")
            db.commit()
            db.close()
            state = hub.Hub(catalog, root / 'data')
            state.sources.save({'id': 'nas', 'name': 'NAS', 'type': 'smb', 'host': 'x'})
            db = sqlite3.connect(catalog / 'catalog.sqlite')
            stats = {item['id']: item for item in state.storage(db)['sources']}
            self.assertEqual(stats['nas']['photos'], 1)
            self.assertEqual(stats['nas']['faces'], 1)
            self.assertEqual(stats['nas']['named_faces'], 1)
            self.assertEqual(stats['pc-x']['photos'], 1)
            state.clean(db, 'nas', ['analysis', 'faces'])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM faces').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM photo_analysis').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM photos').fetchone()[0], 2)
            state.clean(db, 'nas', ['catalog'])
            self.assertEqual([row[0] for row in db.execute('SELECT path FROM photos')],
                             ['pc-x:D:\\b.jpg'])
            db.close()


class TrustKeyTest(unittest.TestCase):
    def make(self, root):
        catalog = root / 'catalog'
        catalog.mkdir()
        state = hub.Hub(catalog, root / 'data', ssh_key=str(root / 'ssh' / 'id_ed25519'))
        state.cores.save({'id': 'pc-t', 'name': 'PC-T', 'host': '127.0.0.1',
                          'sshUser': 'me', 'sshPassword': 'secret'})
        return state

    def installer(self, state, verified):
        job = hub.Installer(state, state.cores.get('pc-t'), 'key')
        job.scripts = []
        job.run_command = lambda client, command, timeout=0: job.scripts.append(command) or 0

        class Client:
            def close(self):
                pass

        def connect(key_only=False):
            if key_only and not verified:
                raise OSError('Authentication failed')
            return Client()
        job.connect = connect
        return job

    def test_key_is_generated_and_password_forgotten(self):
        with tempfile.TemporaryDirectory() as temp:
            state = self.make(Path(temp))
            key = state.ensure_ssh_key()
            self.assertTrue(key.is_file())
            self.assertTrue(state.public_key().startswith('ssh-ed25519 '))
            self.assertEqual(state.ensure_ssh_key(), key)
            self.assertIn(state.public_key(), hub.trust_key_script(state.public_key()))
            job = self.installer(state, verified=True)
            job.run()
            self.assertEqual(job.state['status'], 'completed', job.state['log'])
            self.assertEqual(len(job.scripts), 1)
            self.assertNotIn('password', state.cores.get('pc-t')['ssh'])
            self.assertEqual(state.cores.get('pc-t')['ssh']['user'], 'me')

    def test_password_kept_when_key_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            state = self.make(Path(temp))
            state.ensure_ssh_key()
            job = self.installer(state, verified=False)
            job.run()
            self.assertEqual(job.state['status'], 'error')
            self.assertEqual(state.cores.get('pc-t')['ssh']['password'], 'secret')

    def test_odd_key_is_refused(self):
        with self.assertRaises(ValueError):
            hub.trust_key_script("ssh-ed25519 AAAA'; rm -rf /")


class FakeCores:
    """Хаб для Parallel: два ядра, задания заканчиваются после одного опроса."""

    def __init__(self, capabilities, fail=None):
        self.rows = [{'id': 'pc-x', 'name': 'PC-X', 'primary': True},
                     {'id': 'pc-a', 'name': 'PC-A'}]
        self.capabilities = capabilities
        self.fail = fail
        self.calls = []
        self.polls = {}

    def online_cores(self):
        return [(row, {'online': True, 'device': {'capabilities': self.capabilities[row['id']]},
                       'job': {'active': False}}) for row in self.rows]

    def pick_core(self, source_id=None):
        return self.rows[0]

    def core_call(self, core, path, method='GET', body=None, timeout=10):
        core_id = core if isinstance(core, str) else core['id']
        self.calls.append((core_id, path, body))
        self.polls[core_id] = 0
        return {'job': {}}

    def core_status(self, core_id, max_age=0):
        self.polls[core_id] += 1
        if self.polls[core_id] < 2:
            return {'online': True, 'job': {'active': True, 'phase': 'visual'}}
        status = 'error' if core_id == self.fail else 'completed'
        return {'online': True, 'job': {'active': False, 'status': status}}


class ParallelTest(unittest.TestCase):
    def run_job(self, fake, features):
        from unittest import mock
        job = hub.Parallel(fake)
        with mock.patch.object(hub.time, 'sleep'):
            job.start(['nas:/photo'], features)
            deadline = time.time() + 10
            while job.status()['status'] == 'running' and time.time() < deadline:
                time.sleep(0.01)
        return job.status()

    def test_inventory_shards_then_highlights(self):
        both = {'visual': True, 'faces': True}
        fake = FakeCores({'pc-x': both, 'pc-a': both})
        state = self.run_job(fake, {'visual': True, 'faces': True, 'highlights': True})
        self.assertEqual(state['status'], 'completed', state)
        starts = [(core, body) for core, path, body in fake.calls if path == '/api/device/job/start']
        self.assertEqual(starts[0][0], 'pc-x')
        self.assertEqual(starts[0][1]['features'], {'inventory': True})
        shards = {core: body['shard'] for core, body in starts[1:3]}
        self.assertEqual(shards, {'pc-x': {'index': 0, 'count': 2}, 'pc-a': {'index': 1, 'count': 2}})
        self.assertEqual(starts[1][1]['features'], {'faces': True, 'visual': True})
        self.assertEqual(starts[3][1]['features'], {'highlights': True})
        self.assertEqual(starts[3][1]['shard'], {'index': 0, 'count': 1})

    def test_core_without_capability_is_left_out(self):
        fake = FakeCores({'pc-x': {'visual': True, 'caption': True}, 'pc-a': {'visual': True}})
        state = self.run_job(fake, {'visual': True, 'caption': True})
        self.assertEqual(state['cores'], ['pc-x'])
        self.assertIn('PC-A: нет caption', state['skipped'])

    def test_failed_share_fails_job(self):
        both = {'visual': True}
        fake = FakeCores({'pc-x': both, 'pc-a': both}, fail='pc-a')
        state = self.run_job(fake, {'visual': True})
        self.assertEqual(state['status'], 'error')
        self.assertIn('PC-A', state['error'])


class HostCheckTest(unittest.TestCase):
    def test_hub_accepts_compose_service_name(self):
        import types
        import web_server

        def request(host, hub_state):
            return types.SimpleNamespace(headers={'Host': host},
                                         app=types.SimpleNamespace(hub=hub_state))
        allowed = web_server.Handler.allowed_host
        self.assertTrue(allowed(request('homecloud-hub:18400', object())))
        self.assertFalse(allowed(request('evil.example:18400', None)))
        self.assertTrue(allowed(request('192.168.1.10:18311', None)))


class PersonFilterTest(unittest.TestCase):
    def test_photos_of_person_take_size_from_thumbs(self):
        import types
        import web_server
        db = sqlite3.connect(':memory:')
        db.executescript('''
            CREATE TABLE photos (path TEXT PRIMARY KEY, status TEXT, modified INTEGER,
                kind TEXT, duration REAL, size INTEGER);
            CREATE TABLE faces (id INTEGER PRIMARY KEY, path TEXT);
            CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE face_people (face_id INTEGER, person_id INTEGER);
            CREATE TABLE photo_analysis (path TEXT, content_type TEXT, blur_score REAL,
                caption TEXT, ocr_text TEXT, ocr_status TEXT, caption_status TEXT,
                caption_short TEXT, caption_search TEXT, caption_tags_json TEXT,
                caption_json TEXT, width INTEGER, height INTEGER);
            CREATE TABLE photo_adult_analysis (path TEXT, rating TEXT, adult_score REAL,
                tags_json TEXT, regions_json TEXT, description TEXT);
            CREATE TABLE video_speech (path TEXT, text TEXT);''')
        db.executescript(hub.THUMBS_SCHEMA)
        db.execute("INSERT INTO photos VALUES ('pc-x:D:\\a.jpg','ok',1,'photo',NULL,10)")
        db.execute("INSERT INTO faces VALUES (1,'pc-x:D:\\a.jpg')")
        db.execute("INSERT INTO people VALUES (1,'Хамис')")
        db.execute('INSERT INTO face_people VALUES (1,1)')
        db.execute("INSERT INTO photo_thumbs(path,size,modified,width,height,created_at) "
                   "VALUES ('pc-x:D:\\a.jpg',10,1,4000,3000,0)")
        app = types.SimpleNamespace(store=types.SimpleNamespace(db=db),
                                    PHOTO_COLUMNS=web_server.App.PHOTO_COLUMNS)
        rows = web_server.App._search_rows(app, ['Хамис'], '', " AND photos.status='ok'", [])
        self.assertEqual([(row[0], row[-2], row[-1]) for row in rows],
                         [('pc-x:D:\\a.jpg', 4000, 3000)])


class FaceCropsTest(unittest.TestCase):
    def test_square_with_margins_stays_inside_frame(self):
        import face_crops
        self.assertEqual(face_crops.square([100, 100, 200, 150], 1000, 800),
                         (70, 45, 230, 205))
        # У края квадрат сдвигается внутрь кадра, а не обрезается.
        self.assertEqual(face_crops.square([0, 0, 100, 100], 1000, 800), (0, 0, 160, 160))
        self.assertEqual(face_crops.square([0, 0, 100, 100], 120, 90), (5, 0, 95, 90))

    def test_recut_replaces_tight_thumbnails(self):
        import face_crops
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / 'thumbnails').mkdir()
            db = sqlite3.connect(':memory:')
            db.execute('CREATE TABLE faces (id INTEGER PRIMARY KEY, path TEXT, box TEXT, '
                       'frame_time REAL, thumbnail TEXT)')
            db.execute("INSERT INTO faces VALUES (1,'box:a.jpg','[10,10,50,50]',NULL,'thumbnails/old.jpg')")
            db.execute("INSERT INTO faces VALUES (2,'box:a.jpg','[60,60,90,90]',NULL,'thumbnails/x-p.jpg')")
            opened = []

            def open_image(key, moment):
                opened.append((key, moment))
                return Image.new('RGB', (200, 100))
            self.assertEqual(face_crops.recut(db, data, ['box:a.jpg'], open_image), 1)
            self.assertEqual(opened, [('box:a.jpg', None)])
            name = db.execute('SELECT thumbnail FROM faces WHERE id=1').fetchone()[0]
            self.assertTrue(face_crops.padded(name))
            with Image.open(data / name) as thumb:
                self.assertEqual(thumb.size, (64, 64))
            self.assertEqual(face_crops.recut(db, data, ['box:a.jpg'], open_image), 0)


class PathRulesTest(unittest.TestCase):
    def test_rules_with_and_without_source(self):
        import pathrules
        block, allow = pathrules.prepare({
            'block_paths': 'D:\\Games\npc-x:D:\\trash\\Apps', 'allow_paths': ''})
        self.assertTrue(pathrules.blocked('pc-x:D:\\Games\\a.jpg', block, allow))
        self.assertTrue(pathrules.blocked('pc-a:D:\\Games\\a.jpg', block, allow))
        self.assertTrue(pathrules.blocked('D:\\Games\\a.jpg', block, allow))
        self.assertTrue(pathrules.blocked('pc-x:D:\\trash\\Apps\\c.jpg', block, allow))
        self.assertFalse(pathrules.blocked('pc-a:D:\\trash\\Apps\\c.jpg', block, allow))
        self.assertFalse(pathrules.blocked('netcraze:/HDD/photo/a.jpg', block, allow))
        block, allow = pathrules.prepare({'block_paths': 'D:\\trash',
                                          'allow_paths': 'D:\\trash\\Наши'})
        self.assertTrue(pathrules.enter('pc-x:D:\\trash', block, allow))
        self.assertFalse(pathrules.blocked('pc-x:D:\\trash\\Наши\\a.jpg', block, allow))


class JunctionLoopTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'win32', 'junction есть только в Windows')
    def test_walk_skips_junction_loop(self):
        import subprocess
        import catalog_index
        with tempfile.TemporaryDirectory() as temp:
            photos = Path(temp) / 'Users' / 'All Users'
            photos.mkdir(parents=True)
            (photos / 'a.jpg').write_bytes(b'1')
            # Как в старом профиле Windows: «Application Data» ведёт на свою же папку.
            made = subprocess.run(['cmd', '/c', 'mklink', '/J', str(photos / 'Application Data'),
                                   str(photos)], capture_output=True)
            if made.returncode:
                self.skipTest('mklink /J недоступен')
            record = sources.validate({'id': 'box', 'type': 'local', 'path': temp})
            found, stopped = catalog_index.walk_source(
                pathkeys.make('box', temp), [], access=sources.Access([record]))
            self.assertFalse(stopped)
            self.assertEqual(list(found), [pathkeys.make('box', str(photos / 'a.jpg'))])


class InventoryTest(unittest.TestCase):
    def test_walk_source_folder(self):
        import catalog_index
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            photos = root / 'photos'
            (photos / 'sub').mkdir(parents=True)
            (photos / 'a.jpg').write_bytes(b'1')
            (photos / 'sub' / 'b.png').write_bytes(b'22')
            (photos / 'sub' / 'note.txt').write_bytes(b'x')
            (photos / '.homecloud-trash').mkdir()
            (photos / '.homecloud-trash' / 'c.jpg').write_bytes(b'3')
            record = sources.validate({'id': 'box', 'type': 'local', 'path': str(photos)})
            access = sources.Access([record])
            found, stopped = catalog_index.walk_source(
                pathkeys.make('box', str(photos)), [], access=access)
            self.assertFalse(stopped)
            self.assertEqual(sorted(found), [pathkeys.make('box', str(photos / 'a.jpg')),
                                             pathkeys.make('box', str(photos / 'sub' / 'b.png'))])
            self.assertEqual(found[pathkeys.make('box', str(photos / 'sub' / 'b.png'))][0], 2)

    def test_inventory_marks_videos(self):
        import catalog_index
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            photos = root / 'photos'
            photos.mkdir()
            (photos / 'a.jpg').write_bytes(b'1')
            (photos / 'b.MP4').write_bytes(b'22')
            db = sqlite3.connect(root / 'catalog.sqlite')
            catalog_index.ensure_schema(db)
            # Строка из прежней описи, которая ролик видом не отмечала.
            db.execute("INSERT INTO photos(path,size,modified,status,kind) VALUES(?,0,0,'ok','photo')",
                       (str((photos / 'b.MP4').resolve()),))
            db.commit()
            catalog_index.take_inventory(db, [str(photos)])
            kinds = {Path(path).name: kind for path, kind in db.execute('SELECT path,kind FROM photos')}
            db.close()
            self.assertEqual(kinds, {'a.jpg': 'photo', 'b.MP4': 'video'})


class CompanionsTest(unittest.TestCase):
    """«Часто рядом»: общие файлы, а не лица; скрытые группы не показываются."""

    def test_counts_shared_files(self):
        import threading
        from types import SimpleNamespace
        from unittest import mock
        import web_server
        by_id = {1: (0, 'a:/1.jpg'), 2: (0, 'a:/2.mp4'), 3: (0, 'a:/2.mp4'),
                 4: (0, 'a:/1.jpg'), 5: (0, 'a:/2.mp4'), 6: (0, 'a:/2.mp4'),
                 7: (0, 'a:/2.mp4'), 8: (0, 'a:/3.jpg'), 9: (0, 'a:/1.jpg')}
        groups = [
            {'key': 'person:1', 'kind': 'person', 'name': 'Анна', 'title': 'Анна',
             'face_ids': [1, 2, 3], 'avatar_face': 1},
            {'key': 'person:2', 'kind': 'person', 'name': 'Сергей', 'title': 'Сергей',
             'face_ids': [4, 5, 6, 7], 'avatar_face': None},
            {'key': 'auto:3', 'kind': 'auto', 'name': None, 'title': 'Без имени',
             'face_ids': [8], 'avatar_face': None},
            {'key': 'person:4', 'kind': 'person', 'name': 'Тайна', 'title': 'Тайна',
             'face_ids': [9], 'avatar_face': None},
            {'key': 'noise', 'kind': 'noise', 'name': None, 'title': 'Шум',
             'face_ids': [9], 'avatar_face': None},
        ]
        app = SimpleNamespace(
            lock=threading.RLock(), COMPANION_KINDS=web_server.App.COMPANION_KINDS,
            store=SimpleNamespace(db=None, by_id=by_id, groups=lambda: groups),
            masked_faces=lambda *args: set(), without=web_server.App.without)
        with mock.patch('people_albums.hidden_group_keys', return_value={'person:4'}):
            result = web_server.App.person_companions(app, 'person:1')
        self.assertEqual(result['files'], 2)
        self.assertEqual([(item['key'], item['shared']) for item in result['companions']],
                         [('person:2', 2)])
        self.assertEqual(result['companions'][0]['avatar'], '/media/face-crop/4?size=200')


class SourceHealthTest(unittest.TestCase):
    """Доступность источников: проверка раз в пять минут, а не на каждый кадр."""

    def make_hub(self, drivers):
        import threading
        from types import SimpleNamespace
        records = [{'id': key, 'type': 'smb', 'enabled': True} for key in drivers]
        fake = SimpleNamespace(
            sources=SimpleNamespace(load=lambda: records),
            access=SimpleNamespace(driver=lambda source_id: drivers[source_id],
                                   resolve=lambda key: (drivers[pathkeys.source_of(key)],
                                                        pathkeys.native(key))),
            status_lock=threading.Lock(), status_cache={'pc-a': (0, {'online': False})})
        fake.health = hub.SourceHealth(fake)
        fake.health.TIMEOUT = 0.3
        return fake

    def test_check_all_marks_down_and_hanging_sources(self):
        from types import SimpleNamespace

        def broken():
            raise sources.SourceError('SMB nas: нет связи')

        def hang():
            time.sleep(2)
        fake = self.make_hub({'ok': SimpleNamespace(test=lambda: True),
                              'down': SimpleNamespace(test=broken),
                              'slow': SimpleNamespace(test=hang)})
        started = time.monotonic()
        state = fake.health.check_all()
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertTrue(state['ok']['online'])
        self.assertFalse(state['down']['online'])
        self.assertIn('нет связи', state['down']['error'])
        self.assertFalse(state['slow']['online'])
        self.assertEqual(fake.status_cache, {}, 'ручная проверка заново спрашивает ядра')
        self.assertTrue(fake.health.available('never-checked'))

    def test_health_payload_has_no_connection_details(self):
        from types import SimpleNamespace
        fake = self.make_hub({'nas': SimpleNamespace(test=lambda: True)})
        fake.sources = SimpleNamespace(load=lambda: [
            {'id': 'nas', 'name': 'Netcraze', 'type': 'smb', 'host': '10.0.0.5', 'password': 'x'},
            {'id': 'pc-a', 'name': 'PC-A', 'type': 'device', 'device': 'pc-a'}])
        fake.health.failed('pc-a', 'SSH 10.0.0.9: Unable to connect')
        payload = hub.HubApi(None, fake).health_payload()['sources']
        self.assertEqual(payload['nas'], {'name': 'Netcraze', 'online': True, 'checked_at': None})
        self.assertFalse(payload['pc-a']['online'])
        self.assertNotIn('10.0.0.5', json.dumps(payload))
        self.assertNotIn('10.0.0.9', json.dumps(payload))

    def test_unavailable_source_is_not_touched(self):
        from types import SimpleNamespace
        calls = []
        driver = SimpleNamespace(stat=lambda native: calls.append(native))
        fake = self.make_hub({'nas': driver})
        fake.health.failed('nas', 'timeout')
        media = hub.SourceMedia(fake, 'nas:/HDD/a.jpg')
        self.assertFalse(media.exists())
        with self.assertRaises(sources.SourceError):
            media.open()
        self.assertEqual(calls, [])

    def test_connection_error_marks_source_down(self):
        from types import SimpleNamespace

        def stat(native):
            raise sources.SourceError('SSH 192.168.1.20: Unable to connect')

        def missing(native):
            raise FileNotFoundError(native)
        fake = self.make_hub({'pc-a': SimpleNamespace(stat=stat), 'nas': SimpleNamespace(stat=missing)})
        self.assertFalse(hub.SourceMedia(fake, 'pc-a:F:\a.jpg').exists())
        self.assertFalse(fake.health.available('pc-a'))
        self.assertFalse(hub.SourceMedia(fake, 'nas:/HDD/b.jpg').exists())
        self.assertTrue(fake.health.available('nas'), 'нет файла — это не недоступность')


class CoreStatusCacheTest(unittest.TestCase):
    def test_offline_core_is_remembered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / 'catalog'
            catalog.mkdir()
            sqlite3.connect(catalog / 'catalog.sqlite').close()
            state = hub.Hub(catalog, root / 'data', link_url='')
            core = state.cores.save({'id': 'pc-t', 'name': 'PC-T', 'host': '127.0.0.1'})
            calls = []

            def unreachable(*args, **kwargs):
                calls.append(args[1])
                raise RuntimeError('PC-T недоступно')
            state.core_call = unreachable
            self.assertFalse(state.core_status(core['id'])['online'])
            time.sleep(0.05)
            self.assertFalse(state.core_status(core['id'], max_age=0)['online'])
            self.assertEqual(len(calls), 1, 'второй раз ядро не спрашивали')
            with state.status_lock:
                state.status_cache.pop(core['id'])
            state.core_status(core['id'])
            self.assertEqual(len(calls), 2)

    def test_slow_job_status_keeps_core_online(self):
        # Ядро отвечает, а статус задания не успел: ядро в сети, задание — прошлое.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = root / 'catalog'
            catalog.mkdir()
            sqlite3.connect(catalog / 'catalog.sqlite').close()
            state = hub.Hub(catalog, root / 'data', link_url='')
            core = state.cores.save({'id': 'pc-s', 'name': 'PC-S', 'host': '127.0.0.1'})
            slow = {'job': False}

            def call(core_row, path, *args, **kwargs):
                if path == '/api/device':
                    return {'role': 'core', 'version': 'v'}
                if slow['job']:
                    raise RuntimeError('timed out')
                return {'status': 'completed', 'active': False}
            state.core_call = call
            self.assertEqual(state.core_status(core['id'])['job']['status'], 'completed')
            slow['job'] = True
            status = state.core_status(core['id'], max_age=0)
            self.assertTrue(status['online'])
            self.assertEqual(status['job']['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
