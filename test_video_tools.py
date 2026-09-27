"""Обработка роликов через ffmpeg: решения без ffmpeg и настоящие операции с ним."""
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import hub
import pathkeys
import sources
import video_tools
import web_server


def fake_info(container='mov,mp4,m4a,3gp,3g2,mj2', video='h264', pix_fmt='yuv420p',
              audio=('aac',), duration=10.0, profile='High'):
    return {'container': container, 'duration': duration,
            'video': {'codec': video, 'pix_fmt': pix_fmt, 'profile': profile, 'width': 640,
                      'height': 360, 'fps': 25, 'rotation': 0} if video else None,
            'audio': [{'codec': codec, 'channels': 2, 'language': ''} for codec in audio],
            'subtitles': 0}


class BrowserSupportTest(unittest.TestCase):
    def test_plain_h264_mp4_plays(self):
        self.assertTrue(video_tools.browser_ok(fake_info()))

    def test_hevc_mkv_avi_and_10bit_do_not(self):
        self.assertFalse(video_tools.browser_ok(fake_info(video='hevc')))
        self.assertFalse(video_tools.browser_ok(fake_info(container='avi', video='mpeg4')))
        self.assertFalse(video_tools.browser_ok(fake_info(container='matroska,webm')))
        self.assertFalse(video_tools.browser_ok(fake_info(pix_fmt='yuv420p10le')))
        self.assertFalse(video_tools.browser_ok(fake_info(audio=('ac3',))))

    def test_webm_plays(self):
        self.assertTrue(video_tools.browser_ok(
            fake_info(container='matroska,webm', video='vp9', audio=('opus',))))

    def test_copyable_only_h264_8bit_with_aac(self):
        self.assertTrue(video_tools.copyable(fake_info(container='matroska,webm')))
        self.assertFalse(video_tools.copyable(fake_info(audio=('ac3',))))
        self.assertFalse(video_tools.copyable(fake_info(video='hevc')))

    def test_summarize_reads_streams_and_rotation(self):
        info = video_tools.summarize({
            'format': {'format_name': 'mov,mp4,m4a,3gp,3g2,mj2', 'duration': '12.5',
                       'size': '1000', 'bit_rate': '800'},
            'streams': [
                {'codec_type': 'video', 'codec_name': 'hevc', 'width': 1920, 'height': 1080,
                 'pix_fmt': 'yuv420p', 'avg_frame_rate': '30000/1001',
                 'side_data_list': [{'rotation': -90}]},
                {'codec_type': 'video', 'codec_name': 'mjpeg', 'disposition': {'attached_pic': 1}},
                {'codec_type': 'audio', 'codec_name': 'aac', 'channels': 2,
                 'tags': {'language': 'rus'}},
                {'codec_type': 'subtitle', 'codec_name': 'mov_text'}]})
        self.assertEqual(info['video']['codec'], 'hevc')
        self.assertEqual(info['video']['rotation'], 270)
        self.assertAlmostEqual(info['video']['fps'], 29.97, places=2)
        self.assertEqual(info['audio'][0]['language'], 'rus')
        self.assertEqual(info['subtitles'], 1)
        self.assertFalse(info['browser'])


class ParamsTest(unittest.TestCase):
    def test_trim_checks_order_and_clamps_to_duration(self):
        with self.assertRaises(ValueError):
            video_tools.check_params('trim', {'start': 5, 'end': 3})
        params = video_tools.check_params('trim', {'start': 1, 'end': 99}, {'duration': 10})
        self.assertEqual((params['start'], params['end']), (1.0, 10))

    def test_compress_rotate_and_unknown(self):
        self.assertEqual(video_tools.check_params('compress', {'height': 480})['height'], 480)
        with self.assertRaises(ValueError):
            video_tools.check_params('compress', {'height': 1000})
        self.assertEqual(video_tools.check_params('rotate', {'angle': -90})['angle'], 270)
        with self.assertRaises(ValueError):
            video_tools.check_params('rotate', {'angle': 45})
        with self.assertRaises(ValueError):
            video_tools.check_params('explode', {})

    def test_frame_stays_inside_video(self):
        self.assertEqual(video_tools.check_params('frame', {'time': 50}, {'duration': 10})['time'],
                         9.95)

    def test_progress_lines(self):
        self.assertAlmostEqual(video_tools.parse_progress('out_time_us=5000000', 10), 0.5)
        self.assertEqual(video_tools.parse_progress('progress=end', 10), 1.0)
        self.assertIsNone(video_tools.parse_progress('fps=30', 10))

    def test_output_names(self):
        key = 'nas:/HDD/Видео/Отпуск.mkv'
        self.assertEqual(video_tools.output_name(key, 'trim', {'start': 65, 'end': 3725}),
                         'Отпуск 1.05–1.02.05.mp4')
        self.assertEqual(video_tools.output_name(key, 'audio', {}), 'Отпуск.mp3')

    def test_replace_command_copies_when_it_can(self):
        command = video_tools.build_command('ffmpeg', 'replace', {}, 'a.mkv', 'b.mp4',
                                            fake_info(container='matroska,webm'), 'libx264')
        self.assertIn('copy', command)
        self.assertNotIn('libx264', command)
        command = video_tools.build_command('ffmpeg', 'replace', {}, 'a.mkv', 'b.mp4',
                                            fake_info(video='hevc', audio=('ac3',)), 'h264_nvenc')
        self.assertIn('h264_nvenc', command)
        self.assertIn('aac', command)


class Folder:
    """Драйвер-заглушка для имён: какие файлы «есть» в папке."""
    def __init__(self, names):
        self.names = {name.casefold() for name in names}

    def exists(self, native):
        return native.casefold() in self.names


class ReplacementNameTest(unittest.TestCase):
    def test_other_extension_takes_mp4_name(self):
        self.assertEqual(video_tools.replacement_native(Folder([]), '/HDD/a.mkv'),
                         ('/HDD/a.mp4', '/HDD/a.mp4'))

    def test_taken_name_gets_number(self):
        final, temporary = video_tools.replacement_native(
            Folder(['D:\\v\\a.mp4', 'D:\\v\\a (2).mp4']), 'D:\\v\\a.avi')
        self.assertEqual((final, temporary), ('D:\\v\\a (3).mp4', 'D:\\v\\a (3).mp4'))

    def test_mp4_goes_through_temporary_name(self):
        final, temporary = video_tools.replacement_native(Folder(['/HDD/a.MP4']), '/HDD/a.MP4')
        self.assertEqual(final, '/HDD/a.mp4')
        self.assertEqual(temporary, '/HDD/.a.homecloud-new.mp4')

    def test_trash_path_keeps_smb_share(self):
        self.assertEqual(sources.trash_native({'type': 'smb'}, '/HDD/Видео/a.mkv', '20260927'),
                         '/HDD/.homecloud-trash/20260927/Видео/a.mkv')
        self.assertEqual(sources.trash_native({'type': 'sftp'}, '/home/a.mkv', '20260927'),
                         '/.homecloud-trash/20260927/home/a.mkv')


@unittest.skipUnless(video_tools.available(), 'нет ffmpeg: установите компонент «Видео: ffmpeg»')
class FfmpegJobsTest(unittest.TestCase):
    """Настоящие операции над короткими роликами, которые делает сам ffmpeg."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.videos = cls.root / 'videos'
        cls.videos.mkdir()
        ffmpeg = str(video_tools.find_binary('ffmpeg'))
        source = ['-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=25:duration=4',
                  '-f', 'lavfi', '-i', 'sine=frequency=440:duration=4']
        for name, codecs in (
                ('hevc.mkv', ['-c:v', 'libx265', '-preset', 'ultrafast', '-c:a', 'ac3']),
                ('old.avi', ['-c:v', 'mpeg4', '-c:a', 'libmp3lame']),
                ('ok.mkv', ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac']),
                ('download.mkv', ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac']),
                ('phone.mp4', ['-c:v', 'libx265', '-preset', 'ultrafast', '-tag:v', 'hvc1',
                               '-c:a', 'aac'])):
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', *source,
                            *codecs, '-shortest', str(cls.videos / name)], check=True,
                           creationflags=video_tools.NO_WINDOW)
        cls.trashed = []

        def trash(record, driver, native):
            target = cls.root / 'trash' / Path(native).name
            target.parent.mkdir(exist_ok=True)
            driver.rename(native, str(target))
            cls.trashed.append(Path(native).name)

        access = sources.Access([{'id': 'tt', 'type': 'device', 'device': 'pc-t'}],
                                device_id='pc-t')
        cls.jobs = video_tools.MediaJobs(cls.root / 'media-out', access,
                                         lambda key: pathkeys.native(key), trash)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def key(self, name):
        return pathkeys.make('tt', str(self.videos / name))

    def run_job(self, name, op, params=None):
        job = self.jobs.start(self.key(name), op, params)
        deadline = time.time() + 180
        while time.time() < deadline:
            job = self.jobs.status(job['id'])
            if job['status'] in {'done', 'error', 'cancelled'}:
                break
            time.sleep(0.2)
        self.assertEqual(job['status'], 'done', job.get('error'))
        return job

    def test_probe_sees_hevc(self):
        info = self.jobs.probe(self.key('hevc.mkv'))
        self.assertEqual(info['video']['codec'], 'hevc')
        self.assertFalse(info['browser'])

    def test_replace_mkv_in_source(self):
        job = self.run_job('hevc.mkv', 'replace')
        new = Path(pathkeys.native(job['result']['new']))
        self.assertEqual(new.name, 'hevc.mp4')
        self.assertFalse((self.videos / 'hevc.mkv').exists())
        self.assertIn('hevc.mkv', self.trashed)
        info = video_tools.probe(new)
        self.assertTrue(info['browser'])
        self.assertAlmostEqual(info['duration'], 4, delta=0.5)
        self.assertEqual(job['result']['size'], new.stat().st_size)

    def test_replace_mp4_keeps_name(self):
        job = self.run_job('phone.mp4', 'replace')
        self.assertEqual(job['result']['new'], self.key('phone.mp4'))
        self.assertEqual(video_tools.probe(self.videos / 'phone.mp4')['video']['codec'], 'h264')
        self.assertFalse(any(path.name.startswith('.phone') for path in self.videos.iterdir()))

    def test_replace_avi_and_remux_mkv(self):
        self.assertEqual(Path(pathkeys.native(self.run_job('old.avi', 'replace')['result']['new']))
                         .name, 'old.mp4')
        job = self.run_job('ok.mkv', 'replace')
        self.assertTrue(video_tools.probe(pathkeys.native(job['result']['new']))['browser'])

    def test_download_operations(self):
        source = 'download.mkv'
        trim = self.run_job(source, 'trim', {'start': 1, 'end': 3})
        self.assertIn(trim['encoder'], {'h264_nvenc', 'libx264'})
        path, name = self.jobs.file(trim['id'])
        self.assertAlmostEqual(video_tools.probe(path)['duration'], 2, delta=0.3)
        self.assertEqual(name, 'download 0.01–0.03.mp4')
        compressed, _ = self.jobs.file(self.run_job(source, 'compress', {'height': 360})['id'])
        self.assertEqual(video_tools.probe(compressed)['video']['height'], 360)
        rotated, _ = self.jobs.file(self.run_job(source, 'rotate', {'angle': 90})['id'])
        self.assertEqual(video_tools.probe(rotated)['video']['width'], 360)
        frame, frame_name = self.jobs.file(self.run_job(source, 'frame', {'time': 2})['id'])
        self.assertTrue(frame_name.endswith('.jpg'))
        self.assertEqual(frame.read_bytes()[:2], b'\xff\xd8')
        audio, _ = self.jobs.file(self.run_job(source, 'audio')['id'])
        self.assertEqual(video_tools.probe(audio)['audio'][0]['codec'], 'mp3')

    def test_already_fine_mp4_is_refused(self):
        name = 'fine.mp4'
        ffmpeg = str(video_tools.find_binary('ffmpeg'))
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                        'testsrc2=size=320x240:rate=25:duration=1', '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', str(self.videos / name)], check=True,
                       creationflags=video_tools.NO_WINDOW)
        job = self.jobs.start(self.key(name), 'replace')
        while job['status'] in {'queued', 'running'}:
            time.sleep(0.2)
            job = self.jobs.status(job['id'])
        self.assertEqual(job['status'], 'error')
        self.assertIn('перекодировать нечего', job['error'])


class FakeHub:
    """Хаб без сети: одно ядро, ответы ядра задаются тестом."""
    def __init__(self, job):
        self.job = job
        self.cores = SimpleNamespace(get=lambda core_id: {'id': core_id, 'name': 'PC-T'})
        self.sources = SimpleNamespace(get=lambda source_id: {'id': source_id, 'type': 'smb'})

    def pick_core(self, capability=None, source_id=None, idle=False):
        return {'id': 'pc-t', 'name': 'PC-T'}

    def core_call(self, core, path, method='GET', body=None, timeout=10, raw=False):
        if path == '/api/core/media/start':
            return {**self.job, 'status': 'queued'}
        return dict(self.job)


class HubMediaTest(unittest.TestCase):
    def test_replace_is_finalized_once(self):
        result = {'old': 'nas:/HDD/a.mkv', 'new': 'nas:/HDD/a.mp4', 'size': 5, 'modified': 7}
        fake = FakeHub({'id': 'j1', 'op': 'replace', 'status': 'running', 'result': None})
        calls = []
        media = hub.HubMedia(fake, calls.append)
        media._watch = lambda hub_id: None
        job = media.start('nas:/HDD/a.mkv', 'replace', {})
        self.assertEqual(job['id'], 'pc-t.j1')
        self.assertEqual(media.status('pc-t.j1')['status'], 'running')
        fake.job.update(status='done', result=result)
        self.assertEqual(media.status('pc-t.j1')['status'], 'done')
        self.assertEqual(media.status('pc-t.j1')['download'], '')
        self.assertEqual(calls, [result])

    def test_catalog_failure_is_reported(self):
        fake = FakeHub({'id': 'j2', 'op': 'replace', 'status': 'done',
                        'result': {'old': 'a', 'new': 'b', 'size': 1, 'modified': 1}})

        def broken(result):
            raise sqlite3.OperationalError('database is locked')
        media = hub.HubMedia(fake, broken)
        media._watch = lambda hub_id: None
        media.start('nas:/HDD/a.mkv', 'replace', {})
        job = media.status('pc-t.j2')
        self.assertEqual(job['status'], 'error')
        self.assertIn('каталог не обновился', job['error'])

    def test_download_link_for_files(self):
        fake = FakeHub({'id': 'j3', 'op': 'frame', 'status': 'done',
                        'result': {'name': 'a.jpg', 'size': 1}})
        media = hub.HubMedia(fake, lambda result: None)
        media.start('nas:/HDD/a.mkv', 'frame', {'time': 1})
        self.assertEqual(media.status('pc-t.j3')['download'], '/media/job?id=pc-t.j3')

    def test_device_video_needs_its_own_core(self):
        fake = FakeHub({'id': 'j4', 'op': 'replace', 'status': 'queued'})
        fake.sources = SimpleNamespace(get=lambda source_id: {'type': 'device', 'device': 'pc-a'})
        with self.assertRaises(RuntimeError):
            hub.HubMedia(fake, lambda result: None).start('pc-a:D:\v\a.avi', 'replace', {})


class ReplacedVideoTest(unittest.TestCase):
    def test_keys_move_and_size_updates(self):
        db = sqlite3.connect(':memory:', check_same_thread=False)
        db.executescript("""
            CREATE TABLE photos (path TEXT PRIMARY KEY, dir TEXT, size INTEGER, modified INTEGER);
            CREATE TABLE faces (id INTEGER PRIMARY KEY, path TEXT);
            CREATE TABLE video_speech (path TEXT PRIMARY KEY, text TEXT);
            CREATE TABLE photo_thumbs (path TEXT PRIMARY KEY, size INTEGER, modified INTEGER);
            INSERT INTO photos VALUES ('nas:/HDD/v/a.mkv', 'nas:/HDD/v', 100, 1);
            INSERT INTO faces VALUES (1, 'nas:/HDD/v/a.mkv');
            INSERT INTO video_speech VALUES ('nas:/HDD/v/a.mkv', 'привет');
            INSERT INTO photo_thumbs VALUES ('nas:/HDD/v/a.mkv', 100, 1);
        """)
        renamed = []
        app = SimpleNamespace(
            lock=threading.RLock(), store=SimpleNamespace(db=db, reload_faces=lambda: None),
            RENAMED_TABLES=web_server.App.RENAMED_TABLES,
            hub=SimpleNamespace(rename_thumbs=renamed.extend), group_cache={}, dup_cache={},
            folders=SimpleNamespace(refresh=lambda force=False: None))
        app._rekey = lambda pairs: web_server.App._rekey(app, pairs)
        web_server.App.replaced_video(app, {'old': 'nas:/HDD/v/a.mkv', 'new': 'nas:/HDD/v/a.mp4',
                                            'size': 42, 'modified': 9})
        self.assertEqual(db.execute('SELECT path,dir,size,modified FROM photos').fetchall(),
                         [('nas:/HDD/v/a.mp4', 'nas:/HDD/v', 42, 9)])
        self.assertEqual(db.execute('SELECT path FROM faces').fetchone()[0], 'nas:/HDD/v/a.mp4')
        self.assertEqual(db.execute('SELECT path FROM video_speech').fetchone()[0],
                         'nas:/HDD/v/a.mp4')
        self.assertEqual(db.execute('SELECT size,modified FROM photo_thumbs').fetchone(), (42, 9))
        self.assertEqual(renamed, [('nas:/HDD/v/a.mkv', 'nas:/HDD/v/a.mp4')])


if __name__ == '__main__':
    unittest.main()
