import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import video_identities as vi
from video_tracks import Tracker
from prototype import database, merge_faces


def vector(angle):
    return np.asarray([np.cos(np.radians(angle)), np.sin(np.radians(angle))], dtype='<f4')


def track(fid, angle=0, moments=(0., 1.), core=True, name=None, path='a.mp4'):
    samples = [{'time': time, 'embedding': vector(angle + index), 'quality': .9,
                'reliable': core, 'box': [0,0,100,100]}
               for index,time in enumerate(moments)]
    bank = [s['embedding'] for s in samples] if core else []
    return {'id':fid,'path':path,'start':min(moments),'stop':max(moments),
            'moments':set(moments),'observations':len(moments), 'samples':samples,
            'bank':bank,'match_bank':bank or [vector(angle)],'core':core,
            'named':name,'quality':.9 if core else .1,'first_box':None,'last_box':None}


class StitchTests(unittest.TestCase):
    def test_rejection_cache_matches_exhaustive_complete_support(self):
        for seed in range(12):
            rng=np.random.default_rng(seed)
            tracks={i:track(i,float(rng.uniform(0,180)),
                            (float(i%5),float(i%5+1)),name=(i%2 if i%7==0 else None))
                    for i in range(24)}
            cannot=vi.conflicts(tracks)
            groups={i:{i} for i in tracks}; owners={i:i for i in tracks}
            scores={(a,b):vi.bank_score(tracks[a]['bank'],tracks[b]['bank'])[0]
                    for a in tracks for b in tracks if a<b}
            for (a,b),score in sorted(scores.items(),key=lambda item:(-item[1],*item[0])):
                ga,gb=owners[a],owners[b]
                if ga==gb:continue
                left,right=groups[ga],groups[gb]
                if vi.incompatible(left,right,tracks,cannot):continue
                if min(scores[tuple(sorted((i,j)))] for i in left for j in right)<.3:continue
                groups[ga]|=groups.pop(gb)
                for fid in groups[ga]:owners[fid]=ga
            actual,_,_=vi.stitch(tracks,{'identity_stitch_threshold':.3})
            self.assertEqual({frozenset(g) for g in groups.values()},
                             {frozenset(g['members']) for g in actual})

    def test_same_person_reappears_and_weak_profile_attaches(self):
        tracks={1:track(1),2:track(2,5,(60.,61.)),3:track(3,8,(80.,),False)}
        identities, noise, _ = vi.stitch(tracks)
        self.assertEqual([x['members'] for x in identities],[{1,2,3}])
        self.assertFalse(noise)

    def test_simultaneous_even_identical_vectors_cannot_merge(self):
        identities,_,cannot=vi.stitch({1:track(1),2:track(2)})
        self.assertEqual(len(identities),2)
        self.assertIn((1,2),cannot)

    def test_weak_track_cannot_create_identity(self):
        identities,noise,_=vi.stitch({1:track(1,0,(0.,),False)})
        self.assertFalse(identities); self.assertEqual(noise,{1})

    def test_ambiguous_attachment_stays_unresolved(self):
        tracks={1:track(1,-15),2:track(2,15),3:track(3,0,(3.,),False)}
        identities,noise,_=vi.stitch(tracks)
        self.assertEqual(noise,{3}); self.assertEqual(len(identities),2)

    def test_complete_support_blocks_chain(self):
        identities,_,_=vi.stitch({1:track(1,0),2:track(2,40,(3.,4.)),3:track(3,80,(6.,7.))})
        self.assertEqual(len(identities),2)
        self.assertFalse(any({1,3} <= x['members'] for x in identities))

    def test_manual_names_conflict(self):
        identities,_,_=vi.stitch({1:track(1,name=1),2:track(2,moments=(3.,4.),name=2)})
        self.assertEqual(len(identities),2)

    def test_iou_cannot_switch_identity(self):
        tracker=Tracker()
        tracker.update(0,[([0,0,100,100],vector(0),1.,None)])
        tracker.update(.5,[([0,0,100,100],vector(90),1.,None)])
        self.assertEqual(len(tracker.close_all()),2)

    def test_invalid_vectors_ignored(self):
        tracker=Tracker()
        tracker.update(0,[([0,0,10,10],np.zeros(2),1,None)])
        self.assertFalse(tracker.close_all())

    def test_quality_unknown_is_not_reliable(self):
        self.assertFalse(vi.quality([0,0,100,100])['reliable'])
        json.dumps(vi.quality(np.asarray([0,0,100,100],dtype=np.float32),.99,.2,
                             np.asarray([[20,30],[80,30],[50,50],[30,70],[70,70]])))

    def test_pair_support_does_not_repeat_sample(self):
        _,support=vi.bank_score([vector(0)],[vector(0),vector(1),vector(2)])
        self.assertEqual(support,1)

    def test_decoder_failure_is_not_a_successful_empty_video(self):
        from unittest.mock import MagicMock
        from video_tracks import sample_frames
        capture=MagicMock();capture.isOpened.return_value=True
        capture.read.return_value=(False,None)
        with patch('video.probe',return_value={'fps':30,'frames':300}),patch('video._open',return_value=capture):
            with self.assertRaises(OSError):list(sample_frames('broken.mp4',.5))
        capture.release.assert_called_once()


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.folder=Path(self.temp.name)
        self.db=database(self.folder)

    def tearDown(self):
        self.db.close();self.temp.cleanup()

    def add(self, fid, angle=0, moments=(0.,1.), core=True, path='a.mp4'):
        t=track(fid,angle,moments,core,path=path)
        self.db.execute("INSERT OR IGNORE INTO photos(path,status) VALUES(?,'ok')",(path,))
        self.db.execute('INSERT INTO faces(id,path,box,embedding,frame_time,track_start,track_stop) VALUES(?,?,?,?,?,?,?)',
                        (fid,path,'[0,0,100,100]',vector(angle).tobytes(),moments[0],min(moments),max(moments)))
        for s in t['samples']:
            s.update(size=100,confidence=.99,blur=.2,geometry=1 if core else 0)
        vi.save_track(self.db,fid,{'observations':len(moments),'moments':list(moments),
                                  'first_box':[0,0,100,100],'last_box':[0,0,100,100],
                                  'representatives':t['samples']})
        self.db.commit()

    def test_atomic_stop_and_stable_labels(self):
        self.add(1); self.add(2,5,(3.,4.))
        vi.rebuild(self.db)
        before=self.db.execute('SELECT * FROM face_clusters').fetchall()
        labels=dict(self.db.execute('SELECT face_id,label FROM face_clusters'))
        self.assertGreaterEqual(labels[1],0);self.assertEqual(labels[1],labels[2])
        with self.assertRaises(InterruptedError):vi.rebuild(self.db,stop_check=lambda:True)
        self.assertEqual(before,self.db.execute('SELECT * FROM face_clusters').fetchall())
        vi.rebuild(self.db)
        self.assertEqual(labels,dict(self.db.execute('SELECT face_id,label FROM face_clusters')))

    def test_global_cannot_link_and_single_video_core(self):
        self.add(1);self.add(2)
        result=vi.rebuild(self.db,min_cluster_size=2)
        self.assertNotEqual(result['assignments'][1][0],result['assignments'][2][0])
        self.assertTrue(all(x[0]>=0 for x in result['assignments'].values()))

    def test_legacy_track_is_unresolved(self):
        self.add(1)
        self.db.execute('DELETE FROM face_track_samples');self.db.commit()
        result=vi.rebuild(self.db)
        self.assertEqual(result['assignments'][1][0],-1)

    def test_named_prototypes_not_auto_feedback(self):
        self.add(1,path='known.mp4');self.add(2,path='new.mp4')
        self.db.execute("INSERT INTO people VALUES(1,'Name','now')")
        self.db.execute('INSERT INTO face_people(face_id,person_id) VALUES(1,1)');self.db.commit()
        vi.rebuild(self.db)
        self.assertEqual(self.db.execute('SELECT source FROM face_people WHERE face_id=2').fetchone()[0],'automatic')
        banks=vi.prototype_banks(self.db,vi.load_tracks(self.db),exclude_path='known.mp4')
        self.assertFalse(banks)

    def test_stale_input_cannot_publish(self):
        self.add(1)
        revision=self.db.execute('PRAGMA data_version').fetchone()[0]
        result=vi.build(self.db)
        other=sqlite3.connect(self.folder/'catalog.sqlite')
        other.execute("UPDATE photos SET status='excluded'");other.commit();other.close()
        with self.assertRaises(RuntimeError):vi.publish(self.db,result,revision)
        self.assertEqual(self.db.execute('SELECT count(*) FROM face_clusters').fetchone()[0],0)

    def test_unmatched_manual_track_retained(self):
        self.add(1)
        self.db.execute("INSERT INTO people VALUES(1,'Name','now')")
        self.db.execute('INSERT INTO face_people(face_id,person_id) VALUES(1,1)')
        with self.db:merge_faces(self.db,'a.mp4',[])
        self.assertEqual(self.db.execute('SELECT active FROM face_track_data WHERE face_id=1').fetchone()[0],0)
        self.assertEqual(self.db.execute('SELECT person_id FROM face_people WHERE face_id=1').fetchone()[0],1)

    def test_noise_rebuild_preserves_existing_groups(self):
        self.add(1);vi.rebuild(self.db)
        before=self.db.execute('SELECT * FROM face_clusters WHERE face_id=1').fetchone()
        self.add(2,60,(4.,5.))
        result=vi.rebuild(self.db,scope='noise')
        self.assertNotIn(1,result['assignments'])
        self.assertEqual(before,self.db.execute('SELECT * FROM face_clusters WHERE face_id=1').fetchone())
        self.assertGreaterEqual(result['assignments'][2][0],0)

    def test_repeated_rescan_preserves_face_id_and_name(self):
        self.add(1)
        self.db.execute("INSERT INTO people VALUES(1,'Name','now')")
        self.db.execute('INSERT INTO face_people(face_id,person_id) VALUES(1,1)');self.db.commit()
        t=track(1)
        data={'observations':2,'moments':[0.,1.],'first_box':[0,0,100,100],
              'last_box':[0,0,100,100],'representatives':t['samples']}
        item=('a.mp4','[0,0,100,100]',vector(0).tobytes(),'new.jpg',0.,0.,1.,data)
        for _ in range(2):
            with self.db:merge_faces(self.db,'a.mp4',[item])
        self.assertEqual(self.db.execute('SELECT id FROM faces').fetchall(),[(1,)])
        self.assertEqual(self.db.execute('SELECT source FROM face_people').fetchone()[0],'human')

    def test_exception_during_publish_rolls_back_every_table(self):
        self.add(1);vi.rebuild(self.db)
        before={table:self.db.execute('SELECT * FROM '+table).fetchall()
                for table in ('face_clusters','face_track_identities','video_identities','face_people')}
        result=vi.build(self.db)
        result['names']={1:(999,1.)} # foreign key violation after identity writes
        with self.assertRaises(sqlite3.IntegrityError):
            vi.publish(self.db,result,self.db.execute('PRAGMA data_version').fetchone()[0])
        for table,rows in before.items():
            self.assertEqual(rows,self.db.execute('SELECT * FROM '+table).fetchall())

    def test_global_preserves_complete_support_and_hint_conflict(self):
        self.add(1,0);self.add(2,40,(3.,4.));self.add(3,80,(6.,7.))
        result=vi.rebuild(self.db,min_cluster_size=2)
        self.assertNotEqual(result['assignments'][1][0],result['assignments'][3][0])
        self.assertTrue(vi.hint_status(self.db,'a.mp4',1)['conflict'])

    def test_manual_correction_and_undo_preserve_provenance(self):
        # Some older suites install a lightweight people_gui stub in sys.modules.
        import importlib.util
        spec=importlib.util.spec_from_file_location('identity_people_gui',Path(__file__).with_name('people_gui.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        CatalogStore=module.CatalogStore
        self.add(1,path='known.mp4');self.add(2,path='new.mp4')
        self.db.execute("INSERT INTO people VALUES(1,'Name','now')")
        self.db.execute('INSERT INTO face_people(face_id,person_id) VALUES(1,1)');self.db.commit()
        vi.rebuild(self.db)
        store=CatalogStore(self.folder)
        try:
            store.assign([2],'Other')
            self.assertEqual(store.db.execute('SELECT source FROM face_people WHERE face_id=2').fetchone()[0],'human')
            store.undo()
            self.assertEqual(store.db.execute('SELECT source FROM face_people WHERE face_id=2').fetchone()[0],'automatic')
        finally:store.db.close()

    def test_automatic_name_does_not_train_voice(self):
        import speaker_diarization
        self.add(1)
        self.db.execute("INSERT INTO people VALUES(1,'Name','now')")
        self.db.execute("INSERT INTO face_people(face_id,person_id,source) VALUES(1,1,'automatic')")
        with patch.object(speaker_diarization,'apply_voice_sample') as apply:
            speaker_diarization.update_voice_prints(self.db,{'speaker':(1,1.)},[('speaker',1,vector(0))])
            apply.assert_not_called()


if __name__=='__main__':unittest.main()
