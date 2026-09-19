"""Local diagnostics/calibration. Defaults to read-only; rebuild is explicit.

Examples:
  python diagnose_video_identities.py report --data work/video-identities-validation --output work/video-report
  python diagnose_video_identities.py calibrate --data work/video-identities-validation --output work/video-calibration
  python diagnose_video_identities.py rebuild --data work/video-identities-validation
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3

import numpy as np

import video_identities as vi
import settings
import video


class DecisionSample(list):
    """Bound debug output independently from the number of compared tracks."""
    def __init__(self, limit):
        super().__init__(); self.limit=limit; self.total=0

    def append(self, item):
        self.total+=1
        if len(self)<self.limit:super().append(item)


def metrics(db):
    groups = defaultdict(Counter)
    fragments = defaultdict(set)
    labelled = 0
    for fid,path,person,label in db.execute('''SELECT f.id,f.path,p.person_id,c.label FROM faces f
        JOIN face_people p ON p.face_id=f.id AND p.source='human'
        LEFT JOIN face_clusters c ON c.face_id=f.id'''):
        labelled += 1
        if label is not None and label >= 0:
            groups[label][person] += 1
            if video.is_video(path):
                fragments[(path,person)].add(label)
    covered = sum(sum(x.values()) for x in groups.values())
    wrong = sum(sum(x.values())-max(x.values()) for x in groups.values())
    return {'labelled':labelled,'covered':covered,'wrong':wrong,'error_rate':wrong/max(covered,1),
            'fragment_excess':sum(max(0,len(x)-1) for x in fragments.values())}


def pair_report(db, tracks, options):
    by_path = defaultdict(list)
    for fid,t in tracks.items():by_path[t['path']].append(fid)
    cannot=vi.conflicts(tracks)
    speakers=defaultdict(set)
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='video_speaker_faces'").fetchone():
        for path,speaker,fid,confidence in db.execute('SELECT path,speaker,face_id,confidence FROM video_speaker_faces'):
            if confidence>=.7:speakers[fid].add((path,speaker))
    for path,ids in sorted(by_path.items()):
        for index,a in enumerate(sorted(ids)):
            for b in sorted(ids)[index+1:]:
                x,y=tracks[a],tracks[b]
                score,support=vi.bank_score(x['match_bank'],y['match_bank'],options)
                overlap=(x['start'] is not None and y['start'] is not None
                         and min(x['stop'],y['stop'])>max(x['start'],y['start']))
                known=x['named'] is not None and y['named'] is not None
                yield {'path':path,'track_a':a,'track_b':b,'cosine_similarity':score,
                       'euclidean_distance':float(np.sqrt(max(0,2-2*score))), 'support':support,
                       'quality_a':x['quality'],'quality_b':y['quality'],
                       'temporal_overlap':bool(overlap),'cannot_link':(a,b) in cannot,
                       'speaker_match':bool(speakers[a]&speakers[b]),
                       'truth':('same' if x['named']==y['named'] else 'different') if known else 'unknown'}


def calibration(tracks, options, db=None, progress=None):
    # Whole videos, deterministic split. No frames from a test video in training.
    paths=sorted({t['path'] for t in tracks.values() if t['named'] is not None},
                 key=lambda p:hashlib.sha256(p.encode()).hexdigest())
    test=set(paths[::3]); train=set(paths)-test
    score_cache = {}
    def assess(selected,p):
        local={fid:t for fid,t in tracks.items() if t['path'] in selected}
        # Truth must not act as a cannot-link during evaluation.
        identities,noise,cannot=vi.stitch({i:{**t,'named':None} for i,t in local.items()},p,score_cache=score_cache)
        wrong=covered=0; fragments=defaultdict(set)
        for index,identity in enumerate(identities):
            labels=Counter(local[i]['named'] for i in identity['members'] if local[i]['named'] is not None)
            covered+=sum(labels.values())
            if labels:wrong+=sum(labels.values())-max(labels.values())
            for i in identity['members']:
                if local[i]['named'] is not None:fragments[(local[i]['path'],local[i]['named'])].add(index)
        return {'wrong':wrong,'covered':covered,'fragment_excess':sum(max(0,len(v)-1) for v in fragments.values()),
                'unresolved':len(noise)}
    trials=[]
    thresholds = (.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80,.85,.90)
    for index, threshold in enumerate(thresholds):
        if progress:
            progress({'step':'stitch-threshold','trial':index+1,'trials':len(thresholds),'threshold':threshold})
        p={**options,'identity_stitch_threshold':threshold,
           'identity_attach_threshold':min(.98,threshold+.07),
           'identity_single_threshold':min(.99,threshold+.13)}
        trials.append({'options':{k:p[k] for k in ('identity_stitch_threshold','identity_attach_threshold','identity_single_threshold')},
                       'train':assess(train,p)})
    safe=[r for r in trials if r['train']['wrong']==0 and r['train']['covered']>0]
    chosen=min(safe,key=lambda r:(-r['train']['covered'],r['train']['fragment_excess'],
                                  -r['options']['identity_stitch_threshold'])) if safe else None
    if chosen:
        if progress:progress({'step':'stitch-held-out'})
        chosen={**chosen,'test':assess(test,{**options,**chosen['options']})}
    named_trials=[]
    named_chosen=None
    if db is not None:
        if progress:progress({'step':'named-prototypes'})
        examples=[]
        current_path,banks=None,{}
        candidate_options={**options,**(chosen['options'] if chosen else {})}
        identities,_,_=vi.stitch({i:{**t,'named':None} for i,t in tracks.items() if t['path'] in set(paths)},candidate_options,score_cache=score_cache)
        for identity in identities:
            path=identity['path']
            truth=[tracks[i]['named'] for i in identity['members'] if tracks[i]['named'] is not None]
            if not truth:continue
            if path!=current_path:
                banks=vi.prototype_banks(db,tracks,options,exclude_path=path);current_path=path
            scores=sorted([(vi.bank_score(identity['bank'],bank,options)[0],person)
                           for person,bank in banks.items()],reverse=True)
            if scores:
                examples.append((path,truth,scores[0][1],scores[0][0],
                                 scores[0][0]-(scores[1][0] if len(scores)>1 else -1.)))
        def name_score(selected,threshold,margin):
            accepted=[(truth,person) for path,truth,person,score,gap in examples
                      if path in selected and score>=threshold and gap>=margin]
            return {'covered':sum(len(truth) for truth,_ in accepted),
                    'wrong':sum(sum(t!=person for t in truth) for truth,person in accepted)}
        for threshold in (.65,.70,.75,.78,.80,.85,.90,.95):
            for margin in (.05,.10,.15):
                named_trials.append({'options':{'identity_named_threshold':threshold,'identity_named_margin':margin},
                                     'train':name_score(train,threshold,margin)})
        safe=[r for r in named_trials if r['train']['wrong']==0 and r['train']['covered']>0]
        if safe:
            named_chosen=min(safe,key=lambda r:(-r['train']['covered'],-r['options']['identity_named_threshold'],
                                                 -r['options']['identity_named_margin']))
            named_chosen={**named_chosen,'test':name_score(test,named_chosen['options']['identity_named_threshold'],
                                                          named_chosen['options']['identity_named_margin'])}
    return {'split':{'train':sorted(train),'test':sorted(test)},'trials':trials,'chosen':chosen,
            'named_trials':named_trials,'named_chosen':named_chosen,
            'note':'No truth labels constrain stitching. Naming prototypes exclude the entire target video; automatic names never enter banks.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['report','calibrate','rebuild'])
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('work/video-identities-report'))
    parser.add_argument('--video',action='append',default=[])
    parser.add_argument('--labels',type=Path,help='JSON object: face id -> person id; diagnostic override only')
    parser.add_argument('--max-pairs',type=int,default=10000,help='Bound CSV and debug samples; select --video for detailed inspection')
    args=parser.parse_args()
    if args.command=='rebuild':
        from prototype import database
        db=database(args.data)
        result=vi.rebuild(db,settings.read(db));print(len(result['identities']));db.close();return
    db=sqlite3.connect((args.data/'catalog.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    options={**vi.DEFAULTS,**settings.read(db)}
    tracks=vi.load_tracks(db,options)
    if args.video:tracks={i:t for i,t in tracks.items() if t['path'] in args.video}
    if args.labels:
        for fid,person in json.loads(args.labels.read_text(encoding='utf-8')).items():
            if int(fid) in tracks:tracks[int(fid)]['named']=person
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='calibrate':
        result=calibration(tracks,options,db)
    else:
        decisions=DecisionSample(max(0,args.max_pairs))
        identities,noise,cannot=vi.stitch(tracks,options,decisions)
        result={'version':vi.VERSION,'options':options,'metrics':metrics(db),
                'identities':len(identities),'unresolved':len(noise),'cannot_link_pairs':len(cannot),
                'decisions':decisions, 'decisions_total':decisions.total,
                'pair_output_limit':args.max_pairs}
        with args.output.with_suffix('.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=None
            for row in itertools.islice(pair_report(db,tracks,options),max(0,args.max_pairs)):
                if writer is None:
                    writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                writer.writerow(row)
    args.output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.output.with_suffix('.json'));db.close()


if __name__=='__main__':main()
