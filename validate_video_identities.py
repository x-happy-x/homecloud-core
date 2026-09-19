"""Resume a full validation on a SQLite backup, labelled videos first.

Never writes to the source catalog. Progress, calibration and acceptance results
remain in the destination. --force re-decodes videos; it does not replace the DB.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import diagnose_video_identities as diagnostic
import settings
import video
import video_identities as vi


def write_json(path, value):
    if path.name == 'validation-progress.json':
        value = {**value, 'pid': os.getpid(), 'updated_at': time.time()}
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(5):
        try:
            os.replace(temporary,path);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.05*(attempt+1))


def run(args):
    from prototype import database
    source=args.source.resolve(); target=args.data.resolve()
    if source==target:raise ValueError('Validation needs a separate destination catalog')
    target.mkdir(parents=True,exist_ok=True)
    if not (target/'catalog.sqlite').exists():
        original=sqlite3.connect((source/'catalog.sqlite').as_uri()+'?mode=ro',uri=True)
        copy=sqlite3.connect(target/'catalog.sqlite')
        original.backup(copy);copy.close();original.close()
    db=database(target)
    if not (target/'baseline.json').exists():
        write_json(target/'baseline.json',diagnostic.metrics(db))
    baseline=json.loads((target/'baseline.json').read_text(encoding='utf-8'))
    labelled={r[0] for r in db.execute("SELECT DISTINCT f.path FROM faces f JOIN face_people p ON p.face_id=f.id WHERE f.track_start IS NOT NULL AND p.source='human'")}
    paths=[r[0] for r in db.execute("SELECT path FROM photos WHERE status NOT IN ('excluded','missing')") if video.is_video(r[0])]
    missing=[p for p in paths if not Path(p).is_file()]
    available=[p for p in paths if Path(p).is_file()]
    write_json(target/'inventory.json',{'available':available,'missing':missing})
    db.close()
    stop=target/'stop.txt'
    status=target/'validation-progress.json'
    def scan(phase, selected=None):
        roots=defaultdict(list)
        for path in (selected if selected is not None else available):
            roots[Path(path).anchor].append(path)
        for root, members in sorted(roots.items()):
            if stop.exists():raise InterruptedError('Validation stopped')
            write_json(status,{'status':'running','phase':phase,'root':root,'videos':len(members)})
            command=[sys.executable,str(Path(__file__).with_name('prototype.py')),'scan',
                     '--photos',root,'--models',str(args.models.resolve()),'--data',str(target),
                     '--use-inventory','--kinds','videos','--limit','1000000',
                     '--progress-file',str(target/'scan-progress.json'),'--stop-file',str(stop)]
            if selected is not None:
                for path in sorted(members):command.extend(['--include-path',path])
            if args.force:command.append('--force')
            with (target/(phase+'.log')).open('ab') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                               env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            if stop.exists():raise InterruptedError('Validation stopped')

    def assess(phase):
        db=database(target)
        options=settings.read(db)
        tracks=vi.load_tracks(db,options)
        calibration=diagnostic.calibration(tracks,options,db,progress=lambda detail:
            write_json(status,{'status':'running','phase':phase+'-calibration',**detail}))
        write_json(target/(phase+'-calibration.json'),calibration)
        chosen={}
        for key in ('chosen','named_chosen'):
            candidate=calibration[key]
            if candidate and candidate['test']['wrong']==0 and candidate['test']['covered']>0:
                chosen.update(candidate['options'])
        if chosen:
            options=settings.write(db,chosen)
        result=vi.rebuild(db,options,stop_check=stop.exists)
        after=diagnostic.metrics(db)
        violations=sum(result['assignments'].get(a,(-1,))[0]>=0
                       and result['assignments'].get(a,(-1,))[0]==result['assignments'].get(b,(-1,))[0]
                       for a,b in result['cannot'])
        report={'baseline':{k:v for k,v in baseline.items() if k!='truth'},'after':after,
                'cannot_link_violations':violations,'calibrated_options':chosen,
                'automatic_names':len(result['names']),'unresolved_tracks':len(result['unresolved']),
                'retained_inactive_annotations':db.execute('SELECT count(*) FROM face_track_data WHERE active=0').fetchone()[0],
                'scan_errors':sum(r[0] in set(available) for r in db.execute("SELECT path FROM photos WHERE status='error'"))}
        report['accuracy_gate']=(violations==0 and after['wrong']<=baseline['wrong']
                                 and after['error_rate']<=baseline['error_rate'])
        report['fragmentation_gate']=(after['fragment_excess'] < baseline['fragment_excess']
                                      if 'fragment_excess' in baseline else None)
        report['calibration_gate']=all(calibration[k] and calibration[k]['test']['wrong']==0
                                       and calibration[k]['test']['covered']>0 for k in ('chosen','named_chosen'))
        report['accepted']=bool(report['accuracy_gate'] and report['fragmentation_gate']
                                and report['calibration_gate'] and report['scan_errors']==0)
        write_json(target/(phase+'-results.json'),report)
        db.close()
        return report

    try:
        scan('labelled-scan',sorted(labelled & set(available)))
        write_json(status,{'status':'running','phase':'labelled-calibration'})
        assess('labelled')
        scan('full-scan')
        write_json(status,{'status':'running','phase':'full-calibration'})
        report=assess('full')
        write_json(status,{'status':'completed','phase':'finished','results':report})
    except InterruptedError:
        write_json(status,{'status':'stopped','phase':'resume-with-same-command'})
    except BaseException as exc:
        write_json(status,{'status':'error','error':str(exc)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--models',type=Path,default=Path('models/buffalo_l'))
    parser.add_argument('--force',action='store_true')
    run(parser.parse_args())


if __name__=='__main__':main()
