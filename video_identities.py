"""Track identity banks, conservative stitching and atomic catalog clustering.

One faces row remains one track. Samples are evidence, never extra votes in the
global clusterer. Automatic names never feed the human prototype bank.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import sqlite3

import numpy as np

VERSION = 2
DEFAULTS = {
    'identity_min_size': 32.0, 'identity_min_confidence': 0.8,
    'identity_max_blur': 0.76, 'identity_min_quality': 0.40,
    'identity_core_samples': 2, 'identity_sample_spacing': 0.5,
    'identity_representatives': 5, 'identity_pair_count': 3,
    'identity_stitch_threshold': 0.65, 'identity_attach_threshold': 0.72,
    'identity_single_threshold': 0.78, 'identity_margin': 0.08,
    'identity_named_threshold': 0.78, 'identity_named_margin': 0.10,
    'identity_tracking_threshold': 0.35,
    'identity_temporal_bonus': 0.02, 'identity_temporal_gap': 1.2,
    'identity_temporal_iou': 0.3,
}


def unit(value):
    if value is None:
        return None
    vector = np.frombuffer(value, dtype='<f4') if isinstance(value, bytes) else np.asarray(value)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector / norm, dtype='<f4') if norm > 1e-10 else None


def quality(box, confidence=None, blur=None, landmarks=None, options=None):
    p = {**DEFAULTS, **(options or {})}
    size = float(max(0., min(box[2] - box[0], box[3] - box[1])))
    geometry = None
    if landmarks is not None:
        points = np.asarray(landmarks, dtype=float)
        if points.shape == (5, 2) and np.isfinite(points).all():
            eyes = float(np.linalg.norm(points[1] - points[0]))
            geometry = float(np.clip(eyes / max(size * .3, 1.), 0., 1.))
            if not ((points >= np.asarray(box[:2])).all()
                    and (points <= np.asarray(box[2:])).all()):
                geometry = 0.
    reliable = (size >= p['identity_min_size'] and confidence is not None
                and confidence >= p['identity_min_confidence'] and blur is not None
                and blur < p['identity_max_blur'] and geometry is not None and geometry > 0)
    score = (min(size / 96., 1.) * (confidence or 0.)
             * max(0., 1. - (blur if blur is not None else 1.)) * (geometry or 0.)) ** .5
    return {'quality': float(score), 'reliable': bool(reliable and score >= p['identity_min_quality']),
            'size': size, 'confidence': confidence, 'blur': blur, 'geometry': geometry}


def representatives(samples, options=None):
    """Quality first, then diversity; deterministic ties and distinct moments."""
    p = {**DEFAULTS, **(options or {})}
    candidates = [s for s in samples if unit(s.get('embedding')) is not None]
    candidates.sort(key=lambda s: (-bool(s.get('reliable')), -s.get('quality', 0), s['time']))
    selected = []
    while candidates and len(selected) < p['identity_representatives']:
        if selected:
            candidates.sort(key=lambda s: (
                -bool(s.get('reliable')),
                -(s.get('quality', 0) * (.5 + .5 * (1 - max(
                    float(unit(s['embedding']) @ unit(t['embedding'])) for t in selected)))), s['time']))
        chosen = candidates.pop(0)
        selected.append(chosen)
        candidates = [s for s in candidates
                      if abs(s['time'] - chosen['time']) >= p['identity_sample_spacing']]
    return selected


def bank_score(first, second, options=None):
    """Greedy disjoint best pairs; no repeated sample inflates support."""
    p = {**DEFAULTS, **(options or {})}
    if not first or not second:
        return -1., 0
    if len(first) == len(second) == 1:
        return max(-1., min(1., float(first[0] @ second[0]))), 1
    matrix = np.clip(np.stack(first) @ np.stack(second).T, -1., 1.)
    used_a, used_b, best = set(), set(), []
    for score, a, b in sorted(((float(matrix[a, b]), a, b)
                               for a in range(len(first)) for b in range(len(second))),
                              key=lambda x: (-x[0], x[1], x[2])):
        if a in used_a or b in used_b:
            continue
        best.append(score); used_a.add(a); used_b.add(b)
        if len(best) >= p['identity_pair_count']:
            break
    return float(np.mean(best)), len(best)


def ensure_schema(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS people (
          id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS face_people (
          face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
          person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS face_exclusions (
          face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS face_track_data (
          face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
          active INTEGER NOT NULL DEFAULT 1, observations INTEGER NOT NULL,
          moments TEXT NOT NULL, first_box TEXT, last_box TEXT,
          quality REAL, version INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS face_track_samples (
          face_id INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
          ordinal INTEGER NOT NULL, frame_time REAL NOT NULL, embedding BLOB NOT NULL,
          box TEXT NOT NULL, quality REAL, reliable INTEGER NOT NULL,
          metadata TEXT NOT NULL, PRIMARY KEY(face_id,ordinal));
        CREATE TABLE IF NOT EXISTS video_identities (
          id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL,
          fingerprint TEXT NOT NULL, version INTEGER NOT NULL, metadata TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS face_track_identities (
          face_id INTEGER PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
          identity_id INTEGER REFERENCES video_identities(id),
          status TEXT NOT NULL, score REAL, reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS identity_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    columns = {r[1] for r in db.execute('PRAGMA table_info(face_people)')}
    for name, declaration in [('source', "TEXT NOT NULL DEFAULT 'human'"),
                              ('score', 'REAL'), ('version', 'INTEGER')]:
        if name not in columns:
            db.execute(f'ALTER TABLE face_people ADD COLUMN {name} {declaration}')
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='face_clusters'").fetchone():
        db.execute("INSERT OR IGNORE INTO identity_state(key,value) SELECT 'highest_label',COALESCE(MAX(label),-1) FROM face_clusters")
    db.commit()


def save_track(db, face_id, track):
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='face_quality'").fetchone():
        db.execute('UPDATE face_quality SET version=0 WHERE face_id=?', (face_id,))
    db.execute('INSERT OR REPLACE INTO face_track_data VALUES(?,?,?,?,?,?,?,?)',
               (face_id, 1, track['observations'], json.dumps(track['moments']),
                json.dumps(track['first_box']), json.dumps(track['last_box']),
                max((s['quality'] for s in track['representatives']), default=0), VERSION))
    db.execute('DELETE FROM face_track_samples WHERE face_id=?', (face_id,))
    for ordinal, sample in enumerate(track['representatives']):
        meta = {k: v for k, v in sample.items() if k not in ('embedding', 'box', 'time')}
        db.execute('INSERT INTO face_track_samples VALUES(?,?,?,?,?,?,?,?)',
                   (face_id, ordinal, sample['time'], unit(sample['embedding']).tobytes(),
                    json.dumps(sample['box']), sample['quality'], int(sample['reliable']),
                    json.dumps(meta)))


def load_tracks(db, options=None):
    p = {**DEFAULTS, **(options or {})}
    tracks = {}
    excluded = {r[0] for r in db.execute('SELECT face_id FROM face_exclusions')}
    named = dict(db.execute("SELECT face_id,person_id FROM face_people WHERE source='human'"))
    for fid, path, start, stop, blob, moments, active, observations, first, last in db.execute('''
        SELECT f.id,f.path,f.track_start,f.track_stop,f.embedding,t.moments,
               COALESCE(t.active,1),COALESCE(t.observations,0),t.first_box,t.last_box
        FROM faces f LEFT JOIN face_track_data t ON t.face_id=f.id
        JOIN photos p ON p.path=f.path
        WHERE (f.track_start IS NOT NULL OR f.frame_time IS NOT NULL)
          AND p.status='ok' '''):
        if fid in excluded or not active:
            continue
        vector = unit(blob)
        tracks[fid] = {'id': fid, 'path': path, 'start': start, 'stop': stop,
                       'moments': set(json.loads(moments)) if moments else set(),
                       'observations': observations, 'samples': [], 'bank': [],
                       'fallback': [] if vector is None else [vector], 'named': named.get(fid),
                       'first_box': json.loads(first) if first else None,
                       'last_box': json.loads(last) if last else None}
    for fid, moment, blob, box, q, reliable, raw_meta in db.execute(
            'SELECT face_id,frame_time,embedding,box,quality,reliable,metadata '
            'FROM face_track_samples ORDER BY face_id,ordinal'):
        if fid not in tracks or unit(blob) is None:
            continue
        meta = json.loads(raw_meta)
        # Re-evaluate eligibility when thresholds change without decoding video.
        reliable = (meta.get('size', 0) >= p['identity_min_size']
                    and (meta.get('confidence') or 0) >= p['identity_min_confidence']
                    and meta.get('blur') is not None and meta['blur'] < p['identity_max_blur']
                    and (meta.get('geometry') or 0) > 0 and q >= p['identity_min_quality'])
        tracks[fid]['samples'].append({'time': moment, 'embedding': unit(blob),
                                       'quality': q, 'reliable': reliable, 'box': json.loads(box)})
    for t in tracks.values():
        t['samples'] = representatives(t['samples'], p)
        t['bank'] = [s['embedding'] for s in t['samples'] if s['reliable']]
        t['match_bank'] = t['bank'] or [s['embedding'] for s in t['samples']] or t['fallback']
        t['core'] = len(t['bank']) >= p['identity_core_samples']
        t['quality'] = max((s['quality'] for s in t['samples']), default=0.)
    return tracks


def conflicts(tracks):
    moments, pairs = defaultdict(list), set()
    for fid, track in tracks.items():
        for moment in track['moments']:
            moments[(track['path'], moment)].append(fid)
    for members in moments.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def incompatible(a, b, tracks, cannot):
    names = {tracks[i]['named'] for i in a | b if tracks[i]['named'] is not None}
    return len(names) > 1 or any(tuple(sorted((i, j))) in cannot for i in a for j in b)


def stitch(tracks, options=None, debug=None, stop_check=None, score_cache=None):
    p = {**DEFAULTS, **(options or {})}
    def compare(a, b):
        key = (*sorted((a, b)), p['identity_pair_count'])
        if score_cache is not None and key in score_cache:
            return score_cache[key]
        result = bank_score(tracks[a]['match_bank'], tracks[b]['match_bank'], p)
        if (score_cache is not None and tracks[a]['core'] and tracks[b]['core']
                and len(score_cache) < 1_000_000):
            score_cache[key] = result
        return result
    cannot = conflicts(tracks)
    by_path = defaultdict(list)
    for fid, t in tracks.items():
        by_path[t['path']].append(fid)
    identities, unresolved = [], set(tracks)
    for path, ids in sorted(by_path.items()):
        if stop_check and stop_check():
            raise InterruptedError('Stitching stopped before publication')
        cores = [i for i in sorted(ids) if tracks[i]['core']]
        groups = {i: {i} for i in cores}
        owners = {i: i for i in cores}
        blocked = {i: set() for i in cores}
        scores, edges = {}, []
        for index, a in enumerate(cores):
            if stop_check and stop_check():
                raise InterruptedError('Stitching stopped before publication')
            for b in cores[index + 1:]:
                score, support = compare(a, b)
                scores[(a, b)] = score
                temporal = False
                x, y = sorted((tracks[a], tracks[b]), key=lambda t: t['start'] or 0)
                if x['stop'] is not None and y['start'] is not None:
                    if 0 <= y['start'] - x['stop'] <= p['identity_temporal_gap']:
                        from video_tracks import iou
                        temporal = bool(x['last_box'] and y['first_box']
                                        and iou(x['last_box'], y['first_box']) >= p['identity_temporal_iou'])
                # Bonus only orders already admissible edges; never lowers ArcFace floor.
                edges.append((score + p['identity_temporal_bonus'] * temporal, score, a, b))
        for _, score, a, b in sorted(edges, key=lambda e: (-e[0], e[2], e[3])):
            ga, gb = owners[a], owners[b]
            if ga == gb:
                continue
            if gb in blocked[ga]:
                continue
            if score < p['identity_stitch_threshold']:
                # This original pair is already a witness that complete support
                # fails, regardless of the other members of either group.
                blocked[ga].add(gb)
                blocked[gb].add(ga)
                if debug is not None:
                    debug.append({'track_a':a,'track_b':b,'arcface_similarity':score,
                                  'decision':'reject','reason':'below-stitch-threshold'})
                continue
            left, right = groups[ga], groups[gb]
            conflict = incompatible(left, right, tracks, cannot)
            complete_score = min(scores[tuple(sorted((i, j)))] for i in left for j in right)
            merge = not conflict and complete_score >= p['identity_stitch_threshold']
            if debug is not None:
                debug.append({'track_a': a, 'track_b': b, 'arcface_similarity': score,
                              'distance': math.sqrt(max(0., 2 - 2 * score)),
                              'quality_a': tracks[a]['quality'], 'quality_b': tracks[b]['quality'],
                              'temporal_overlap_conflict': tuple(sorted((a,b))) in cannot,
                              'constraint_conflict': conflict, 'complete_score': complete_score,
                              'decision': 'merge' if merge else 'reject'})
            if merge:
                # Rejection is monotone: growing either group cannot remove
                # a cannot-link or improve its minimum cross-track score.
                neighbors = (blocked[ga] | blocked[gb]) - {ga, gb}
                for neighbor in neighbors:
                    blocked[neighbor].discard(gb)
                    blocked[neighbor].add(ga)
                blocked[ga] = neighbors
                del blocked[gb]
                groups[ga] |= groups.pop(gb)
                for i in groups[ga]:
                    owners[i] = ga
            else:
                blocked[ga].add(gb)
                blocked[gb].add(ga)
        local = [{'path': path, 'core': set(g), 'members': set(g), 'attachment': {}}
                 for _, g in sorted(groups.items())]
        # Frozen core banks: attachments cannot strengthen the next attachment.
        for fid in sorted(set(ids) - set(cores)):
            candidates = []
            for index, identity in enumerate(local):
                if incompatible({fid}, identity['members'], tracks, cannot):
                    continue
                matches = []
                floor = (p['identity_single_threshold'] if len(tracks[fid]['match_bank']) < 2
                         else p['identity_attach_threshold']) - p['identity_margin']
                for i in sorted(identity['core']):
                    match = compare(fid, i)
                    if match[0] < floor:
                        break  # Cannot win or make an above-threshold winner ambiguous.
                    matches.append(match)
                else:
                    score, support = min(matches)
                    candidates.append((score, index, support))
            candidates.sort(key=lambda x: (-x[0], x[1]))
            attached = False
            if candidates:
                score, index, support = candidates[0]
                second = candidates[1][0] if len(candidates) > 1 else -1.
                threshold = p['identity_single_threshold'] if support < 2 else p['identity_attach_threshold']
                if score >= threshold and score - second >= p['identity_margin']:
                    local[index]['members'].add(fid)
                    local[index]['attachment'][fid] = score
                    attached = True
            if debug is not None:
                debug.append({'track_a': fid, 'track_b': min(local[candidates[0][1]]['core']) if candidates else None,
                              'arcface_similarity': candidates[0][0] if candidates else None,
                              'quality_a': tracks[fid]['quality'],
                              'decision': 'attach' if attached else 'unresolved',
                              'reason': 'confident-attachment' if attached else 'insufficient-or-ambiguous-evidence'})
        for identity in local:
            samples = [s for i in sorted(identity['core']) for s in tracks[i]['samples'] if s['reliable']]
            # Distinct tracks may reuse timestamps; diversity here does not use time filtering.
            identity['bank'] = diverse_bank([(s['quality'], s['embedding']) for s in samples], p)
            identity['vector'] = unit(np.mean(identity['bank'], axis=0))
            identities.append(identity)
            unresolved -= identity['members']
    return identities, unresolved, cannot


def diverse_bank(samples, options=None):
    p = {**DEFAULTS, **(options or {})}
    pool = sorted(samples, key=lambda x: -x[0])
    bank = []
    while pool and len(bank) < p['identity_representatives']:
        best = max(range(len(pool)), key=lambda i: pool[i][0] * (
            1 if not bank else .5 + .5 * (1 - max(float(pool[i][1] @ v) for v in bank))))
        _, vector = pool.pop(best)
        if not bank or max(float(vector @ v) for v in bank) < .9999:
            bank.append(vector)
    return bank


def prototype_banks(db, tracks, options=None, exclude_path=None):
    samples = defaultdict(list)
    has_quality = db.execute("SELECT 1 FROM sqlite_master WHERE name='face_quality'").fetchone()
    quality_rows = dict((r[0], r[1:]) for r in db.execute('SELECT face_id,blur,size FROM face_quality')) if has_quality else {}
    for fid, person, path, blob, start, moment in db.execute('''
        SELECT f.id,fp.person_id,f.path,f.embedding,f.track_start,f.frame_time FROM faces f
        JOIN face_people fp ON fp.face_id=f.id AND fp.source='human'
        JOIN photos p ON p.path=f.path
        LEFT JOIN face_track_data t ON t.face_id=f.id
        LEFT JOIN face_exclusions e ON e.face_id=f.id
        WHERE e.face_id IS NULL AND p.status='ok' AND COALESCE(t.active,1)=1'''):
        if path == exclude_path:
            continue
        if start is not None or moment is not None:
            if fid in tracks:
                samples[person].extend((s['quality'], s['embedding']) for s in tracks[fid]['samples'] if s['reliable'])
        else:
            blur, size = quality_rows.get(fid, (None, None))
            p = {**DEFAULTS, **(options or {})}
            vector = unit(blob)
            if (vector is not None and blur is not None and size is not None
                    and blur < p['identity_max_blur'] and size >= p['identity_min_size']):
                samples[person].append((1-blur, vector))
    return {person: diverse_bank(items, options) for person, items in samples.items()}


def stable_labels(groups, old, highest):
    """One-to-one maximum-overlap greedy mapping, largest surviving parts first."""
    edges = []
    for index, members in enumerate(groups):
        counts = Counter(old[i] for i in members if old.get(i, -1) >= 0)
        edges.extend((count, label, index) for label, count in counts.items())
    result, used = {}, set()
    for count, label, index in sorted(edges, key=lambda x: (-x[0], x[1], x[2])):
        if label not in used and index not in result:
            result[index] = label; used.add(label)
    for index in range(len(groups)):
        if index not in result:
            highest += 1; result[index] = highest
    return result, highest


def build(db, options=None, min_cluster_size=8, stop_check=None, debug=None, scope='all'):
    """Pure read/compute phase. Caller publishes only if data_version still matches."""
    from prototype import cluster_embeddings
    p = {**DEFAULTS, **(options or {})}
    def check():
        if stop_check and stop_check():
            raise InterruptedError('Identity rebuild stopped before publication')
    check()
    old = dict(db.execute('SELECT face_id,label FROM face_clusters'))
    all_tracks = load_tracks(db, p)
    eligible = {r[0] for r in db.execute('SELECT id FROM faces')
                if scope == 'all' or old.get(r[0], -1) < 0}
    if scope == 'noise':
        eligible -= {r[0] for r in db.execute('SELECT face_id FROM face_people')}
        # Never split a published video identity just to process its leftovers.
        previous = defaultdict(set)
        for fid,ident in db.execute('SELECT face_id,identity_id FROM face_track_identities WHERE identity_id IS NOT NULL'):
            previous[ident].add(fid)
        for members in previous.values():
            if not members <= eligible:
                eligible -= members
    tracks = {i:t for i,t in all_tracks.items() if i in eligible}
    identities, unresolved, cannot = stitch(tracks, p, debug, stop_check)
    cannot = conflicts(all_tracks)
    check()
    named = dict(db.execute("SELECT face_id,person_id FROM face_people WHERE source='human'"))
    occupied = {fid:person for fid,person in db.execute('SELECT face_id,person_id FROM face_people') if fid not in eligible}
    auto_names = {}
    # Leave-one-video-out prototypes prevent self-confirmation of a video.
    current_path, banks = None, {}
    for ident in identities:
        check()
        path = ident['path']
        if path != current_path:
            banks = prototype_banks(db, all_tracks, p, exclude_path=path)
            current_path = path
        manual = {named[i] for i in ident['members'] if i in named}
        scores = sorted([(bank_score(ident['bank'], bank, p)[0], person)
                         for person, bank in banks.items()], key=lambda x: (-x[0], x[1]))
        person, confidence = None, 0.
        if len(manual) == 1:
            person, confidence = next(iter(manual)), 1.
        elif scores:
            confidence, candidate = scores[0]
            runner = scores[1][0] if len(scores) > 1 else -1.
            if confidence >= p['identity_named_threshold'] and confidence-runner >= p['identity_named_margin']:
                person = candidate
        if person is not None:
            others = {i for i, name in {**occupied, **named, **{k:v[0] for k,v in auto_names.items()}}.items() if name == person}
            if any(tuple(sorted((i,j))) in cannot for i in ident['members'] for j in others-ident['members']):
                person = None
        ident['person'] = person
        if person is not None:
            for fid in ident['members'] - named.keys():
                auto_names[fid] = (person, confidence)
    check()
    units = [{'members': x['members'], 'vector': x['vector'], 'identity': x,
              'person': x['person']} for x in identities]
    quality_rows = {}
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='face_quality'").fetchone():
        quality_rows = {r[0]: r[1:] for r in db.execute('SELECT face_id,blur,size FROM face_quality')}
    for fid, blob in db.execute('''SELECT f.id,f.embedding FROM faces f
        LEFT JOIN face_exclusions e ON e.face_id=f.id JOIN photos p ON p.path=f.path
        LEFT JOIN face_track_data t ON t.face_id=f.id
        WHERE f.track_start IS NULL AND f.frame_time IS NULL AND e.face_id IS NULL
          AND p.status='ok' AND COALESCE(t.active,1)=1'''):
        if fid not in eligible:
            continue
        blur, size = quality_rows.get(fid, (None, None))
        vector = unit(blob)
        if vector is None or (blur is not None and blur >= p.get('face_blur_threshold', .76)) or (size is not None and size < p.get('face_min_size',20)):
            continue
        units.append({'members': {fid}, 'vector': vector, 'identity': None, 'person': named.get(fid)})
    labels = [-1] * len(units)
    if len(units) >= 2:
        # The photo-only minimum is applied after constraints. Two video units
        # must be allowed to meet even when the old detection minimum was eight.
        labels, _ = cluster_embeddings(np.stack([u['vector'] for u in units]), min_cluster_size=2)
    check()
    coarse = defaultdict(list)
    for index, label in enumerate(labels):
        coarse[int(label) if label >= 0 else ('noise', index)].append(index)
    groups = []
    for indices in coarse.values():
        check()
        partitions = []
        for index in indices:
            candidate = units[index]
            for partition in partitions:
                valid = True
                for other_index in partition:
                    other = units[other_index]
                    if candidate['person'] is not None and other['person'] is not None and candidate['person'] != other['person']:
                        valid = False; break
                    if any(tuple(sorted((a,b))) in cannot for a in candidate['members'] for b in other['members']):
                        valid = False; break
                    if candidate['identity'] or other['identity']:
                        if (candidate['identity'] and other['identity']
                                and candidate['identity']['path'] == other['identity']['path']):
                            # Global clustering must not undo complete support
                            # checks performed before truncating identity banks.
                            if any(bank_score(all_tracks[a]['bank'], all_tracks[b]['bank'], p)[0]
                                   < p['identity_stitch_threshold']
                                   for a in candidate['identity']['core'] for b in other['identity']['core']):
                                valid = False; break
                        a = candidate['identity']['bank'] if candidate['identity'] else [candidate['vector']]
                        b = other['identity']['bank'] if other['identity'] else [other['vector']]
                        score, support = bank_score(a,b,p)
                        if score < (p['identity_stitch_threshold'] if support >= 2 else p['identity_single_threshold']):
                            valid = False; break
                if valid:
                    partition.append(index); break
            else:
                partitions.append([index])
        for partition in partitions:
            if any(units[i]['identity'] for i in partition) or len(partition) >= min_cluster_size:
                groups.append(set().union(*(units[i]['members'] for i in partition)))
    old = dict(db.execute('SELECT face_id,label FROM face_clusters'))
    state = dict(db.execute('SELECT key,value FROM identity_state'))
    mapping, highest = stable_labels(groups, old, max(int(state.get('highest_label', '-1')), max(old.values(), default=-1)))
    assignments = {fid: (-1, 0., 'identity-unresolved') for fid in eligible}
    for index, members in enumerate(groups):
        for fid in members:
            assignments[fid] = (mapping[index], 1., 'track-identities-v2')
    fingerprint = hashlib.sha256(json.dumps({'version':VERSION,'options':p,
        'tracks':[(i,sorted(t['moments']),[s['embedding'].tobytes().hex() for s in t['samples']])
                  for i,t in sorted(tracks.items())]},sort_keys=True).encode()).hexdigest()
    return {'identities': identities, 'unresolved': unresolved, 'assignments': assignments,
            'names': auto_names, 'highest': highest, 'fingerprint': fingerprint,
            'tracks': tracks, 'cannot': cannot, 'scope': scope}


def publish(db, result, data_version):
    now = datetime.now(timezone.utc).isoformat()
    db.execute('BEGIN IMMEDIATE')
    try:
        if db.execute('PRAGMA data_version').fetchone()[0] != data_version:
            raise RuntimeError('Catalog changed during identity computation; retry rebuild')
        old_ids = dict(db.execute('SELECT face_id,identity_id FROM face_track_identities WHERE identity_id IS NOT NULL'))
        old_groups = defaultdict(set)
        for fid, ident in old_ids.items(): old_groups[ident].add(fid)
        sequence = db.execute("SELECT seq FROM sqlite_sequence WHERE name='video_identities'").fetchone()
        id_map, _ = stable_labels([x['members'] for x in result['identities']], old_ids,
                                 max(max(old_groups, default=0), sequence[0] if sequence else 0))
        db.execute('CREATE TEMP TABLE IF NOT EXISTS identity_publish_ids(id INTEGER PRIMARY KEY)')
        db.execute('DELETE FROM identity_publish_ids')
        db.executemany('INSERT INTO identity_publish_ids VALUES(?)', [(i,) for i in result['assignments']])
        db.execute('DELETE FROM face_track_identities WHERE face_id IN (SELECT id FROM identity_publish_ids)')
        for index, ident in enumerate(result['identities']):
            ident_id = id_map[index]
            db.execute('INSERT OR REPLACE INTO video_identities VALUES(?,?,?,?,?)',
                       (ident_id, ident['path'], result['fingerprint'], VERSION,
                        json.dumps({'core':sorted(ident['core']), 'person':ident['person']})))
            for fid in ident['members']:
                core = fid in ident['core']
                db.execute('INSERT INTO face_track_identities VALUES(?,?,?,?,?)',
                           (fid, ident_id, 'core' if core else 'attached',
                            1. if core else ident['attachment'][fid], 'quality-core' if core else 'arcface-attachment'))
        db.executemany('INSERT INTO face_track_identities VALUES(?,NULL,?,0,?)',
                       [(i,'unresolved','insufficient-or-ambiguous-evidence') for i in result['unresolved']])
        db.execute('DELETE FROM video_identities WHERE id NOT IN (SELECT identity_id FROM face_track_identities WHERE identity_id IS NOT NULL)')
        db.execute("DELETE FROM face_people WHERE source='automatic' AND face_id IN (SELECT id FROM identity_publish_ids)")
        db.executemany("INSERT INTO face_people(face_id,person_id,source,score,version) VALUES(?,?,'automatic',?,?) ON CONFLICT(face_id) DO NOTHING",
                       [(fid,person,score,VERSION) for fid,(person,score) in result['names'].items()])
        db.execute('DELETE FROM face_clusters WHERE face_id IN (SELECT id FROM identity_publish_ids)')
        db.executemany('INSERT INTO face_clusters VALUES(?,?,?,?,?)',
                       [(fid,label,score,method,now) for fid,(label,score,method) in result['assignments'].items()])
        db.execute("INSERT OR REPLACE INTO identity_state VALUES('highest_label',?)", (str(result['highest']),))
        if result['scope'] == 'all':
            db.execute("INSERT OR REPLACE INTO identity_state VALUES('dirty','0')")
        db.commit()
    except BaseException:
        db.rollback(); raise


def rebuild(db, options=None, min_cluster_size=8, stop_check=None, scope='all'):
    ensure_schema(db)
    version = db.execute('PRAGMA data_version').fetchone()[0]
    result = build(db, options, min_cluster_size, stop_check, scope=scope)
    if stop_check and stop_check():
        raise InterruptedError('Identity rebuild stopped before publication')
    publish(db, result, version)
    return result


def mark_dirty(db):
    db.execute("INSERT OR REPLACE INTO identity_state VALUES('dirty','1')")


def needs_rebuild(db):
    dirty = db.execute("SELECT value FROM identity_state WHERE key='dirty'").fetchone()
    return bool(dirty and dirty[0] == '1')


def hint_status(db, path, count):
    observed = db.execute('''SELECT count(DISTINCT COALESCE('person:' || p.person_id,
        'identity:' || t.identity_id)) FROM faces f
        JOIN face_track_identities t ON t.face_id=f.id AND t.identity_id IS NOT NULL
        LEFT JOIN face_people p ON p.face_id=f.id WHERE f.path=?''', (path,)).fetchone()[0]
    return {'observed': observed, 'conflict': bool(count and observed > count),
            'reason': 'similarity-or-cannot-link-prevents-merge' if count and observed > count else None}
