"""Evaluate preliminary clusters using LFW folder labels (not used for clustering)."""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import numpy as np
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

from prototype import cluster_embeddings

root = Path(__file__).resolve().parent
db = sqlite3.connect(root / 'data' / 'catalog.sqlite')
rows = db.execute('SELECT faces.path,box,embedding FROM faces JOIN photos USING(path) WHERE status="ok" ORDER BY faces.id').fetchall()
if not rows:
    raise SystemExit('No faces')
vectors = np.stack([np.frombuffer(row[2], dtype='<f4') for row in rows])
labels, probabilities = cluster_embeddings(vectors, algorithm='hdbscan', min_cluster_size=3)

# Each LFW file is labelled for its main subject, but may contain bystanders. Use the
# largest detected face per photo for evaluation while still clustering every face.
by_photo = defaultdict(list)
for index, row in enumerate(rows):
    x1, y1, x2, y2 = json.loads(row[1])
    by_photo[row[0]].append((max(0, x2 - x1) * max(0, y2 - y1), index))
primary = np.array([max(faces)[1] for faces in by_photo.values()])
truth = np.array([Path(rows[index][0]).parent.name for index in primary])
predicted = labels[primary]
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(truth, predicted)
report = {
    'photos_ok': db.execute('SELECT COUNT(*) FROM photos WHERE status="ok"').fetchone()[0],
    'photo_errors': db.execute('SELECT COUNT(*) FROM photos WHERE status="error"').fetchone()[0],
    'faces': len(rows),
    'groups': len(set(labels) - {-1}),
    'ungrouped_faces': int(sum(labels == -1)),
    'primary_faces_evaluated': len(primary),
    'ungrouped_primary_faces': int(sum(predicted == -1)),
    'mixed_primary_groups': sum(len(set(truth[predicted == label])) > 1 for label in set(predicted) - {-1}),
    'adjusted_rand_index': adjusted_rand_score(truth, predicted),
    'homogeneity': homogeneity,
    'completeness': completeness,
    'v_measure': v_measure,
    'mean_membership_confidence': float(np.mean(probabilities[labels != -1])),
    'algorithm': 'HDBSCAN(min_cluster_size=3,min_samples=2,euclidean_on_normalized_vectors)',
    'notes': 'LFW labels describe the main subject. Evaluation uses the largest detected face per photo; all faces still participate in clustering. This curated benchmark is not a family-photo accuracy estimate.'
}
(root / 'test-results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
