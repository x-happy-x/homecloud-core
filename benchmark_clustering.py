"""Compare clustering strategies on the prepared LFW sample."""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure


ROOT = Path(__file__).resolve().parent
db = sqlite3.connect(ROOT / "data" / "catalog.sqlite")
rows = db.execute(
    'SELECT faces.id,faces.path,faces.box,faces.embedding '
    'FROM faces JOIN photos USING(path) WHERE status="ok" ORDER BY faces.id'
).fetchall()
vectors = np.stack([np.frombuffer(row[3], dtype="<f4") for row in rows])
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

by_photo = defaultdict(list)
for index, row in enumerate(rows):
    x1, y1, x2, y2 = json.loads(row[2])
    by_photo[row[1]].append((max(0, x2 - x1) * max(0, y2 - y1), index))
primary = np.array([max(faces)[1] for faces in by_photo.values()])
truth = np.array([Path(rows[index][1]).parent.name for index in primary])


def score(name, labels):
    selected = labels[primary]
    ari = adjusted_rand_score(truth, selected)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(truth, selected)
    groups = len(set(labels) - {-1})
    noise = int(np.sum(labels == -1))
    return (ari, v_measure, name, groups, noise, homogeneity, completeness)


results = []
for distance in np.arange(0.20, 0.451, 0.01):
    labels = DBSCAN(eps=distance, min_samples=2, metric="cosine", n_jobs=-1).fit_predict(vectors)
    results.append(score(f"DBSCAN cosine={distance:.2f}", labels))

for distance in np.arange(0.20, 0.451, 0.01):
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance
    ).fit_predict(vectors)
    results.append(score(f"Agglomerative average={distance:.2f}", labels))
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="complete", distance_threshold=distance
    ).fit_predict(vectors)
    results.append(score(f"Agglomerative complete={distance:.2f}", labels))

for minimum in (2, 3, 4, 5, 8):
    labels = HDBSCAN(min_cluster_size=minimum, min_samples=2, metric="euclidean").fit_predict(vectors)
    results.append(score(f"HDBSCAN min_cluster={minimum}", labels))

print("ARI     V       groups noise homogeneity completeness algorithm")
for ari, v_measure, name, groups, noise, homogeneity, completeness in sorted(results, reverse=True)[:20]:
    print(f"{ari:.4f}  {v_measure:.4f}  {groups:>3}   {noise:>3}   {homogeneity:.4f}      {completeness:.4f}    {name}")
