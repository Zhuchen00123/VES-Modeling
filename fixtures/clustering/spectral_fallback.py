"""Trusted fixture candidate: sklearn SpectralClustering fallback.

Clusters the test feature matrix directly (transductive fallback) with an
inertia-elbow k choice from train.  Writes VES_OUTPUT_DIR/predictions.json
(input mode without an id column, id mode with one).  Self-reported scores
are never trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")


def to_scalar(value):
    return value.item() if hasattr(value, "item") else value


def choose_k(features, max_k=8):
    max_k = min(max_k, max(2, features.shape[0] - 1))
    if max_k < 2:
        return 2
    inertias = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=5, random_state=42).fit(features)
        inertias.append(model.inertia_)
    best = 0
    best_drop = 0.0
    for index in range(1, len(inertias)):
        drop = (inertias[index - 1] - inertias[index]) / inertias[index - 1]
        if drop > best_drop:
            best_drop = drop
            best = index
    return best + 2


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
id_col = "id_col" if "id_col" in test.columns else None
features = [c for c in test.columns if c != id_col]
train_matrix = train[features].to_numpy(dtype=np.float64)
test_matrix = test[features].to_numpy(dtype=np.float64)

k = choose_k(train_matrix)
k = min(k, max(2, test_matrix.shape[0] - 1))
model = SpectralClustering(
    n_clusters=k, random_state=42, affinity="nearest_neighbors"
).fit(test_matrix)
labels = [f"cluster_{int(value)}" for value in model.labels_]

if id_col:
    rows = [
        {"id": to_scalar(row[1][id_col]), "label": label}
        for row, label in zip(test.iterrows(), labels)
    ]
    payload = {"predictions": rows}
else:
    payload = {"labels": labels}

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
