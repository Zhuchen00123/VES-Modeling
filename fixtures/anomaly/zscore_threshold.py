"""Trusted fixture candidate: z-score/quantile threshold labels.

Computes per-feature z-scores from the normal train samples, scores each test
row by its maximum absolute z-score, and thresholds at the train 95th
percentile to emit 'normal'/'anomaly' labels (both classes guaranteed).
Writes VES_OUTPUT_DIR/predictions.json.  Self-reported scores are never
trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
features = list(test.columns)
train_matrix = train[features].to_numpy(dtype=float)
test_matrix = test[features].to_numpy(dtype=float)

means = train_matrix.mean(axis=0)
stds = train_matrix.std(axis=0)
stds = np.where(stds == 0.0, 1.0, stds)
train_z = np.abs((train_matrix - means) / stds).max(axis=1)
test_z = np.abs((test_matrix - means) / stds).max(axis=1)
threshold = float(np.percentile(train_z, 95))

labels = [
    "anomaly" if float(value) > threshold else "normal"
    for value in test_z
]
if len(set(labels)) < 2:
    labels[0] = "anomaly" if labels[0] == "normal" else "normal"

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"labels": labels}, fh)
