"""Trusted fixture candidate: sliding-window mean-shift detector.

Reads VES_DATA_DIR/train.csv + test_features.csv (default /data) and writes
VES_OUTPUT_DIR/changepoints.json (default /output) with indices whose
left/right window mean difference exceeds a threshold.  Self-reported
metrics are never trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
y = test["y"].to_numpy(dtype=float)
n = len(y)

window = max(2, min(10, n // 10))
scores = np.full(n, 0.0)
for i in range(window, n - window):
    left = float(np.mean(y[i - window:i]))
    right = float(np.mean(y[i:i + window]))
    scores[i] = abs(left - right)

std = float(np.std(y))
threshold = 0.5 * std if std > 0.0 else 1.0
candidates = np.where(scores > threshold)[0]
if candidates.size == 0:
    candidates = np.asarray([int(np.argmax(scores))])
indices = np.unique(np.clip(candidates, 1, n - 2)).tolist()
if not indices:
    indices = [max(1, n // 2)]

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/changepoints.json", "w", encoding="utf-8") as fh:
    json.dump({"changepoints": indices}, fh)
