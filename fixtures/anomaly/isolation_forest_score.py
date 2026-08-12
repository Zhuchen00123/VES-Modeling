"""Trusted fixture candidate: sklearn IsolationForest anomaly scores.

Fits IsolationForest on the normal train samples and writes one anomaly
score per test row (higher = more anomalous) to
VES_OUTPUT_DIR/predictions.json.  Self-reported scores are never trusted by
the host.
"""

import json
import os

import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
features = list(test.columns)

model = IsolationForest(contamination="auto", random_state=42)
model.fit(train[features].to_numpy(dtype=float))
scores = -model.decision_function(test[features].to_numpy(dtype=float))

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"scores": [float(value) for value in scores]}, fh)
