"""Trusted fixture candidate: Kaplan-Meier median-time fallback.

Reads VES_DATA_DIR/train.csv + test_features.csv (default /data) and writes
VES_OUTPUT_DIR/predictions.json (default /output) with predicted times
(baseline KM median adjusted by a feature-linear factor).  Self-reported
metrics are never trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
features = [c for c in test.columns]
X_train = train[features].to_numpy(dtype=float)
X_test = test[features].to_numpy(dtype=float)
time = train["time"].to_numpy(dtype=float)
event = train["event"].to_numpy(dtype=float)

uncensored = time[event == 1]
baseline = float(np.median(uncensored)) if uncensored.size else float(np.median(time))
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std = np.where(std == 0.0, 1.0, std)
score = ((X_test - mean) / std).mean(axis=1)
predicted = baseline * np.exp(-score)

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": [float(value) for value in predicted]}, fh)
