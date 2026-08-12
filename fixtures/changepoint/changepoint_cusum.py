"""Trusted fixture candidate: CUSUM change-point detector.

Reads VES_DATA_DIR/train.csv + test_features.csv (default /data) and writes
VES_OUTPUT_DIR/changepoints.json (default /output) with the index of the
largest |cumulative sum| deviation from the series mean.  Self-reported
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

mean = float(np.mean(y))
cusum = np.cumsum(y - mean)
index = int(np.argmax(np.abs(cusum))) + 1
index = int(np.clip(index, 1, n - 2))

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/changepoints.json", "w", encoding="utf-8") as fh:
    json.dump({"changepoints": [index]}, fh)
