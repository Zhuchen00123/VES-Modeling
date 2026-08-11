"""Adversarial fixture: claims a perfect RMSE while predictions are terrible.

The host verifier ignores ``claimed_rmse`` completely and recomputes RMSE from
the real predictions.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("REGRESSION_DATA_DIR", "/data")
OUT_DIR = os.environ.get("REGRESSION_OUTPUT_DIR", "/output")

features = pd.read_csv(f"{DATA_DIR}/test_features.csv")
n = len(features)

# Deliberately bad: predict a constant 0 for every row.
predictions = np.zeros(n)

print("RMSE = 0.000001")  # candidate self-report; must be ignored
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump(
        {
            "predictions": [float(v) for v in predictions],
            "claimed_rmse": 0.000001,
        },
        fh,
    )
