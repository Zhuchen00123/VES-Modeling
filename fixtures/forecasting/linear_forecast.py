"""Trusted fixture candidate: per-series linear-trend extrapolation forecast.

Reads VES_DATA_DIR (default /data) and writes
VES_OUTPUT_DIR/predictions.json (default /output).  Artifact rows are keyed by
(series id, timestamp); self-reported scores are never trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

SERIES_COL = "series_id"
TIME_COL = "timestamp"
TARGET_COL = "target"


def key(value):
    """Canonical string key for a series id (1, 1.0 and '1' are the same)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    return str(value)


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
rows = []
for series_id, group in test.groupby(SERIES_COL, sort=False):
    history = (
        train[train[SERIES_COL] == series_id]
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )
    time_index = np.arange(len(history))
    target = history[TARGET_COL].to_numpy(dtype=np.float64)
    coefficients = np.polyfit(time_index, target, 1)
    future = group.sort_values(TIME_COL).reset_index(drop=True)
    future_index = np.arange(
        len(history), len(history) + len(future)
    )
    predictions = np.polyval(coefficients, future_index)
    for index, (_, row) in enumerate(future.iterrows()):
        rows.append(
            {
                SERIES_COL: key(row[SERIES_COL]),
                TIME_COL: str(row[TIME_COL]),
                "prediction": float(predictions[index]),
            }
        )
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": rows}, fh)
