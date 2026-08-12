"""Trusted fixture candidate (method family: statistical) — linear+seasonal.

Linear trend plus monthly seasonal dummies fitted by least squares.  Reads
VES_DATA_DIR (default /data) and writes VES_OUTPUT_DIR/predictions.json
(default /output); keyed artifact; self-reported scores are never trusted.
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


def design(months, t):
    """Design matrix: intercept, time, monthly dummies (drop month 12)."""
    matrix = np.ones((len(months), 13))
    matrix[:, 1] = t
    for month in range(1, 12):
        matrix[:, month + 1] = (months == month).astype(np.float64)
    return matrix


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
rows = []
for series_id, group in test.groupby(SERIES_COL, sort=False):
    history = (
        train[train[SERIES_COL] == series_id]
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )
    target = history[TARGET_COL].to_numpy(dtype=np.float64)
    months = pd.to_datetime(history[TIME_COL]).dt.month.to_numpy()
    coefficients = np.linalg.lstsq(
        design(months, np.arange(len(target))), target, rcond=None
    )[0]
    future = group.sort_values(TIME_COL).reset_index(drop=True)
    future_months = pd.to_datetime(future[TIME_COL]).dt.month.to_numpy()
    future_index = np.arange(len(target), len(target) + len(future))
    predictions = design(future_months, future_index) @ coefficients
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
