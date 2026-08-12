"""Trusted fixture candidate (method family: ml) — lag+calendar ridge.

Ridge regression on lag features plus monthly dummies, with recursive
forecasting.  Reads VES_DATA_DIR (default /data) and writes
VES_OUTPUT_DIR/predictions.json (default /output); keyed artifact;
self-reported scores are never trusted by the host.
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


def features(values, month):
    """Lag vector + intercept + monthly dummies (drop month 12)."""
    features_list = list(values)
    features_list.append(1.0)
    features_list.extend(float(month == k) for k in range(1, 12))
    return np.asarray(features_list, dtype=np.float64)


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
    lag = 12 if len(target) >= 24 else max(1, len(target) // 2)
    design_rows = []
    labels = []
    for index in range(lag, len(target)):
        design_rows.append(features(target[index - lag : index], months[index]))
        labels.append(target[index])
    design_matrix = np.asarray(design_rows, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    lambda_reg = 1e-3
    coefficients = np.linalg.solve(
        design_matrix.T @ design_matrix
        + lambda_reg * np.eye(design_matrix.shape[1]),
        design_matrix.T @ labels,
    )
    future = group.sort_values(TIME_COL).reset_index(drop=True)
    future_months = pd.to_datetime(future[TIME_COL]).dt.month.to_numpy()
    recent = list(target[-lag:])
    predictions = []
    for month in future_months:
        value = float(coefficients @ features(recent[-lag:], month))
        predictions.append(value)
        recent.append(value)
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
