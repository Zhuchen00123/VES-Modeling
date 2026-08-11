"""Trusted fixture candidate: per-series naive (last train value) forecast.

Reads VES_DATA_DIR (default /data) and writes
VES_OUTPUT_DIR/predictions.json (default /output).  Artifact rows are keyed by
(series id, timestamp); self-reported scores are never trusted by the host.
"""

import json
import os

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
last = (
    train.sort_values(TIME_COL)
    .groupby(SERIES_COL, as_index=False)[TARGET_COL]
    .last()
)
mapping = {
    key(series_id): float(value)
    for series_id, value in zip(last[SERIES_COL], last[TARGET_COL])
}
rows = [
    {
        SERIES_COL: key(row[SERIES_COL]),
        TIME_COL: str(row[TIME_COL]),
        "prediction": mapping[key(row[SERIES_COL])],
    }
    for _, row in test.iterrows()
]
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": rows}, fh)
