"""Trusted fixture candidate: per-trajectory numpy linear-trend fallback.

Fits a degree-1 polynomial per trajectory and evaluates it at every test t.
Writes VES_OUTPUT_DIR/predictions.json (array mode without trajectory_id,
key mode with it).  Self-reported scores are never trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

T_COL = "t"
Y_COL = "y"
TRAJ_COL = "trajectory_id"


def key(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    return str(value)


def fit_predict(history, test_times):
    t = history[T_COL].to_numpy(dtype=float)
    y = history[Y_COL].to_numpy(dtype=float)
    if len(t) < 2:
        return np.full(len(test_times), float(y[-1]))
    trend = np.polyfit(t, y, 1)
    return np.polyval(trend, test_times)


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
has_trajectory = TRAJ_COL in test.columns

if has_trajectory:
    rows = []
    for trajectory_key, group in test.groupby(TRAJ_COL, sort=False):
        history = train[train[TRAJ_COL] == group[TRAJ_COL].iloc[0]]
        history = history.sort_values(T_COL).reset_index(drop=True)
        test_group = group.sort_values(T_COL).reset_index(drop=True)
        predictions = fit_predict(
            history, test_group[T_COL].to_numpy(dtype=float)
        )
        for index, (_, row) in enumerate(test_group.iterrows()):
            rows.append(
                {
                    TRAJ_COL: key(row[TRAJ_COL]),
                    T_COL: float(row[T_COL]),
                    "prediction": float(predictions[index]),
                }
            )
else:
    history = train.sort_values(T_COL).reset_index(drop=True)
    predictions = fit_predict(
        history, test[T_COL].to_numpy(dtype=float)
    )
    rows = [float(value) for value in predictions]

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": rows}, fh)
