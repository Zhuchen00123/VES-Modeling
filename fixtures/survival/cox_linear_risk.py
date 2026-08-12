"""Trusted fixture candidate: simplified Cox-style linear risk score.

Reads VES_DATA_DIR/train.csv + test_features.csv (default /data) and writes
VES_OUTPUT_DIR/predictions.json (default /output) with risk scores (higher =
higher risk).  Self-reported metrics are never trusted by the host.
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
event = train["event"].to_numpy(dtype=float)
time = train["time"].to_numpy(dtype=float)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std = np.where(std == 0.0, 1.0, std)
Xs = (X_train - mean) / std

# One-step Cox-like partial likelihood score: regress event on standardized
# features (a robust fallback for the vertical slice).
coefficients, *_ = np.linalg.lstsq(
    np.column_stack([Xs, np.ones(len(Xs))]), event, rcond=None
)
coef = coefficients[:-1]
risk = (X_test - mean) / std @ coef
risk = risk - risk.mean()

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": [float(value) for value in risk]}, fh)
