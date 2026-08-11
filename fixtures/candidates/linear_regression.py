"""Trusted fixture candidate: LinearRegression baseline.

Reads REGRESSION_DATA_DIR (default /data) and writes
REGRESSION_OUTPUT_DIR/predictions.json (default /output).  Self-reported
scores are never trusted by the host verifier.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_DIR = os.environ.get("REGRESSION_DATA_DIR", "/data")
OUT_DIR = os.environ.get("REGRESSION_OUTPUT_DIR", "/output")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
features = pd.read_csv(f"{DATA_DIR}/test_features.csv")
X = train.drop(columns=["target"]).to_numpy(dtype=np.float64)
y = train["target"].to_numpy(dtype=np.float64)

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(features.to_numpy(dtype=np.float64))

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": [float(v) for v in np.asarray(predictions).ravel()]}, fh)
