"""Trusted fixture candidate: RandomForestRegressor.

Same /data -> /output contract as the other fixtures.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = os.environ.get("REGRESSION_DATA_DIR", "/data")
OUT_DIR = os.environ.get("REGRESSION_OUTPUT_DIR", "/output")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
features = pd.read_csv(f"{DATA_DIR}/test_features.csv")
X = train.drop(columns=["target"]).to_numpy(dtype=np.float64)
y = train["target"].to_numpy(dtype=np.float64)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)
predictions = model.predict(features.to_numpy(dtype=np.float64))

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": [float(v) for v in np.asarray(predictions).ravel()]}, fh)
