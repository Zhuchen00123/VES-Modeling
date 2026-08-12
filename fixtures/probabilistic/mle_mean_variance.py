"""Trusted fixture candidate: MLE mean/variance estimator.

Reads VES_DATA_DIR/problem.json + train.csv (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a bootstrap CI.
Self-reported references/parameters are never trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

quantity = problem["quantity"]
samples = pd.read_csv(f"{DATA_DIR}/train.csv")["value"].to_numpy(dtype=float)

if quantity == "mean":
    estimate = float(np.mean(samples))
    stat = lambda values: float(np.mean(values))
else:
    estimate = float(np.var(samples, ddof=1))
    stat = lambda values: float(np.var(values, ddof=1))

rng = np.random.default_rng(42)
resampled = rng.choice(samples, size=(1000, len(samples)), replace=True)
bootstrap = np.asarray([stat(row) for row in resampled])
lo = float(np.percentile(bootstrap, 2.5))
hi = float(np.percentile(bootstrap, 97.5))

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump(
        {"estimate": estimate, "confidence_interval": [lo, hi]}, fh
    )
