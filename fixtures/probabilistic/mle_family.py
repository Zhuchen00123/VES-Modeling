"""Trusted fixture candidate: gamma/beta fit + probability_ge estimator.

Reads VES_DATA_DIR/problem.json + train.csv (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a bootstrap CI on the
survival probability.  Self-reported references/parameters are never trusted
by the host.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

family = problem["family"]
threshold = float(problem["threshold"])
samples = pd.read_csv(f"{DATA_DIR}/train.csv")["value"].to_numpy(dtype=float)


def fit_survival(values):
    if family == "gamma":
        a, loc, scale = stats.gamma.fit(values)
        return float(stats.gamma.sf(threshold, a, loc=loc, scale=scale))
    a, b, loc, scale = stats.beta.fit(values)
    return float(stats.beta.sf(threshold, a, b, loc=loc, scale=scale))


estimate = fit_survival(samples)
rng = np.random.default_rng(42)
resampled = rng.choice(samples, size=(500, len(samples)), replace=True)
bootstrap = np.asarray([fit_survival(row) for row in resampled])
lo = float(np.percentile(bootstrap, 2.5))
hi = float(np.percentile(bootstrap, 97.5))

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump(
        {"estimate": estimate, "confidence_interval": [lo, hi]}, fh
    )
