"""Trusted fixture candidate: Monte Carlo expectation estimator.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a Monte Carlo estimate
of the requested expectation target plus a normal-approximation CI.
Self-reported reference values are never trusted by the host.
"""

import json
import os

import numpy as np

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

params = problem["params"]
outcomes = np.asarray(params["outcomes"], dtype=float)
probabilities = np.asarray(params["probabilities"], dtype=float)
target = params["target"]
threshold = params.get("threshold")
n_samples = 200_000
rng = np.random.default_rng(42)
samples = rng.choice(outcomes, size=n_samples, p=probabilities)

if target == "mean":
    values = samples
elif target in ("second_moment", "variance"):
    values = samples**2
elif target == "prob_ge":
    values = (samples >= threshold).astype(float)
else:
    values = (samples <= threshold).astype(float)

if target == "variance":
    estimate = float(np.mean(values) - np.mean(samples) ** 2)
else:
    estimate = float(np.mean(values))
std = float(np.std(values, ddof=1))
half_width = 1.96 * std / np.sqrt(n_samples)
confidence_interval = [estimate - half_width, estimate + half_width]

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump(
        {"estimate": estimate, "confidence_interval": confidence_interval},
        fh,
    )
