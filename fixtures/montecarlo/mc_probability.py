"""Trusted fixture candidate: Monte Carlo binomial probability estimator.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a Monte Carlo estimate
of the requested binomial event probability plus a normal-approximation CI.
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
n = params["n"]
p = params["p"]
event = params["event"]
threshold = params["threshold"]
n_samples = 200_000
rng = np.random.default_rng(42)
samples = rng.binomial(n, p, size=n_samples)

if event == "ge":
    values = (samples >= threshold).astype(float)
elif event == "le":
    values = (samples <= threshold).astype(float)
else:
    values = (samples == threshold).astype(float)

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
