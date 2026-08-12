"""Trusted fixture candidate: empirical transition-frequency estimator.

Reads VES_DATA_DIR/problem.json + train.csv (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a bootstrap CI.
Self-reported references are never trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

states = problem["states"]
index = {state: i for i, state in enumerate(states)}
train = pd.read_csv(f"{DATA_DIR}/train.csv")["state"]
sequence = list(train)


def transition_estimate(values):
    counts = np.zeros((len(states), len(states)))
    for i in range(len(values) - 1):
        counts[index[values[i]], index[values[i + 1]]] += 1
    rows = counts.sum(axis=1, keepdims=True)
    rows = np.where(rows == 0.0, 1.0, rows)
    return counts / rows


estimate_matrix = transition_estimate(sequence)
if problem["quantity"] == "transition_probability":
    from_index = index[problem["from_state"]]
    to_index = index[problem["to_state"]]
    stat = lambda matrix: float(matrix[from_index, to_index])
else:
    raise ValueError("markov_frequency fixture is for transition_probability")

estimate = stat(estimate_matrix)
rng = np.random.default_rng(42)
indices = rng.integers(0, len(sequence), size=(200, len(sequence)))
bootstrap = np.asarray(
    [stat(transition_estimate([sequence[i] for i in ids])) for ids in indices]
)
lo = float(np.percentile(bootstrap, 2.5))
hi = float(np.percentile(bootstrap, 97.5))
lo = min(lo, estimate)
hi = max(hi, estimate)

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump(
        {"estimate": estimate, "confidence_interval": [lo, hi]}, fh
    )
