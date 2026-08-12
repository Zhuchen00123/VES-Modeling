"""Trusted fixture candidate: exact elementary-CA iteration (numpy bool).

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with the exact CA quantity.
Self-reported references are never trusted by the host.
"""

import json
import os

import numpy as np

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

rule = int(problem["rule"])
width = int(problem["width"])
steps = int(problem["steps"])
initial = np.asarray(problem["initial"], dtype=bool)
quantity = problem["quantity"]
index = problem.get("index")

lookup = np.asarray(
    [(rule >> bit) & 1 for bit in range(8)], dtype=np.uint8
)


def step(state: np.ndarray) -> np.ndarray:
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    triplets = (left.astype(np.uint8) << 2) | (
        state.astype(np.uint8) << 1
    ) | right.astype(np.uint8)
    return lookup[triplets].astype(bool)


state = initial.copy()
persistent = state.copy()
for _ in range(steps):
    state = step(state)
    persistent = persistent & state

if quantity == "final_density":
    estimate = float(np.count_nonzero(state)) / width
elif quantity == "cell_state":
    estimate = float(state[index])
else:
    estimate = float(np.count_nonzero(persistent))

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump(
        {
            "estimate": estimate,
            "confidence_interval": [estimate, estimate],
        },
        fh,
    )
