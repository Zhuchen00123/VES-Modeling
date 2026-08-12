"""Trusted fixture candidate: static proportional feedback fallback.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a control sequence from
the static feedback u = -R^-1 B' x.  Self-reported metrics are never trusted
by the host.
"""

import json
import os

import numpy as np

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

A = np.asarray(problem["A"], dtype=float)
B = np.asarray(problem["B"], dtype=float)
R = np.asarray(problem["R"], dtype=float)
x0 = np.asarray(problem["x0"], dtype=float)
N = int(problem["horizon"])

K = np.linalg.solve(R, B.T)

x = x0
control = []
for _ in range(N):
    u = -K @ x
    control.append([float(value) for value in u])
    x = A @ x + B @ u

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"control": control}, fh)
