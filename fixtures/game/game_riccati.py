"""Trusted fixture candidate: game-Riccati minimax controller.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with the minimax control
sequence from the game Riccati recursion.  Self-reported metrics are never
trusted by the host.
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
C = np.asarray(problem["C"], dtype=float)
Q = np.asarray(problem["Q"], dtype=float)
R = np.asarray(problem["R"], dtype=float)
S = np.asarray(problem["S"], dtype=float)
QN = np.asarray(problem.get("terminal_weight", Q), dtype=float)
x0 = np.asarray(problem["x0"], dtype=float)
N = int(problem["horizon"])

m = int(B.shape[1])
p = int(C.shape[1])
H = np.concatenate([B, C], axis=1)
D = np.zeros((m + p, m + p))
D[:m, :m] = R
D[m:, m:] = -S

P = QN
controller_gains = []
disturbance_gains = []
for _ in range(N):
    feedback = -np.linalg.solve(D + H.T @ P @ H, H.T @ P @ A)
    controller_gains.append(feedback[:m, :])
    disturbance_gains.append(feedback[m:, :])
    P = Q + A.T @ P @ A - A.T @ P @ H @ np.linalg.solve(
        D + H.T @ P @ H, H.T @ P @ A
    )
controller_gains.reverse()
disturbance_gains.reverse()

x = x0
control = []
for K, W in zip(controller_gains, disturbance_gains):
    u = K @ x
    w = W @ x
    control.append([float(value) for value in u])
    x = A @ x + B @ u + C @ w

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"control": control}, fh)
