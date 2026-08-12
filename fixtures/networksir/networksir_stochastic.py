"""Trusted fixture candidate: discrete-time network SIR, replications.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with the network-SIR quantity
averaged over replications plus a percentile confidence interval.
Self-reported references are never trusted by the host.
"""

import json
import os

import numpy as np

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

beta = float(problem["beta"])
gamma = float(problem["gamma"])
n = int(problem["n_nodes"])
i0 = int(problem["i0"])
t_end = float(problem["t_end"])
quantity = problem["quantity"]
target_t = float(problem.get("t", t_end))

adjacency: list[list[int]] = [[] for _ in range(n)]
for u, v in problem["edges"]:
    adjacency[u].append(v)
    adjacency[v].append(u)

steps = max(1, round(t_end))
target_step = max(1, round(target_t))
rng = np.random.default_rng(42)
replications = 800
values = []

for _ in range(replications):
    state = np.zeros(n, dtype=np.int8)
    state[:i0] = 1
    cumulative = state == 1
    peak = i0
    infected_at_step = None
    for step in range(1, steps + 1):
        for node in np.where(state == 1)[0]:
            for neighbor in adjacency[node]:
                if state[neighbor] == 0 and rng.random() < beta:
                    state[neighbor] = 1
                    cumulative[neighbor] = True
        infected_mask = state == 1
        recover = rng.random(n) < gamma
        state[recover & infected_mask] = 2
        infected_count = int(np.count_nonzero(state == 1))
        peak = max(peak, infected_count)
        if step == target_step:
            infected_at_step = infected_count
    if quantity == "final_size":
        value = float(np.count_nonzero(cumulative)) / n
    elif quantity == "peak_infected":
        value = peak / n
    else:
        value = (infected_at_step if infected_at_step is not None else 0) / n
    values.append(value)

estimate = float(np.mean(values))
lo = float(np.percentile(values, 2.5))
hi = float(np.percentile(values, 97.5))
lo = min(lo, estimate)
hi = max(hi, estimate)

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump(
        {"estimate": estimate, "confidence_interval": [lo, hi]}, fh
    )
