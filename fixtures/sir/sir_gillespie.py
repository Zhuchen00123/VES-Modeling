"""Trusted fixture candidate: stochastic Gillespie SIR with replications.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with the SIR quantity
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
N = int(problem["N"])
i0 = int(problem["i0"])
r0 = int(problem.get("r0", 0))
t_end = float(problem["t_end"])
quantity = problem["quantity"]
target_t = float(problem.get("t", t_end))

rng = np.random.default_rng(42)
replications = 60
values = []
for _ in range(replications):
    s = float(N - i0 - r0)
    i = float(i0)
    r = float(r0)
    time = 0.0
    peak = i
    infected_at_t = None
    while time < t_end:
        rate_inf = beta * s * i / N
        rate_rec = gamma * i
        total = rate_inf + rate_rec
        if total <= 0.0:
            break
        wait = float(rng.exponential(1.0 / total))
        time += wait
        if time >= t_end:
            if infected_at_t is None:
                infected_at_t = i
            break
        if infected_at_t is None and time >= target_t:
            infected_at_t = i
        if rng.random() < rate_inf / total:
            s -= 1.0
            i += 1.0
        else:
            i -= 1.0
            r += 1.0
        peak = max(peak, i)
    if quantity == "final_size":
        value = (N - s) / N
    elif quantity == "peak_infected":
        value = peak / N
    else:
        value = (infected_at_t if infected_at_t is not None else i) / N
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
