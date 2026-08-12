"""Trusted fixture candidate: mean-field ODE network-SIR fallback.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with the network-SIR quantity
from a mean-field Euler integration.  Self-reported references are never
trusted by the host.
"""

import json
import os

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

edge_count = len(problem["edges"])
k_avg = 2.0 * edge_count / n if n else 0.0

s = float(n - i0)
i = float(i0)
r = 0.0

steps = max(2000, round(t_end / 0.01))
dt = t_end / steps
peak = i
infected_at_t = None

for step in range(steps):
    rate = beta * k_avg * s * i / n
    ds = -rate
    di = rate - gamma * i
    dr = gamma * i
    s += dt * ds
    i += dt * di
    r += dt * dr
    peak = max(peak, i)
    if infected_at_t is None and (step + 1) * dt >= target_t:
        infected_at_t = i

if quantity == "final_size":
    estimate = (n - s) / n
elif quantity == "peak_infected":
    estimate = peak / n
else:
    estimate = (infected_at_t if infected_at_t is not None else i) / n

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"estimate": estimate}, fh)
