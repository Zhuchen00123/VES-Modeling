"""Trusted fixture candidate: deterministic 4th-order Runge-Kutta SIR.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with the SIR quantity from
RK4 integration.  Self-reported references are never trusted by the host.
"""

import json
import os

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

s = float(N - i0 - r0)
i = float(i0)
r = float(r0)

steps = max(2000, round(t_end / 0.001))
dt = t_end / steps
peak = i
infected_at_t = None

for step in range(steps):
    s1 = -beta * s * i / N
    i1 = beta * s * i / N - gamma * i
    r1 = gamma * i
    s2 = -beta * (s + 0.5 * dt * s1) * (i + 0.5 * dt * i1) / N
    i2 = (
        beta * (s + 0.5 * dt * s1) * (i + 0.5 * dt * i1) / N
        - gamma * (i + 0.5 * dt * i1)
    )
    r2 = gamma * (i + 0.5 * dt * i1)
    s3 = -beta * (s + 0.5 * dt * s2) * (i + 0.5 * dt * i2) / N
    i3 = (
        beta * (s + 0.5 * dt * s2) * (i + 0.5 * dt * i2) / N
        - gamma * (i + 0.5 * dt * i2)
    )
    r3 = gamma * (i + 0.5 * dt * i2)
    s4 = -beta * (s + dt * s3) * (i + dt * i3) / N
    i4 = beta * (s + dt * s3) * (i + dt * i3) / N - gamma * (i + dt * i3)
    r4 = gamma * (i + dt * i3)

    s += dt * (s1 + 2.0 * s2 + 2.0 * s3 + s4) / 6.0
    i += dt * (i1 + 2.0 * i2 + 2.0 * i3 + i4) / 6.0
    r += dt * (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
    peak = max(peak, i)
    if infected_at_t is None and (step + 1) * dt >= target_t:
        infected_at_t = i

if quantity == "final_size":
    estimate = (N - s) / N
elif quantity == "peak_infected":
    estimate = peak / N
else:
    estimate = (infected_at_t if infected_at_t is not None else i) / N

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"estimate": estimate}, fh)
