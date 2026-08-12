"""Trusted fixture candidate: bounded random sampling solution set.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with feasible solutions from
bounded random sampling.  Self-reported objective values are never trusted
by the host.
"""

import json
import os

import numpy as np

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

variables = problem["variables"]
names = list(variables)
lower = np.asarray(
    [float(variables[name]["lower"]) for name in names], dtype=float
)
upper = np.asarray(
    [float(variables[name]["upper"]) for name in names], dtype=float
)
integrality = np.asarray(
    [
        1 if variables[name]["type"] in ("integer", "binary") else 0
        for name in names
    ],
    dtype=int,
)
A = []
b = []
for constraint in problem.get("constraints", []):
    row = np.asarray(
        [
            float(constraint["coefficients"].get(name, 0.0))
            for name in names
        ],
        dtype=float,
    )
    rhs = float(constraint["rhs"])
    sense = constraint["sense"]
    if sense == "<=":
        A.append(row)
        b.append(rhs)
    elif sense == ">=":
        A.append(-row)
        b.append(-rhs)
    else:
        A.append(row)
        b.append(rhs)
        A.append(-row)
        b.append(-rhs)


def feasible(values):
    bounds_ok = not np.any(values < lower - 1e-9) and not np.any(
        values > upper + 1e-9
    )
    constraint_ok = not A or not np.any(
        np.asarray(A) @ values > np.asarray(b) + 1e-9
    )
    return bounds_ok and constraint_ok


solutions = []
rng = np.random.default_rng(7)
for _ in range(400):
    candidate = rng.uniform(lower, upper)
    if integrality.any():
        candidate = np.round(candidate)
    if feasible(candidate):
        solutions.append(candidate)

if not solutions:
    solutions.append((lower + upper) / 2.0)

unique = []
seen = set()
for solution in solutions:
    key = tuple(np.round(solution, 9).tolist())
    if key not in seen:
        seen.add(key)
        unique.append(solution)

payload = {
    "solutions": [
        {
            "variables": {
                name: float(value)
                for name, value in zip(names, solution)
            }
        }
        for solution in unique
    ]
}
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
