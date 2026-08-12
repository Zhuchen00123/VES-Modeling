"""Trusted fixture candidate: random-weight scalarization solution set.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a spread of feasible
solutions from random-weight scalarization (scipy linprog/milp) plus bounded
random samples.  Self-reported objective values are never trusted by the
host.
"""

import json
import os

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

variables = problem["variables"]
names = list(variables)
objective_maps = [
    objective.get("coefficients", {}) for objective in problem["objectives"]
]
objective_constants = [
    objective.get("constant", 0.0) for objective in problem["objectives"]
]
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
A_ub = []
b_ub = []
A_eq = []
b_eq = []
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
        A_ub.append(row)
        b_ub.append(rhs)
    elif sense == ">=":
        A_ub.append(-row)
        b_ub.append(-rhs)
    else:
        A_eq.append(row)
        b_eq.append(rhs)


def objective_vector(index):
    return np.asarray(
        [float(objective_maps[index].get(name, 0.0)) for name in names],
        dtype=float,
    )


objective_1 = objective_vector(0)
objective_2 = objective_vector(1)


def feasible(values):
    bounds_ok = not np.any(values < lower - 1e-9) and not np.any(
        values > upper + 1e-9
    )
    constraint_ok = not A_ub or not np.any(
        np.asarray(A_ub) @ values > np.asarray(b_ub) + 1e-9
    )
    equality_ok = not A_eq or not np.any(
        np.abs(np.asarray(A_eq) @ values - np.asarray(b_eq)) > 1e-9
    )
    integrality_ok = not integrality.any() or not np.any(
        np.abs(values - np.round(values)) > 1e-9
    )
    return bounds_ok and constraint_ok and equality_ok and integrality_ok


solutions = []
rng = np.random.default_rng(42)
for _ in range(14):
    weights = rng.dirichlet(np.ones(2))
    combined = (
        weights[0] * objective_1 + weights[1] * objective_2
    )
    if integrality.any():
        result = milp(
            -combined,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=[
                LinearConstraint(np.asarray(A_ub), -np.inf, np.asarray(b_ub))
            ]
            if A_ub
            else [],
        )
    else:
        result = linprog(
            -combined,
            A_ub=np.asarray(A_ub) if A_ub else None,
            b_ub=np.asarray(b_ub) if A_ub else None,
            A_eq=np.asarray(A_eq) if A_eq else None,
            b_eq=np.asarray(b_eq) if A_eq else None,
            bounds=list(zip(lower, upper)),
            method="highs",
        )
    if result.x is not None and feasible(result.x):
        solutions.append(result.x)

for _ in range(30):
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
