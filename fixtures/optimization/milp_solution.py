"""Trusted fixture candidate: SciPy milp (MILP) solver.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with all declared variables.
Self-reported objective/feasibility/optimality/gap are never trusted by the
host, which recomputes every fact from problem.json.
"""

import json
import os

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

variables = problem["variables"]
names = list(variables)
coefficients = problem["objective"].get("coefficients", {})
c = np.asarray(
    [float(coefficients.get(name, 0.0)) for name in names], dtype=float
)
if problem["sense"] == "maximize":
    c = -c
integrality = np.asarray(
    [
        1 if variables[name]["type"] in ("integer", "binary") else 0
        for name in names
    ],
    dtype=int,
)
bounds = Bounds(
    lb=[float(variables[name]["lower"]) for name in names],
    ub=[float(variables[name]["upper"]) for name in names],
)
constraints = []
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
        constraints.append(LinearConstraint(row, -np.inf, rhs))
    elif sense == ">=":
        constraints.append(LinearConstraint(row, rhs, np.inf))
    else:
        constraints.append(LinearConstraint(row, rhs, rhs))

result = milp(
    c,
    integrality=integrality,
    bounds=bounds,
    constraints=constraints,
)
if result.x is None:
    values = []
    for index, name in enumerate(names):
        value = (
            bounds.lb[index] + bounds.ub[index]
        ) / 2.0
        if integrality[index]:
            value = float(round(value))
        values.append(value)
else:
    values = [float(value) for value in result.x]

solution = {
    "variables": {
        name: float(value) for name, value in zip(names, values)
    }
}
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump(solution, fh)
