"""Trusted fixture candidate: SciPy linprog (continuous LP) solver.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with all declared variables.
Self-reported objective/feasibility/optimality/gap are never trusted by the
host, which recomputes every fact from problem.json.
"""

import json
import os

from scipy.optimize import linprog

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

variables = problem["variables"]
names = list(variables)
coefficients = problem["objective"].get("coefficients", {})
c = [float(coefficients.get(name, 0.0)) for name in names]
if problem["sense"] == "maximize":
    c = [-value for value in c]
bounds = [
    (float(variables[name]["lower"]), float(variables[name]["upper"]))
    for name in names
]
A_ub = []
b_ub = []
A_eq = []
b_eq = []
for constraint in problem.get("constraints", []):
    row = [
        float(constraint["coefficients"].get(name, 0.0))
        for name in names
    ]
    rhs = float(constraint["rhs"])
    sense = constraint["sense"]
    if sense == "<=":
        A_ub.append(row)
        b_ub.append(rhs)
    elif sense == ">=":
        A_ub.append([-value for value in row])
        b_ub.append(-rhs)
    else:
        A_eq.append(row)
        b_eq.append(rhs)

result = linprog(
    c,
    A_ub=A_ub or None,
    b_ub=b_ub or None,
    A_eq=A_eq or None,
    b_eq=b_eq or None,
    bounds=bounds,
    method="highs",
)
if result.x is None:
    values = [
        (lower + upper) / 2.0 for lower, upper in bounds
    ]
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
