"""Trusted fixture candidate: greedy + Hungarian assignment solver.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with an assignment
permutation.  Self-reported costs are never trusted by the host.
"""

import json
import os

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

costs = [list(map(float, row)) for row in problem["costs"]]
n = problem["size"]


def greedy_assignment(costs, n):
    assignment = [-1] * n
    used = set()
    for i in range(n):
        best = None
        best_j = None
        for j in range(n):
            if j in used:
                continue
            if best is None or costs[i][j] < best:
                best = costs[i][j]
                best_j = j
        assignment[i] = best_j
        used.add(best_j)
    return assignment


def hungarian_assignment(costs, n):
    """Compact pure-Python Hungarian (Kuhn-Munkres) for square matrices."""
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


assignment = hungarian_assignment(costs, n)

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"assignment": assignment}, fh)
