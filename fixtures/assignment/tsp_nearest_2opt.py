"""Trusted fixture candidate: nearest-neighbor + 2-opt TSP solver.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a tour starting at the
declared start.  Self-reported costs are never trusted by the host.
"""

import json
import os

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

costs = [list(map(float, row)) for row in problem["costs"]]
n = problem["size"]
start = problem.get("start", 0)


def tour_cost(tour):
    return sum(
        costs[tour[i]][tour[(i + 1) % n]] for i in range(n)
    )


unvisited = set(range(n))
unvisited.remove(start)
tour = [start]
current = start
while unvisited:
    nxt = min(unvisited, key=lambda node: costs[current][node])
    tour.append(nxt)
    unvisited.remove(nxt)
    current = nxt

improved = True
while improved:
    improved = False
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            a, b, c, d = tour[i - 1], tour[i], tour[j], tour[(j + 1) % n]
            if costs[a][b] + costs[c][d] > costs[a][c] + costs[b][d]:
                tour[i : j + 1] = reversed(tour[i : j + 1])
                improved = True

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"tour": tour}, fh)
