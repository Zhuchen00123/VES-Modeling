"""Trusted fixture candidate: pure-Python Dijkstra shortest path.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a simple source->target
path.  Self-reported scores are never trusted by the host.
"""

import heapq
import json
import os

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

source = problem["source"]
target = problem["target"]
adjacency = {node: [] for node in problem["nodes"]}
for edge in problem["edges"]:
    u = edge["u"]
    v = edge["v"]
    weight = float(edge["weight"])
    adjacency[u].append((v, weight))
    adjacency[v].append((u, weight))

distances = {node: float("inf") for node in adjacency}
previous = {node: None for node in adjacency}
distances[source] = 0.0
queue = [(0.0, source)]
while queue:
    distance, node = heapq.heappop(queue)
    if distance > distances[node]:
        continue
    for neighbor, weight in adjacency[node]:
        candidate = distance + weight
        if candidate < distances[neighbor]:
            distances[neighbor] = candidate
            previous[neighbor] = node
            heapq.heappush(queue, (candidate, neighbor))

if distances[target] == float("inf"):
    path = [source, target]
else:
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"path": path}, fh)
