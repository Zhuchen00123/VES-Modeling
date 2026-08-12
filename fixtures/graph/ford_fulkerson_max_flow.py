"""Trusted fixture candidate: pure-Python Ford-Fulkerson max flow.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with a flow value per declared
directed edge.  Self-reported scores are never trusted by the host.
"""

import json
import os

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

source = problem["source"]
target = problem["target"]
capacity = {}
for edge in problem["edges"]:
    capacity[(edge["u"], edge["v"])] = float(edge["weight"])
adjacency = {node: [] for node in problem["nodes"]}
for (u, v) in capacity:
    adjacency[u].append(v)
    adjacency[v].append(u)
flow = {edge_key: 0.0 for edge_key in capacity}


def residual(u, v):
    if (u, v) in capacity:
        return capacity[(u, v)] - flow.get((u, v), 0.0)
    return flow.get((v, u), 0.0)


def dfs(node, bottleneck, visited):
    if node == target:
        return bottleneck
    visited.add(node)
    for neighbor in adjacency[node]:
        if neighbor in visited:
            continue
        cap = residual(node, neighbor)
        if cap <= 0.0:
            continue
        sent = dfs(neighbor, min(bottleneck, cap), visited)
        if sent > 0.0:
            if (node, neighbor) in capacity:
                flow[(node, neighbor)] = flow.get((node, neighbor), 0.0) + sent
            else:
                flow[(neighbor, node)] = flow.get((neighbor, node), 0.0) - sent
            return sent
    return 0.0


while True:
    sent = dfs(source, float("inf"), set())
    if sent <= 0.0:
        break

payload = {
    "flow": {
        f"{u}->{v}": float(value) for (u, v), value in flow.items()
    }
}
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
