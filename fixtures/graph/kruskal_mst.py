"""Trusted fixture candidate: pure-Python Kruskal minimum spanning tree.

Reads VES_DATA_DIR/problem.json (default /data) and writes
VES_OUTPUT_DIR/solution.json (default /output) with exactly n-1 edges forming
a spanning tree.  Self-reported scores are never trusted by the host.
"""

import json
import os

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

nodes = problem["nodes"]
parent = {node: node for node in nodes}
rank = {node: 0 for node in nodes}


def find(node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def union(u, v):
    root_u = find(u)
    root_v = find(v)
    if root_u == root_v:
        return False
    if rank[root_u] < rank[root_v]:
        parent[root_u] = root_v
    elif rank[root_u] > rank[root_v]:
        parent[root_v] = root_u
    else:
        parent[root_v] = root_u
        rank[root_u] += 1
    return True


sorted_edges = sorted(
    problem["edges"], key=lambda edge: float(edge["weight"])
)
selected = []
for edge in sorted_edges:
    if len(selected) == len(nodes) - 1:
        break
    if union(edge["u"], edge["v"]):
        selected.append([edge["u"], edge["v"]])

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"edges": selected}, fh)
