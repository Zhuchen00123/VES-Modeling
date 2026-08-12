"""Trusted fixture candidate: First-Fit-Decreasing bin packing."""

import json
import os

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

capacity = float(problem["capacity"])
items = [float(size) for size in problem["items"]]
order = sorted(range(len(items)), key=lambda i: items[i], reverse=True)
bin_remaining = []
assignment = [0] * len(items)
for index in order:
    placed = None
    for bin_index, remaining in enumerate(bin_remaining):
        if items[index] <= remaining:
            placed = bin_index
            break
    if placed is None:
        placed = len(bin_remaining)
        bin_remaining.append(capacity)
    bin_remaining[placed] -= items[index]
    assignment[index] = placed

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"assignment": assignment}, fh)
