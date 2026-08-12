"""Trusted fixture candidate: Best-Fit bin packing."""

import json
import os

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

with open(f"{DATA_DIR}/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)

capacity = float(problem["capacity"])
items = [float(size) for size in problem["items"]]
bin_remaining = []
assignment = [0] * len(items)
for index, size in enumerate(items):
    best_index = None
    best_remaining = None
    for bin_index, remaining in enumerate(bin_remaining):
        if size <= remaining and (
            best_remaining is None or remaining < best_remaining
        ):
            best_index = bin_index
            best_remaining = remaining
    if best_index is None:
        best_index = len(bin_remaining)
        bin_remaining.append(capacity)
    bin_remaining[best_index] -= size
    assignment[index] = best_index

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"assignment": assignment}, fh)
