"""Trusted fixture candidate: simplified PrefixSpan (contiguous pairs).

Reads VES_DATA_DIR/train.csv (default /data) and writes
VES_OUTPUT_DIR/patterns.json (default /output) with the most frequent
contiguous two-event patterns.  Self-reported metrics are never trusted by
the host.
"""

import collections
import json
import os

import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
raw: dict[str, list[tuple[int, str]]] = {}
for _, row in train.iterrows():
    sid = str(row["sequence_id"])
    raw.setdefault(sid, []).append((int(row["step"]), str(row["event"])))
sequences = [
    [event for _, event in sorted(items)] for items in raw.values()
]

support = collections.Counter()
for sequence in sequences:
    seen = set()
    for i in range(len(sequence) - 1):
        seen.add((sequence[i], sequence[i + 1]))
    for pair in seen:
        support[pair] += 1

patterns = []
for (a, b), _ in sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
    patterns.append({"prefix": [a], "suffix": [b]})

if not patterns:
    events = sorted({event for sequence in sequences for event in sequence})
    if len(events) >= 2:
        patterns = [{"prefix": [events[0]], "suffix": [events[1]]}]

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/patterns.json", "w", encoding="utf-8") as fh:
    json.dump({"patterns": patterns}, fh)
