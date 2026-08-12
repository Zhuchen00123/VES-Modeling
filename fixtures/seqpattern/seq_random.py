"""Trusted fixture candidate: deterministic random-pattern fallback.

Reads VES_DATA_DIR/train.csv (default /data) and writes
VES_OUTPUT_DIR/patterns.json (default /output) with one pattern: the most
frequent event as prefix and the least frequent different event as suffix.
Self-reported metrics are never trusted by the host.
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

counts = collections.Counter(
    event for sequence in sequences for event in sequence
)
events = sorted(counts)
prefix = max(events, key=lambda event: (counts[event], event))
candidates = [event for event in events if event != prefix]
if candidates:
    suffix = min(candidates, key=lambda event: (counts[event], event))
    patterns = [{"prefix": [prefix], "suffix": [suffix]}]
else:
    patterns = []

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/patterns.json", "w", encoding="utf-8") as fh:
    json.dump({"patterns": patterns}, fh)
