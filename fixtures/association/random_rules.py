"""Trusted fixture candidate: random association rules fallback.

Reads VES_DATA_DIR/train.csv (default /data) and writes
VES_OUTPUT_DIR/rules.json (default /output) with random rules from the train
item set.  Self-reported metrics are never trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

train = pd.read_csv(f"{DATA_DIR}/train.csv")


def key(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    return str(value)


items = sorted({key(value) for value in pd.unique(train["item"])})
rng = np.random.default_rng(7)
rules = []
seen = set()
for _ in range(30):
    if len(items) < 2:
        break
    pair = rng.choice(items, size=2, replace=False)
    canonical = (tuple(sorted([pair[0]])), tuple(sorted([pair[1]])))
    if canonical in seen:
        continue
    seen.add(canonical)
    rules.append({"antecedent": [pair[0]], "consequent": [pair[1]]})
if not rules and len(items) >= 2:
    rules.append({"antecedent": [items[0]], "consequent": [items[1]]})

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/rules.json", "w", encoding="utf-8") as fh:
    json.dump({"rules": rules}, fh)
