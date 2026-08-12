"""Trusted fixture candidate: simplified Apriori association rules.

Reads VES_DATA_DIR/train.csv (default /data) and writes
VES_OUTPUT_DIR/rules.json (default /output) with rules from frequent
2-itemsets.  Self-reported metrics are never trusted by the host.
"""

import json
import os
from collections import Counter

import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
transactions = (
    train.groupby("transaction_id")["item"].apply(set).tolist()
)


def key(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    return str(value)


transactions = [
    {key(item) for item in transaction} for transaction in transactions
]
item_counts = Counter(
    item for transaction in transactions for item in transaction
)
n = len(transactions)
min_support = max(2, int(0.2 * n))
frequent = {
    item for item, count in item_counts.items() if count >= min_support
}

rules = []
seen = set()
for transaction in transactions:
    frequent_items = sorted(frequent & transaction)
    for a_index, a in enumerate(frequent_items):
        for b in frequent_items[a_index + 1 :]:
            for antecedent, consequent in ((a, b), (b, a)):
                canonical = (
                    tuple(sorted([antecedent])),
                    tuple(sorted([consequent])),
                )
                if canonical in seen:
                    continue
                seen.add(canonical)
                rules.append(
                    {
                        "antecedent": [antecedent],
                        "consequent": [consequent],
                    }
                )
if not rules:
    # Fallback: single rule from the two most frequent items.
    top = [item for item, _count in item_counts.most_common(2)]
    if len(top) == 2:
        rules.append({"antecedent": [top[0]], "consequent": [top[1]]})

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/rules.json", "w", encoding="utf-8") as fh:
    json.dump({"rules": rules}, fh)
