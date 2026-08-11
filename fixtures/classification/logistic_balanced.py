"""Trusted fixture candidate: balanced LogisticRegression classifier.

Reads VES_DATA_DIR (default /data) and writes
VES_OUTPUT_DIR/predictions.json (default /output).  The class order is first
appearance in train (matching the host contract when classes=None); labels
are the host-order argmax with ties choosing the first class; probabilities
are reordered to that host order.  Class values are converted to Python
scalars so numeric classes serialize to JSON.  Self-reported scores are never
trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

LABEL_COL = "target"


def to_scalar(value):
    return value.item() if hasattr(value, "item") else value


train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
features = list(test.columns)
classes = [to_scalar(value) for value in pd.unique(train[LABEL_COL])]
class_to_index = {value: index for index, value in enumerate(classes)}
y_index = np.asarray(
    [class_to_index[to_scalar(value)] for value in train[LABEL_COL]],
    dtype=np.int64,
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(train[features].to_numpy(dtype=np.float64), y_index)
probabilities = model.predict_proba(
    test[features].to_numpy(dtype=np.float64)
)
model_class_list = [int(value) for value in model.classes_]
column_of_index = {
    int(value): column for column, value in enumerate(model.classes_)
}
ordered = np.zeros((probabilities.shape[0], len(classes)), dtype=np.float64)
for host_index, class_value in enumerate(classes):
    ordered[:, host_index] = probabilities[
        :, column_of_index[class_to_index[class_value]]
    ]
label_indices = np.argmax(ordered, axis=1)  # ties: first class wins
rows = [
    {
        "label": to_scalar(classes[int(label_index)]),
        "probabilities": [float(value) for value in ordered[row_index]],
    }
    for row_index, label_index in enumerate(label_indices)
]
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": rows}, fh)
