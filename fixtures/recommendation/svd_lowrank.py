"""Trusted fixture candidate: SVD low-rank matrix completion.

Reads VES_DATA_DIR/train.csv and test_features.csv (default /data) and writes
VES_OUTPUT_DIR/predictions.json (default /output) with keyed predictions
from a numpy.linalg.svd low-rank reconstruction (fallback: bias baseline).
Self-reported scores are never trusted by the host.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")
ROW_ORDER = os.environ.get("VES_RECOMMENDATION_ROW_ORDER", "key")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
user_col = "user_id"
item_col = "item_id"


def key(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    return str(value)


global_mean = float(train["rating"].mean())
user_mean = train.groupby(user_col)["rating"].mean()
item_mean = train.groupby(item_col)["rating"].mean()
user_bias = (user_mean - global_mean).to_dict()
item_bias = (item_mean - global_mean).to_dict()

user_keys = [key(value) for value in pd.unique(train[user_col])]
item_keys = [key(value) for value in pd.unique(train[item_col])]
user_index = {value: index for index, value in enumerate(user_keys)}
item_index = {value: index for index, value in enumerate(item_keys)}
matrix = np.full((len(user_keys), len(item_keys)), global_mean)
for _, row in train.iterrows():
    u = key(row[user_col])
    i = key(row[item_col])
    matrix[user_index[u], item_index[i]] = float(row["rating"])

residuals = matrix - global_mean
u_matrix, s_vector, vt_matrix = np.linalg.svd(
    residuals, full_matrices=False
)
rank = min(5, len(s_vector))
reconstruction = (
    u_matrix[:, :rank]
    @ np.diag(s_vector[:rank])
    @ vt_matrix[:rank, :]
)
reconstruction += global_mean

records = []
for _, row in test.iterrows():
    u = key(row[user_col])
    i = key(row[item_col])
    if u in user_index and i in item_index:
        prediction = reconstruction[user_index[u], item_index[i]]
    else:
        prediction = (
            global_mean
            + user_bias.get(u, 0.0)
            + item_bias.get(i, 0.0)
        )
    records.append(
        {
            user_col: u,
            item_col: i,
            "prediction": float(prediction),
        }
    )

if ROW_ORDER == "input":
    payload = {"predictions": [record["prediction"] for record in records]}
else:
    payload = {"predictions": records}
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
