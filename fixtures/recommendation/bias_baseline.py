"""Trusted fixture candidate: global mean + user/item bias baseline.

Reads VES_DATA_DIR/train.csv and test_features.csv (default /data) and writes
VES_OUTPUT_DIR/predictions.json (default /output) with keyed predictions.
Self-reported scores are never trusted by the host.
"""

import json
import os

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

records = []
for _, row in test.iterrows():
    user_key = key(row[user_col])
    item_key = key(row[item_col])
    prediction = (
        global_mean
        + user_bias.get(user_key, 0.0)
        + item_bias.get(item_key, 0.0)
    )
    records.append(
        {
            user_col: user_key,
            item_col: item_key,
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
