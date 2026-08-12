"""Anomaly detection data contract validation (R13).

Public files: ``train.csv`` (feature matrix of normal samples, no labels)
and ``test_features.csv`` (samples to score/label).  Host-only file:
``hidden_test_labels.csv`` (binary labels: ``normal``/``anomaly`` strings or
``0``/``1`` numbers, at least one of each class).

The prediction artifact is ``predictions.json`` with exactly one shape:
- score mode: ``{"scores": [number, ...]}`` (higher = more anomalous);
- label mode: ``{"labels": ["normal"|"anomaly"|0|1, ...]}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ves_modeling.classification.data_contract import _label_key
from ves_modeling.regression.data_contract import (
    _check_no_duplicate_headers,
    _raw_headers,
)

OUTPUT_MODES = ("score", "label")

POSITIVE_KEYS = {"s:anomaly", "n:1"}
NEGATIVE_KEYS = {"s:normal", "n:0"}
ALL_BINARY_KEYS = POSITIVE_KEYS | NEGATIVE_KEYS


@dataclass(frozen=True)
class AnomalyDataContract:
    """Public anomaly input contract (never hidden values)."""

    label_column: str
    input_columns: tuple[str, ...]
    train_rows: int
    test_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_column": self.label_column,
            "input_columns": list(self.input_columns),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
        }


def _validate_feature_frame(
    frame: pd.DataFrame, source: str
) -> None:
    for column in frame.columns:
        if pd.api.types.is_bool_dtype(frame[column].dtype):
            raise ValueError(
                f"{source} feature column {column!r} must not be boolean"
            )
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{source} feature columns must be numeric"
        ) from exc
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(
            f"{source} feature columns must be non-empty and finite"
        )


def validate_anomaly_data(
    public_dir: Path,
    *,
    label_column: str = "label",
) -> AnomalyDataContract:
    """Validate candidate-visible anomaly CSVs and return the contract."""
    if not label_column or not str(label_column).strip():
        raise ValueError("label_column must be non-empty")
    public_dir = Path(public_dir)
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test_features.csv"
    _check_no_duplicate_headers(train_path, _raw_headers(train_path))
    _check_no_duplicate_headers(test_path, _raw_headers(test_path))
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if len(train) == 0:
        raise ValueError("train.csv must have at least one row")
    if len(test) == 0:
        raise ValueError("test_features.csv must have at least one row")
    if list(test.columns) != list(train.columns):
        raise ValueError(
            "test_features.csv columns must match train.csv exactly "
            "in name and order"
        )
    input_columns = tuple(test.columns)
    if not input_columns:
        raise ValueError("at least one feature column is required")
    _validate_feature_frame(train, "train.csv")
    _validate_feature_frame(test, "test_features.csv")
    return AnomalyDataContract(
        label_column=label_column,
        input_columns=input_columns,
        train_rows=len(train),
        test_rows=len(test),
    )


def _binary_keys(keys: list[str], source: str) -> list[int]:
    """Validate binary label keys and map to 1 (anomaly) / 0 (normal)."""
    if not keys:
        raise ValueError(f"{source} labels must be non-empty")
    if any(key not in ALL_BINARY_KEYS for key in keys):
        raise ValueError(
            f"{source} labels must be 'normal'/'anomaly' or 0/1 binary "
            "classes"
        )
    has_strings = any(key.startswith("s:") for key in keys)
    has_numbers = any(key.startswith("n:") for key in keys)
    if has_strings and has_numbers:
        raise ValueError(
            f"{source} labels must not mix string and numeric encodings"
        )
    values = [
        1 if key in POSITIVE_KEYS else 0 for key in keys
    ]
    if len(set(values)) < 2:
        raise ValueError(
            f"{source} labels must contain at least one of each class"
        )
    return values


def load_host_labels(
    host_dir: Path, contract: AnomalyDataContract
) -> np.ndarray:
    """Load binary hidden labels as 0/1 (1 = anomaly)."""
    host_path = host_dir / "hidden_test_labels.csv"
    _check_no_duplicate_headers(host_path, _raw_headers(host_path))
    host = pd.read_csv(host_path)
    if contract.label_column not in host.columns:
        raise ValueError(
            "hidden_test_labels.csv must contain label column "
            f"{contract.label_column!r}"
        )
    labels = host[contract.label_column]
    if labels.isna().any():
        raise ValueError("hidden labels must not contain nulls")
    keys = [_label_key(value) for value in labels]
    if len(keys) != contract.test_rows:
        raise ValueError(
            f"hidden labels count {len(keys)} != test rows "
            f"{contract.test_rows}"
        )
    return np.asarray(_binary_keys(keys, "hidden"), dtype=np.int64)


def validate_predictions(
    payload: dict[str, Any],
    *,
    expected_count: int,
    mode: str,
) -> np.ndarray:
    """Validate an anomaly artifact.

    Score mode returns the raw score array; label mode returns binary 0/1
    (1 = anomaly).  Candidate self-reported metrics are ignored by the host.
    """
    if mode == "score":
        if "scores" not in payload:
            raise ValueError("missing required field 'scores'")
        raw = payload["scores"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
            raise ValueError("'scores' must be a JSON array")
        if len(raw) != expected_count:
            raise ValueError(
                f"score count {len(raw)} != expected {expected_count}"
            )
        values: list[float] = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    "scores must be numbers, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            if not np.isfinite(value):
                raise ValueError("scores must be finite (no NaN/Infinity)")
            values.append(value)
        return np.asarray(values, dtype=np.float64)
    if "labels" not in payload:
        raise ValueError("missing required field 'labels'")
    raw = payload["labels"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError("'labels' must be a JSON array")
    if len(raw) != expected_count:
        raise ValueError(
            f"label count {len(raw)} != expected {expected_count}"
        )
    keys = [_label_key(value) for value in raw]
    return np.asarray(_binary_keys(keys, "prediction"), dtype=np.int64)
