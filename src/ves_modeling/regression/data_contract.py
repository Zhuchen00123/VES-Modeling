"""Regression data contract validation (R7.3 Batch B).

Pre-execution checks on candidate-visible CSV inputs plus a shared prediction
artifact validator used by both the host verifier (search) and the application
path (apply).  Host labels are never part of the contract output.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROW_ORDERS = ("input", "id")


@dataclass(frozen=True)
class RegressionDataContract:
    """Public regression input contract (never hidden values)."""

    target_column: str
    id_column: str | None
    row_order: str
    input_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    train_rows: int
    test_rows: int
    test_ids: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "id_column": self.id_column,
            "row_order": self.row_order,
            "input_columns": list(self.input_columns),
            "feature_columns": list(self.feature_columns),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
        }


def _id_key(value: Any) -> str:
    """Stable string key for CSV/JSON ids (1 vs 1.0 vs '1' are the same)."""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float, str)):
        return str(value)
    raise ValueError(f"id must be a scalar, got {type(value).__name__}")


def _raw_headers(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.reader(handle))
    except StopIteration:
        raise ValueError(f"CSV is empty: {path.name}") from None
    return row


def _check_no_duplicate_headers(path: Path, headers: list[str]) -> None:
    if len(headers) != len(set(headers)):
        raise ValueError(f"duplicate column names in {path.name}")


def _check_ids(
    frame: pd.DataFrame, id_column: str, source: str
) -> tuple[str, ...]:
    if id_column not in frame.columns:
        raise ValueError(f"{source} must contain id column {id_column!r}")
    ids = frame[id_column]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        raise ValueError(f"{source} contains empty ids in column {id_column!r}")
    keys: list[str] = []
    for value in ids:
        if isinstance(value, bool):
            raise ValueError(f"{source} ids must not be booleans")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{source} ids must be finite")
        keys.append(_id_key(value))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{source} contains duplicate ids in column {id_column!r}")
    return tuple(keys)


def _is_valid_id(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return not (isinstance(value, str) and not value.strip())


def validate_regression_data(
    public_dir: Path,
    *,
    target_column: str = "target",
    id_column: str | None = None,
    row_order: str = "input",
) -> RegressionDataContract:
    """Validate candidate-visible CSVs and return the public contract."""
    if row_order not in ROW_ORDERS:
        raise ValueError(
            f"row_order must be one of {ROW_ORDERS}, got {row_order!r}"
        )
    if row_order == "id" and not id_column:
        raise ValueError("row_order='id' requires id_column")
    if not target_column or not str(target_column).strip():
        raise ValueError("target_column must be non-empty")
    if id_column and id_column == target_column:
        raise ValueError("id_column must differ from target_column")

    public_dir = Path(public_dir)
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test_features.csv"
    train_headers = _raw_headers(train_path)
    test_headers = _raw_headers(test_path)
    _check_no_duplicate_headers(train_path, train_headers)
    _check_no_duplicate_headers(test_path, test_headers)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if len(train) == 0:
        raise ValueError("train.csv must have at least one row")
    if len(test) == 0:
        raise ValueError("test_features.csv must have at least one row")
    if target_column not in train.columns:
        raise ValueError(
            f"train.csv must contain target column {target_column!r}"
        )
    if target_column in test.columns:
        raise ValueError(
            f"test_features.csv must not contain target column "
            f"{target_column!r}"
        )
    try:
        target_values = train[target_column].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"train target column {target_column!r} must be numeric"
        ) from exc
    if target_values.size == 0 or not np.isfinite(target_values).all():
        raise ValueError(
            "train target column must be non-empty and finite"
        )
    input_columns = tuple(test.columns)
    train_input = tuple(
        column for column in train.columns if column != target_column
    )
    if list(input_columns) != list(train_input):
        raise ValueError(
            "test_features.csv columns must match train features exactly "
            "in name and order"
        )
    test_ids: tuple[str, ...] | None = None
    if id_column:
        _check_ids(train, id_column, "train.csv")
        test_ids = _check_ids(test, id_column, "test_features.csv")
    feature_columns = tuple(
        column for column in input_columns if column != id_column
    )
    if not feature_columns:
        raise ValueError(
            "at least one model feature column is required "
            "(id is not a model feature)"
        )
    contract = RegressionDataContract(
        target_column=target_column,
        id_column=id_column,
        row_order=row_order,
        input_columns=input_columns,
        feature_columns=feature_columns,
        train_rows=len(train),
        test_rows=len(test),
        test_ids=test_ids,
    )
    return contract


def load_host_labels(
    host_dir: Path, contract: RegressionDataContract
) -> np.ndarray:
    """Load and validate host labels, aligned to public test order."""
    host_path = host_dir / "hidden_test_labels.csv"
    host_headers = _raw_headers(host_path)
    _check_no_duplicate_headers(host_path, host_headers)
    host = pd.read_csv(host_path)
    if contract.target_column not in host.columns:
        raise ValueError(
            "hidden_test_labels.csv must contain target column "
            f"{contract.target_column!r}"
        )
    try:
        labels = host[contract.target_column].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"hidden labels target column {contract.target_column!r} "
            "must be numeric"
        ) from exc
    if labels.size == 0 or not np.isfinite(labels).all():
        raise ValueError("hidden labels must be non-empty and finite")
    if labels.size != contract.test_rows:
        raise ValueError(
            f"hidden labels count {labels.size} != test rows "
            f"{contract.test_rows}"
        )
    if contract.id_column is None:
        return labels
    host_ids = _check_ids(host, contract.id_column, "hidden_test_labels.csv")
    test_ids = contract.test_ids
    if test_ids is None:
        raise ValueError("test ids missing from contract")
    if set(host_ids) != set(test_ids):
        raise ValueError(
            "hidden_test_labels.csv ids must match public test ids exactly"
        )
    order = {host_id: index for index, host_id in enumerate(host_ids)}
    return np.asarray(
        [labels[order[test_id]] for test_id in test_ids], dtype=np.float64
    )


def validate_predictions(
    payload: dict[str, Any],
    *,
    expected_count: int,
    test_ids: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Validate a prediction artifact and return the aligned value array.

    ``test_ids=None`` is the default array format (one value per test row, in
    test row order).  With ``test_ids`` the artifact must be a list of
    ``{"id": ..., "prediction": number}`` objects; missing, duplicate or extra
    ids are rejected and values are aligned to public test id order.
    """
    if "predictions" not in payload:
        raise ValueError("missing required field 'predictions'")
    raw = payload["predictions"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError("'predictions' must be a JSON array")
    if test_ids is None:
        if len(raw) != expected_count:
            raise ValueError(
                f"prediction count {len(raw)} != expected {expected_count}"
            )
        values: list[float] = []
        for item in raw:
            if isinstance(item, bool):
                raise ValueError("predictions must not contain booleans")
            if not isinstance(item, (int, float)):
                raise ValueError(
                    f"predictions must be numbers, got {type(item).__name__}"
                )
            value = float(item)
            if not math.isfinite(value):
                raise ValueError(
                    "predictions must be finite (no NaN/Infinity)"
                )
            values.append(value)
        return np.asarray(values, dtype=np.float64)

    by_id: dict[str, float] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(
                "id-mode predictions must be objects with 'id' and "
                "'prediction'"
            )
        if set(item.keys()) != {"id", "prediction"}:
            raise ValueError(
                "id-mode prediction objects must contain exactly 'id' and "
                "'prediction'"
            )
        raw_id = item["id"]
        if not _is_valid_id(raw_id):
            raise ValueError("id must be a non-empty finite scalar")
        key = _id_key(raw_id)
        if key in by_id:
            raise ValueError(f"duplicate id in predictions: {key!r}")
        prediction = item["prediction"]
        if isinstance(prediction, bool) or not isinstance(
            prediction, (int, float)
        ):
            raise ValueError(
                f"prediction must be a number, got "
                f"{type(prediction).__name__}"
            )
        value = float(prediction)
        if not math.isfinite(value):
            raise ValueError("predictions must be finite (no NaN/Infinity)")
        by_id[key] = value
    expected_ids = set(test_ids)
    missing = [test_id for test_id in test_ids if test_id not in by_id]
    extra = [key for key in by_id if key not in expected_ids]
    if missing or extra:
        raise ValueError(
            "id-mode predictions ids must match public test ids exactly "
            f"(missing={missing}, extra={extra})"
        )
    return np.asarray([by_id[test_id] for test_id in test_ids], dtype=np.float64)
