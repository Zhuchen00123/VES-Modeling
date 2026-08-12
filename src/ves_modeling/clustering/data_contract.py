"""Clustering data contract validation (R12).

Public files: ``train.csv`` (feature matrix, no labels) and
``test_features.csv`` (samples to assign).  Host-only file:
``hidden_test_labels.csv`` (reference partition labels, str/finite numbers).

The prediction artifact is ``predictions.json``:
- input mode: ``{"labels": ["cluster_label", ...]}`` (test row order);
- id mode: ``{"predictions": [{"id": ..., "label": ...}, ...]}`` with exact
  coverage.

Cluster labels are arbitrary names; the host ARI/NMI/V-measure metrics are
permutation-invariant, so candidate names never need to match host names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ves_modeling.classification.data_contract import _label_key
from ves_modeling.regression.data_contract import (
    _check_ids,
    _check_no_duplicate_headers,
    _id_key,
    _is_valid_id,
    _raw_headers,
)

ROW_ORDERS = ("input", "id")
MIN_DISTINCT_LABELS = 2


@dataclass(frozen=True)
class ClusteringDataContract:
    """Public clustering input contract (never hidden values)."""

    label_column: str
    id_column: str | None
    row_order: str
    input_columns: tuple[str, ...]
    train_rows: int
    test_rows: int
    test_ids: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_column": self.label_column,
            "id_column": self.id_column,
            "row_order": self.row_order,
            "input_columns": list(self.input_columns),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
        }


def _validate_feature_frame(
    frame: pd.DataFrame, source: str
) -> None:
    """Feature matrices must be numeric and finite."""
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


def validate_clustering_data(
    public_dir: Path,
    *,
    label_column: str = "label",
    id_column: str | None = None,
    row_order: str = "input",
) -> ClusteringDataContract:
    """Validate candidate-visible clustering CSVs and return the contract."""
    if row_order not in ROW_ORDERS:
        raise ValueError(
            f"row_order must be one of {ROW_ORDERS}, got {row_order!r}"
        )
    if row_order == "id" and not id_column:
        raise ValueError("row_order='id' requires id_column")
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
    feature_columns = tuple(
        column for column in test.columns if column != id_column
    )
    if not feature_columns:
        raise ValueError(
            "at least one feature column is required "
            "(id is not a feature)"
        )
    _validate_feature_frame(train[list(feature_columns)], "train.csv")
    _validate_feature_frame(test[list(feature_columns)], "test_features.csv")
    test_ids: tuple[str, ...] | None = None
    if id_column:
        _check_ids(train, id_column, "train.csv")
        test_ids = _check_ids(test, id_column, "test_features.csv")
    return ClusteringDataContract(
        label_column=label_column,
        id_column=id_column,
        row_order=row_order,
        input_columns=feature_columns,
        train_rows=len(train),
        test_rows=len(test),
        test_ids=test_ids,
    )


def load_host_labels(
    host_dir: Path, contract: ClusteringDataContract
) -> tuple[list[str], tuple[str, ...]]:
    """Load reference partition labels as canonical keys.

    Returns ``(host_label_keys, distinct_label_keys)`` aligned to public
    test order.  The host partition must cover every test row and contain at
    least two distinct labels (no degenerate single-cluster partition).
    """
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
    raw_keys = [_label_key(value) for value in labels]
    if len(raw_keys) != contract.test_rows:
        raise ValueError(
            f"hidden labels count {len(raw_keys)} != test rows "
            f"{contract.test_rows}"
        )
    distinct = sorted(set(raw_keys))
    if len(distinct) < MIN_DISTINCT_LABELS:
        raise ValueError(
            "host partition must contain at least two distinct labels"
        )
    if contract.row_order == "input":
        return raw_keys, tuple(distinct)
    host_ids = _check_ids(host, contract.id_column, "hidden_test_labels.csv")
    test_ids = contract.test_ids
    if test_ids is None:
        raise ValueError("test ids missing from contract")
    if set(host_ids) != set(test_ids):
        raise ValueError(
            "hidden_test_labels.csv ids must match public test ids exactly"
        )
    order = {host_id: index for index, host_id in enumerate(host_ids)}
    aligned = [raw_keys[order[test_id]] for test_id in test_ids]
    return aligned, tuple(distinct)


def validate_predictions(
    payload: dict[str, Any],
    *,
    expected_count: int,
    test_ids: tuple[str, ...] | None = None,
    id_column: str | None = None,
) -> list[str]:
    """Validate a clustering artifact and return aligned label keys.

    ``test_ids=None`` is the array format ``{"labels": [...]}`` in test row
    order.  With ``test_ids`` the artifact must be
    ``{"predictions": [{"id": ..., "label": ...}, ...]}`` with exact id
    coverage.  At least two distinct labels are required.
    """
    if test_ids is None:
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
        if len(set(keys)) < MIN_DISTINCT_LABELS:
            raise ValueError(
                "predictions must contain at least two distinct cluster "
                "labels"
            )
        return keys

    if "predictions" not in payload:
        raise ValueError("missing required field 'predictions'")
    raw = payload["predictions"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError("'predictions' must be a JSON array")
    by_id: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(
                "id-mode predictions must be objects with 'id' and 'label'"
            )
        if set(item.keys()) != {"id", "label"}:
            raise ValueError(
                "id-mode prediction objects must contain exactly 'id' and "
                "'label'"
            )
        raw_id = item["id"]
        if not _is_valid_id(raw_id):
            raise ValueError("id must be a non-empty finite scalar")
        key = _id_key(raw_id)
        if key in by_id:
            raise ValueError(f"duplicate id in predictions: {key!r}")
        by_id[key] = _label_key(item["label"])
    expected_ids = set(test_ids)
    missing = [test_id for test_id in test_ids if test_id not in by_id]
    extra = [key for key in by_id if key not in expected_ids]
    if missing or extra:
        raise ValueError(
            "id-mode predictions ids must match public test ids exactly "
            f"(missing={missing}, extra={extra})"
        )
    keys = [by_id[test_id] for test_id in test_ids]
    if len(set(keys)) < MIN_DISTINCT_LABELS:
        raise ValueError(
            "predictions must contain at least two distinct cluster labels"
        )
    return keys
