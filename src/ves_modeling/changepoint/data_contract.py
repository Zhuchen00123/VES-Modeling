"""Time-series change-point detection data contract (R25).

Public files: ``train.csv`` (a time series ``t``, ``y``; finite, t strictly
increasing, n >= 40) and ``test_features.csv`` (a second series to detect
changes in; same shape).  Host-only file: ``hidden_test_changepoints.csv``
(true change-point indices for the test series, non-empty, strictly
increasing, in [1, n-2]).

Artifact ``changepoints.json``: ``{"changepoints": [index, ...]}`` with
strictly increasing integers in [1, n-2] and at least one index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ves_modeling.regression.data_contract import (
    _check_no_duplicate_headers,
    _raw_headers,
)


@dataclass(frozen=True)
class ChangepointDataContract:
    """Canonical public change-point input (never hidden truth)."""

    train_rows: int
    test_rows: int
    t_column: str
    y_column: str
    tolerance_window: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": [self.t_column, self.y_column],
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "changepoint_index_range": "[1, n-2]",
            "tolerance_window": self.tolerance_window,
        }


def _read_series(
    path: Path, *, t_column: str, y_column: str, source: str
) -> np.ndarray:
    """Read one public series and return its ``t`` column (validated)."""
    _check_no_duplicate_headers(path, _raw_headers(path))
    frame = pd.read_csv(path)
    if list(frame.columns) != [t_column, y_column]:
        raise ValueError(
            f"{source} must contain exactly columns {t_column!r} and "
            f"{y_column!r} in that order"
        )
    try:
        t = frame[t_column].to_numpy(dtype=np.float64)
        y = frame[y_column].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{source} columns must be numeric") from exc
    if t.size == 0 or not np.isfinite(t).all():
        raise ValueError(f"{source} t values must be finite")
    if not np.isfinite(y).all():
        raise ValueError(f"{source} y values must be finite")
    if t.size < 2 or (np.diff(t) <= 0).any():
        raise ValueError(f"{source} t must be strictly increasing")
    return t


def validate_changepoint_data(
    public_dir: Path,
    *,
    t_column: str = "t",
    y_column: str = "y",
    tolerance_window: int = 3,
) -> ChangepointDataContract:
    """Validate candidate-visible change-point CSVs and return the contract."""
    if not t_column.strip() or not y_column.strip():
        raise ValueError("t_column and y_column must be non-empty")
    if t_column == y_column:
        raise ValueError("t_column must differ from y_column")
    if isinstance(tolerance_window, bool) or not isinstance(
        tolerance_window, int
    ):
        raise ValueError("tolerance_window must be an integer")
    if tolerance_window < 0:
        raise ValueError("tolerance_window must be >= 0")
    public_dir = Path(public_dir)
    train_t = _read_series(
        public_dir / "train.csv",
        t_column=t_column,
        y_column=y_column,
        source="train.csv",
    )
    test_t = _read_series(
        public_dir / "test_features.csv",
        t_column=t_column,
        y_column=y_column,
        source="test_features.csv",
    )
    if train_t.size < 40:
        raise ValueError("train.csv must have at least 40 rows")
    if test_t.size < 3:
        raise ValueError(
            "test_features.csv must have at least 3 rows "
            "(indices 1..n-2 need data on both sides)"
        )
    return ChangepointDataContract(
        train_rows=int(train_t.size),
        test_rows=int(test_t.size),
        t_column=t_column,
        y_column=y_column,
        tolerance_window=tolerance_window,
    )


def _validate_indices(indices: np.ndarray, *, n: int, source: str) -> None:
    """Enforce non-empty, in-range and strictly increasing indices."""
    if indices.size == 0:
        raise ValueError(f"{source} must be non-empty")
    if (indices < 1).any() or (indices > n - 2).any():
        raise ValueError(f"{source} must lie in [1, n-2] with n={n}")
    if (np.diff(indices) <= 0).any():
        raise ValueError(f"{source} must be strictly increasing")


def load_hidden_changepoints(
    host_dir: Path, contract: ChangepointDataContract
) -> np.ndarray:
    """Load hidden true change-point indices for the test series."""
    path = Path(host_dir) / "hidden_test_changepoints.csv"
    _check_no_duplicate_headers(path, _raw_headers(path))
    frame = pd.read_csv(path)
    if list(frame.columns) != ["changepoint"]:
        raise ValueError(
            "hidden_test_changepoints.csv must contain exactly column "
            "'changepoint'"
        )
    raw = frame["changepoint"].to_numpy()
    if raw.size == 0:
        raise ValueError("hidden_test_changepoints.csv must be non-empty")
    values: list[int] = []
    for value in raw.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
            raise ValueError("hidden changepoints must be integers")
        values.append(value)
    indices = np.asarray(values, dtype=np.int64)
    _validate_indices(
        indices, n=contract.test_rows, source="hidden changepoints"
    )
    return indices


def validate_changepoints(
    payload: dict[str, Any], *, n: int
) -> np.ndarray:
    """Validate a change-point artifact; returns the index array."""
    if n < 3:
        raise ValueError("n must be at least 3")
    if "changepoints" not in payload:
        raise ValueError("missing required field 'changepoints'")
    raw = payload["changepoints"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError("'changepoints' must be a JSON array")
    if not raw:
        raise ValueError("'changepoints' must contain at least one index")
    values: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
            raise ValueError(f"changepoints[{index}] must be an integer")
        values.append(value)
    indices = np.asarray(values, dtype=np.int64)
    _validate_indices(indices, n=n, source="changepoints")
    return indices
