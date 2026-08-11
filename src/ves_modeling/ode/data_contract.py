"""ODE differential-equation modeling data contract (R11).

Public files: ``train.csv`` (observed t/y points, optional trajectory_id),
``test_features.csv`` (points to predict).  Host-only file:
``hidden_test_values.csv``.  ``t`` is a finite number, strictly increasing
per trajectory; ``(trajectory_id, t)`` keys never repeat inside train or
test.

The prediction artifact is ``predictions.json``: array mode aligned to test
row order, or key mode ``{"trajectory_id": ..., "t": ..., "prediction": n}``
with exact coverage when trajectories are present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ves_modeling.regression.data_contract import (
    _check_no_duplicate_headers,
    _id_key,
    _raw_headers,
)

ROW_ORDERS = ("input", "key")

MIN_ROWS_SINGLE_TRAJECTORY = 16
MIN_ROWS_PER_TRAJECTORY = 8


@dataclass(frozen=True)
class OdeDataContract:
    """Public ODE input contract (never hidden values)."""

    time_column: str
    value_column: str
    trajectory_id_column: str | None
    row_order: str
    train_rows: int
    test_rows: int
    n_trajectories: int
    input_columns: tuple[str, ...] = field(repr=False, compare=False)
    test_keys: tuple[tuple[str, float], ...] = field(
        default=(), repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_column": self.time_column,
            "value_column": self.value_column,
            "trajectory_id_column": self.trajectory_id_column,
            "row_order": self.row_order,
            "input_columns": list(self.input_columns),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "n_trajectories": self.n_trajectories,
        }


def _trajectory_key(value: Any) -> str:
    """Canonical trajectory key (same rules as regression ids)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, str)
    ):
        raise ValueError(
            "trajectory ids must be a scalar string or finite number, "
            f"got {type(value).__name__}"
        )
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("trajectory ids must not be empty")
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("trajectory ids must be finite")
    return _id_key(value)


def _t_value(value: Any) -> float:
    """Finite numeric t value (bool/NaN/Inf rejected)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float)
    ):
        raise ValueError(
            "t must be a finite number, "
            f"got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("t must be finite (no NaN/Infinity)")
    return number


def _validate_trajectories(
    frame: pd.DataFrame,
    *,
    time_column: str,
    trajectory_id_column: str | None,
    source: str,
    value_column: str | None = None,
    require_increasing: bool = True,
) -> tuple[int, list[tuple[str, float]]]:
    """Validate per-trajectory structure and return canonical keys."""
    if trajectory_id_column is None:
        if value_column is not None:
            try:
                values = frame[value_column].to_numpy(dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{source} value column {value_column!r} must be numeric"
                ) from exc
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(
                    f"{source} value column {value_column!r} must be "
                    "non-empty and finite"
                )
        times = [_t_value(value) for value in frame[time_column]]
        if require_increasing:
            _check_strictly_increasing(times, source, "trajectory")
        return 1, [("", time_value) for time_value in times]
    if trajectory_id_column not in frame.columns:
        raise ValueError(
            f"{source} must contain trajectory id column "
            f"{trajectory_id_column!r}"
        )
    ids = frame[trajectory_id_column]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        raise ValueError(
            f"{source} contains empty trajectory ids in column "
            f"{trajectory_id_column!r}"
        )
    raw_ids = [_trajectory_key(value) for value in ids]
    times = [_t_value(value) for value in frame[time_column]]
    if value_column is not None:
        try:
            values = frame[value_column].to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{source} value column {value_column!r} must be numeric"
            ) from exc
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(
                f"{source} value column {value_column!r} must be non-empty "
                "and finite"
            )
    grouped: dict[str, list[float]] = {}
    order: list[str] = []
    for trajectory_key, time_value in zip(raw_ids, times):
        if trajectory_key not in grouped:
            grouped[trajectory_key] = []
            order.append(trajectory_key)
        grouped[trajectory_key].append(time_value)
    for trajectory_key in order:
        trajectory_times = grouped[trajectory_key]
        if require_increasing:
            _check_strictly_increasing(
                trajectory_times, source, f"trajectory {trajectory_key!r}"
            )
        if len(trajectory_times) != len(set(trajectory_times)):
            raise ValueError(
                f"{source} contains duplicate t within trajectory "
                f"{trajectory_key!r}"
            )
    # Keys preserve public row order so key-mode predictions align to the
    # test file exactly (grouping above was only for validation).
    keys = list(zip(raw_ids, times))
    return len(order), keys


def _check_strictly_increasing(
    times: list[float], source: str, label: str
) -> None:
    for previous, current in pairwise(times):
        if current <= previous:
            raise ValueError(
                f"{source} t must be strictly increasing per {label}"
            )


def validate_ode_data(
    public_dir: Path,
    *,
    time_column: str = "t",
    value_column: str = "y",
    trajectory_id_column: str | None = None,
    row_order: str = "input",
) -> OdeDataContract:
    """Validate candidate-visible ODE CSVs and return the public contract."""
    if row_order not in ROW_ORDERS:
        raise ValueError(
            f"row_order must be one of {ROW_ORDERS}, got {row_order!r}"
        )
    if row_order == "key" and not trajectory_id_column:
        raise ValueError(
            "row_order='key' requires trajectory_id_column"
        )
    if not time_column.strip():
        raise ValueError("time_column must be non-empty")
    if not value_column.strip():
        raise ValueError("value_column must be non-empty")
    if trajectory_id_column and trajectory_id_column == value_column:
        raise ValueError(
            "trajectory_id_column must differ from value_column"
        )
    if trajectory_id_column == time_column:
        raise ValueError("trajectory_id_column must differ from time_column")

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
    for frame, source in ((train, "train.csv"), (test, "test_features.csv")):
        if time_column not in frame.columns:
            raise ValueError(
                f"{source} must contain time column {time_column!r}"
            )
    if value_column not in train.columns:
        raise ValueError(
            f"train.csv must contain value column {value_column!r}"
        )
    if value_column in test.columns:
        raise ValueError(
            f"test_features.csv must not contain value column "
            f"{value_column!r}"
        )
    train_input = tuple(
        column for column in train.columns if column != value_column
    )
    input_columns = tuple(test.columns)
    if list(input_columns) != list(train_input):
        raise ValueError(
            "test_features.csv columns must match train features exactly "
            "in name and order"
        )
    n_train_trajectories, _train_keys = _validate_trajectories(
        train,
        time_column=time_column,
        trajectory_id_column=trajectory_id_column,
        source="train.csv",
        value_column=value_column,
    )
    n_test_trajectories, test_keys = _validate_trajectories(
        test,
        time_column=time_column,
        trajectory_id_column=trajectory_id_column,
        source="test_features.csv",
    )
    if trajectory_id_column is None:
        if len(train) < MIN_ROWS_SINGLE_TRAJECTORY:
            raise ValueError(
                f"train.csv needs at least {MIN_ROWS_SINGLE_TRAJECTORY} rows "
                "for a single trajectory"
            )
        if len(test) < 1:
            raise ValueError("test_features.csv needs at least one point")
    else:
        train_counts: dict[str, int] = {}
        for key in _train_keys:
            train_counts[key[0]] = train_counts.get(key[0], 0) + 1
        test_counts: dict[str, int] = {}
        for key in test_keys:
            test_counts[key[0]] = test_counts.get(key[0], 0) + 1
        for trajectory_key, count in train_counts.items():
            if count < MIN_ROWS_PER_TRAJECTORY:
                raise ValueError(
                    f"train trajectory {trajectory_key!r} needs at least "
                    f"{MIN_ROWS_PER_TRAJECTORY} rows"
                )
        missing_history = sorted(set(test_counts) - set(train_counts))
        if missing_history:
            raise ValueError(
                "every test trajectory must appear in train.csv "
                f"(missing history: {missing_history})"
            )
    for frame, source in ((train, "train.csv"), (test, "test_features.csv")):
        subset = [trajectory_id_column, time_column] if trajectory_id_column else [time_column]
        if frame.duplicated(subset=subset).any():
            raise ValueError(
                f"{source} contains duplicate (trajectory, t) pairs"
            )
    return OdeDataContract(
        time_column=time_column,
        value_column=value_column,
        trajectory_id_column=trajectory_id_column,
        row_order=row_order,
        train_rows=len(train),
        test_rows=len(test),
        n_trajectories=max(n_train_trajectories, n_test_trajectories),
        input_columns=input_columns,
        test_keys=tuple(test_keys),
    )


def load_host_values(
    host_dir: Path, contract: OdeDataContract
) -> np.ndarray:
    """Load and validate host values, aligned to public test keys/order."""
    host_path = host_dir / "hidden_test_values.csv"
    _check_no_duplicate_headers(host_path, _raw_headers(host_path))
    host = pd.read_csv(host_path)
    if contract.value_column not in host.columns:
        raise ValueError(
            "hidden_test_values.csv must contain value column "
            f"{contract.value_column!r}"
        )
    if contract.time_column not in host.columns:
        raise ValueError(
            "hidden_test_values.csv must contain time column "
            f"{contract.time_column!r}"
        )
    try:
        values = host[contract.value_column].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"hidden values column {contract.value_column!r} must be numeric"
        ) from exc
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("hidden values must be non-empty and finite")
    if values.size != contract.test_rows:
        raise ValueError(
            f"hidden values count {values.size} != test rows "
            f"{contract.test_rows}"
        )
    if contract.row_order == "input":
        return values
    _n_host_trajectories, host_keys = _validate_trajectories(
        host,
        time_column=contract.time_column,
        trajectory_id_column=contract.trajectory_id_column,
        source="hidden_test_values.csv",
        require_increasing=False,
    )
    test_keys = contract.test_keys
    if set(host_keys) != set(test_keys):
        raise ValueError(
            "hidden_test_values.csv keys must match public test keys exactly"
        )
    order = {host_key: index for index, host_key in enumerate(host_keys)}
    return np.asarray(
        [values[order[test_key]] for test_key in test_keys],
        dtype=np.float64,
    )


def validate_predictions(
    payload: dict[str, Any],
    *,
    expected_count: int,
    test_keys: tuple[tuple[str, float], ...] | None = None,
    key_columns: tuple[str, str] = ("trajectory_id", "t"),
) -> np.ndarray:
    """Validate an ODE prediction artifact and return aligned values."""
    if "predictions" not in payload:
        raise ValueError("missing required field 'predictions'")
    raw = payload["predictions"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError("'predictions' must be a JSON array")
    if test_keys is None:
        if len(raw) != expected_count:
            raise ValueError(
                f"prediction count {len(raw)} != expected {expected_count}"
            )
        values: list[float] = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    "predictions must be numbers, "
                    f"got {type(item).__name__}"
                )
            value = float(item)
            if not math.isfinite(value):
                raise ValueError(
                    "predictions must be finite (no NaN/Infinity)"
                )
            values.append(value)
        return np.asarray(values, dtype=np.float64)

    trajectory_column, time_column = key_columns
    by_key: dict[tuple[str, float], float] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(
                "key-mode predictions must be objects with trajectory id, "
                "t and prediction"
            )
        if set(item.keys()) != {trajectory_column, time_column, "prediction"}:
            raise ValueError(
                "key-mode prediction objects must contain exactly "
                f"{trajectory_column!r}, {time_column!r} and 'prediction'"
            )
        trajectory_key = _trajectory_key(item[trajectory_column])
        time_value = _t_value(item[time_column])
        key = (trajectory_key, time_value)
        if key in by_key:
            raise ValueError(f"duplicate key in predictions: {key!r}")
        prediction = item["prediction"]
        if isinstance(prediction, bool) or not isinstance(
            prediction, (int, float)
        ):
            raise ValueError(
                "prediction must be a number, "
                f"got {type(prediction).__name__}"
            )
        value = float(prediction)
        if not math.isfinite(value):
            raise ValueError("predictions must be finite (no NaN/Infinity)")
        by_key[key] = value
    expected_keys = set(test_keys)
    missing = [key for key in test_keys if key not in by_key]
    extra = [key for key in by_key if key not in expected_keys]
    if missing or extra:
        raise ValueError(
            "key-mode predictions keys must match public test keys exactly "
            f"(missing={missing}, extra={extra})"
        )
    return np.asarray(
        [by_key[key] for key in test_keys], dtype=np.float64
    )
