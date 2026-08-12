"""Survival analysis data contract (R21).

Public files: ``train.csv`` (time > 0 finite, event 0/1 censoring marker,
optional feature columns) and ``test_features.csv`` (individuals to score,
same features).  Host-only file: ``hidden_test_outcomes.csv`` (time/event for
test rows).

The prediction artifact is ``predictions.json``: input order or key mode
with canonical ids; values are risk scores (higher = higher risk) or
predicted times depending on ``output_kind``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ves_modeling.regression.data_contract import (
    _check_ids,
    _check_no_duplicate_headers,
    _id_key,
    _is_valid_id,
    _raw_headers,
)

ROW_ORDERS = ("input", "id")
OUTPUT_KINDS = ("risk_score", "time")


@dataclass(frozen=True)
class SurvivalDataContract:
    """Public survival input contract (never hidden outcomes)."""

    time_column: str
    event_column: str
    id_column: str | None
    row_order: str
    output_kind: str
    input_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    train_rows: int
    test_rows: int
    test_ids: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_column": self.time_column,
            "event_column": self.event_column,
            "id_column": self.id_column,
            "row_order": self.row_order,
            "output_kind": self.output_kind,
            "input_columns": list(self.input_columns),
            "feature_columns": list(self.feature_columns),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
        }


def _validate_features(
    frame: pd.DataFrame, columns: tuple[str, ...], source: str
) -> None:
    for column in columns:
        if pd.api.types.is_bool_dtype(frame[column].dtype):
            raise ValueError(
                f"{source} feature column {column!r} must not be boolean"
            )
    try:
        values = frame[list(columns)].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{source} feature columns must be numeric"
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{source} feature columns must be finite")


def validate_survival_data(
    public_dir: Path,
    *,
    time_column: str = "time",
    event_column: str = "event",
    id_column: str | None = None,
    row_order: str = "input",
    output_kind: str = "risk_score",
) -> SurvivalDataContract:
    """Validate candidate-visible survival CSVs and return the contract."""
    if row_order not in ROW_ORDERS:
        raise ValueError(
            f"row_order must be one of {ROW_ORDERS}, got {row_order!r}"
        )
    if row_order == "id" and not id_column:
        raise ValueError("row_order='id' requires id_column")
    if output_kind not in OUTPUT_KINDS:
        raise ValueError(f"output_kind must be one of {OUTPUT_KINDS}")
    if not time_column.strip():
        raise ValueError("time_column must be non-empty")
    if not event_column.strip():
        raise ValueError("event_column must be non-empty")
    if id_column and id_column == time_column:
        raise ValueError("id_column must differ from time_column")
    if id_column and id_column == event_column:
        raise ValueError("id_column must differ from event_column")
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
    if time_column not in train.columns:
        raise ValueError(
            f"train.csv must contain time column {time_column!r}"
        )
    if event_column not in train.columns:
        raise ValueError(
            f"train.csv must contain event column {event_column!r}"
        )
    if time_column in test.columns or event_column in test.columns:
        raise ValueError(
            "test_features.csv must not contain time/event columns"
        )
    try:
        times = train[time_column].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"train time column {time_column!r} must be numeric"
        ) from exc
    if times.size == 0 or not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError("train times must be finite and positive")
    events = train[event_column]
    if not set(events.unique().tolist()) <= {0, 1}:
        raise ValueError("train events must be 0/1")
    input_columns = tuple(test.columns)
    train_input = tuple(
        column
        for column in train.columns
        if column not in (time_column, event_column)
    )
    if list(input_columns) != list(train_input):
        raise ValueError(
            "test_features.csv columns must match train features exactly "
            "in name and order"
        )
    feature_columns = tuple(
        column for column in input_columns if column != id_column
    )
    if not feature_columns:
        raise ValueError(
            "at least one model feature column is required "
            "(id is not a feature)"
        )
    _validate_features(train, feature_columns, "train.csv")
    _validate_features(test, feature_columns, "test_features.csv")
    test_ids: tuple[str, ...] | None = None
    if id_column:
        _check_ids(train, id_column, "train.csv")
        test_ids = _check_ids(test, id_column, "test_features.csv")
    return SurvivalDataContract(
        time_column=time_column,
        event_column=event_column,
        id_column=id_column,
        row_order=row_order,
        output_kind=output_kind,
        input_columns=input_columns,
        feature_columns=feature_columns,
        train_rows=len(train),
        test_rows=len(test),
        test_ids=test_ids,
    )


def load_hidden_outcomes(
    host_dir: Path, contract: SurvivalDataContract
) -> tuple[np.ndarray, np.ndarray]:
    """Load hidden test outcomes (time, event), aligned to public order."""
    host_path = host_dir / "hidden_test_outcomes.csv"
    _check_no_duplicate_headers(host_path, _raw_headers(host_path))
    host = pd.read_csv(host_path)
    if contract.time_column not in host.columns:
        raise ValueError(
            "hidden_test_outcomes.csv must contain time column "
            f"{contract.time_column!r}"
        )
    if contract.event_column not in host.columns:
        raise ValueError(
            "hidden_test_outcomes.csv must contain event column "
            f"{contract.event_column!r}"
        )
    try:
        times = host[contract.time_column].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("hidden times must be numeric") from exc
    if times.size == 0 or not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError("hidden times must be finite and positive")
    events = host[contract.event_column]
    if not set(events.unique().tolist()) <= {0, 1}:
        raise ValueError("hidden events must be 0/1")
    if int(np.sum(events.to_numpy(dtype=np.int64))) < 1:
        raise ValueError(
            "hidden test outcomes must contain at least one event"
        )
    if times.size != contract.test_rows:
        raise ValueError(
            f"hidden outcomes count {times.size} != test rows "
            f"{contract.test_rows}"
        )
    event_values = events.to_numpy(dtype=np.int64)
    if contract.row_order == "input":
        return times, event_values
    host_ids = _check_ids(host, contract.id_column, "hidden_test_outcomes.csv")
    test_ids = contract.test_ids
    if test_ids is None:
        raise ValueError("test ids missing from contract")
    if set(host_ids) != set(test_ids):
        raise ValueError(
            "hidden_test_outcomes.csv ids must match public test ids exactly"
        )
    order = {host_id: index for index, host_id in enumerate(host_ids)}
    aligned_times = np.asarray(
        [times[order[test_id]] for test_id in test_ids], dtype=np.float64
    )
    aligned_events = np.asarray(
        [event_values[order[test_id]] for test_id in test_ids],
        dtype=np.int64,
    )
    return aligned_times, aligned_events


def validate_predictions(
    payload: dict[str, Any],
    *,
    expected_count: int,
    test_ids: tuple[str, ...] | None = None,
    id_column: str | None = None,
) -> np.ndarray:
    """Validate a survival artifact; returns aligned prediction values."""
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
                "prediction must be a number, "
                f"got {type(prediction).__name__}"
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
