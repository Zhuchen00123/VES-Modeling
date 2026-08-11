"""Forecasting data contract validation (R8).

Pre-execution checks on candidate-visible CSV inputs plus a shared prediction
artifact validator used by both the host verifier (search) and the application
path (apply).  Host labels never become part of the contract output.

Long-format time series:

  train.csv / test_features.csv / hidden_test_labels.csv (host-only)
  series_id, timestamp, [exogenous columns...], target (train/host only)

Series ids repeat across rows (one row per ``(series_id, timestamp)``), so the
prediction artifact is keyed by ``(series_id, timestamp)`` and interleaved
test row order can never misalign values.  Timestamps must be strict ISO 8601
strings (numeric timestamps are rejected) and must follow the declared
frequency, which is validated through the real pandas offset machinery.
"""

from __future__ import annotations

import math
import numbers
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from ves_modeling.regression.data_contract import (
    _check_no_duplicate_headers,
    _raw_headers,
)

ROW_ORDERS = ("input", "key")

_ISO_TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[Tt ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:[Zz]|[+-]\d{2}:?\d{2})?)?$"
)


@dataclass(frozen=True)
class ForecastingDataContract:
    """Public forecasting input contract (never hidden values)."""

    time_column: str
    series_id_column: str
    target_column: str
    frequency: str
    row_order: str
    input_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    train_rows: int
    test_rows: int
    horizon: int
    n_series: int
    series_ids: tuple[str, ...] = field(
        default=(), repr=False, compare=False
    )
    test_keys: tuple[tuple[str, str], ...] = field(
        default=(), repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_column": self.time_column,
            "series_id_column": self.series_id_column,
            "target_column": self.target_column,
            "frequency": self.frequency,
            "row_order": self.row_order,
            "input_columns": list(self.input_columns),
            "feature_columns": list(self.feature_columns),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "horizon": self.horizon,
            "n_series": self.n_series,
        }


def _series_key(value: Any) -> str:
    """Canonical string key for a series id (1, 1.0 and '1' are the same)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("series ids must not be booleans")
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("series ids must be finite")
        if number.is_integer():
            return str(int(number))
        return str(number)
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("series ids must not be empty")
        return value
    raise ValueError(
        "series ids must be a scalar string or finite number, "
        f"got {type(value).__name__}"
    )


def _canonical_time(value: Any) -> str:
    """Strict ISO 8601 string -> canonical instant key (numeric rejected)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, str):
        raise ValueError(
            "timestamps must be strict ISO 8601 strings "
            "(numeric timestamps are rejected)"
        )
    if _ISO_TIME_RE.fullmatch(value) is None:
        raise ValueError(
            f"timestamp {value!r} is not a strict ISO 8601 string"
        )
    try:
        parsed = pd.to_datetime(value, format="ISO8601")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"timestamp {value!r} is not valid ISO 8601: {exc}"
        ) from None
    if pd.isna(parsed):
        raise ValueError(f"timestamp {value!r} is empty/invalid")
    return parsed.isoformat()


def _validate_series_ids(
    frame: pd.DataFrame, series_id_column: str, source: str
) -> tuple[str, ...]:
    if series_id_column not in frame.columns:
        raise ValueError(f"{source} must contain series id column "
                         f"{series_id_column!r}")
    ids = frame[series_id_column]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        raise ValueError(
            f"{source} contains empty series ids in column "
            f"{series_id_column!r}"
        )
    keys = tuple(_series_key(value) for value in ids)
    if any(not key for key in keys):
        raise ValueError(f"{source} contains empty series ids")
    return keys


def _validate_times(
    frame: pd.DataFrame, time_column: str, source: str
) -> tuple[str, ...]:
    if time_column not in frame.columns:
        raise ValueError(f"{source} must contain time column {time_column!r}")
    times = frame[time_column]
    if pd.api.types.is_numeric_dtype(times.dtype):
        raise ValueError(
            f"{source} time column {time_column!r} must be strict ISO 8601 "
            "strings (numeric timestamps are rejected)"
        )
    if times.isna().any() or (times.astype(str).str.strip() == "").any():
        raise ValueError(f"{source} contains empty timestamps")
    return tuple(_canonical_time(value) for value in times)


def _validate_frequency_sequence(
    frame: pd.DataFrame,
    series_id_column: str,
    time_column: str,
    source: str,
    frequency: str,
) -> None:
    """Real-offset validation: frequency must parse, and every series must
    sit on offset boundaries and follow the offset exactly."""
    offset = to_offset(frequency)
    canonical_series = [
        _series_key(value) for value in frame[series_id_column]
    ]
    grouped = pd.DataFrame(
        {
            "series": canonical_series,
            "timestamp": frame[time_column].to_numpy(),
        }
    ).groupby("series", sort=False)
    for key, group in grouped:
        times = group["timestamp"]
        if times.duplicated().any():
            raise ValueError(
                f"{source} contains duplicate timestamps for series {key!r}"
            )
        parsed = [pd.Timestamp(_canonical_time(value)) for value in times]
        if parsed != sorted(parsed):
            raise ValueError(
                f"{source} timestamps for series {key!r} must be sorted "
                "ascending"
            )
        start = parsed[0]
        if offset.rollback(start) != start or offset.rollforward(start) != start:
            raise ValueError(
                f"{source} series {key!r} first timestamp {start} does not "
                f"align to frequency {frequency!r}"
            )
        if len(parsed) > 1:
            expected = pd.date_range(
                start=start, periods=len(parsed), freq=offset
            )
            if list(expected) != parsed:
                raise ValueError(
                    f"{source} series {key!r} timestamps do not follow "
                    f"frequency {frequency!r}"
                )


def validate_forecasting_data(
    public_dir: Path,
    *,
    time_column: str = "timestamp",
    series_id_column: str = "series_id",
    target_column: str = "target",
    frequency: str = "D",
    row_order: str = "key",
) -> ForecastingDataContract:
    """Validate candidate-visible forecasting CSVs and return the contract."""
    if row_order not in ROW_ORDERS:
        raise ValueError(
            f"row_order must be one of {ROW_ORDERS}, got {row_order!r}"
        )
    if not time_column.strip():
        raise ValueError("time_column must be non-empty")
    if not series_id_column.strip():
        raise ValueError("series_id_column must be non-empty")
    if not target_column.strip():
        raise ValueError("target_column must be non-empty")
    if len({time_column, series_id_column, target_column}) != 3:
        raise ValueError(
            "time_column, series_id_column and target_column must differ"
        )
    try:
        to_offset(frequency)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"frequency {frequency!r} is not a real pandas offset: {exc}"
        ) from None

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
    for frame, source in ((train, "train.csv"), (test, "test_features.csv")):
        if series_id_column not in frame.columns:
            raise ValueError(
                f"{source} must contain series id column "
                f"{series_id_column!r}"
            )
        if time_column not in frame.columns:
            raise ValueError(
                f"{source} must contain time column {time_column!r}"
            )
    if target_column not in train.columns:
        raise ValueError(
            f"train.csv must contain target column {target_column!r}"
        )
    if target_column in test.columns:
        raise ValueError(
            "test_features.csv must not contain target column "
            f"{target_column!r}"
        )
    try:
        target_values = train[target_column].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"train target column {target_column!r} must be numeric"
        ) from exc
    if target_values.size == 0 or not np.isfinite(target_values).all():
        raise ValueError("train target column must be non-empty and finite")

    _validate_series_ids(train, series_id_column, "train.csv")
    _validate_series_ids(test, series_id_column, "test_features.csv")
    _validate_times(train, time_column, "train.csv")
    _validate_times(test, time_column, "test_features.csv")
    _validate_frequency_sequence(
        train, series_id_column, time_column, "train.csv", frequency
    )
    _validate_frequency_sequence(
        test, series_id_column, time_column, "test_features.csv", frequency
    )
    for frame, source in ((train, "train.csv"), (test, "test_features.csv")):
        duplicates = frame.duplicated(
            subset=[series_id_column, time_column]
        )
        if duplicates.any():
            raise ValueError(
                f"{source} contains duplicate (series, time) pairs"
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
    feature_columns = tuple(
        column
        for column in input_columns
        if column not in (series_id_column, time_column)
    )

    train_series = set(
        _validate_series_ids(train, series_id_column, "train.csv")
    )
    test_series_raw = _validate_series_ids(
        test, series_id_column, "test_features.csv"
    )
    test_series_set = set(test_series_raw)
    missing_history = sorted(test_series_set - train_series)
    if missing_history:
        raise ValueError(
            "every test series must appear in train.csv "
            f"(missing history: {missing_history})"
        )

    per_series = pd.Series(test_series_raw).value_counts()
    horizon = int(per_series.iloc[0])
    if (per_series != horizon).any():
        raise ValueError(
            "every test series must have the same number of rows (horizon)"
        )

    test_times = _validate_times(test, time_column, "test_features.csv")
    series_ids: list[str] = []
    seen: set[str] = set()
    for key in test_series_raw:
        if key not in seen:
            seen.add(key)
            series_ids.append(key)
    test_keys = tuple(
        (series_key, time_key)
        for series_key, time_key in zip(test_series_raw, test_times)
    )
    return ForecastingDataContract(
        time_column=time_column,
        series_id_column=series_id_column,
        target_column=target_column,
        frequency=frequency,
        row_order=row_order,
        input_columns=input_columns,
        feature_columns=feature_columns,
        series_ids=tuple(series_ids),
        test_keys=test_keys,
        train_rows=len(train),
        test_rows=len(test),
        horizon=horizon,
        n_series=len(series_ids),
    )


def load_host_labels(
    host_dir: Path, contract: ForecastingDataContract
) -> np.ndarray:
    """Load and validate host labels, aligned to public test keys/order."""
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
    if contract.row_order == "input":
        return labels
    host_series = _validate_series_ids(
        host, contract.series_id_column, "hidden_test_labels.csv"
    )
    host_times = _validate_times(
        host, contract.time_column, "hidden_test_labels.csv"
    )
    duplicates = host.duplicated(
        subset=[contract.series_id_column, contract.time_column]
    )
    if duplicates.any():
        raise ValueError(
            "hidden_test_labels.csv contains duplicate (series, time) pairs"
        )
    host_keys = tuple(
        (series_key, time_key)
        for series_key, time_key in zip(host_series, host_times)
    )
    test_keys = contract.test_keys
    if len(set(host_keys)) != len(host_keys):
        raise ValueError("hidden_test_labels.csv contains duplicate keys")
    if set(host_keys) != set(test_keys):
        raise ValueError(
            "hidden_test_labels.csv keys must match public test keys exactly"
        )
    order = {host_key: index for index, host_key in enumerate(host_keys)}
    return np.asarray(
        [labels[order[test_key]] for test_key in test_keys],
        dtype=np.float64,
    )


def validate_predictions(
    payload: dict[str, Any],
    *,
    expected_count: int,
    test_keys: tuple[tuple[str, str], ...] | None = None,
    key_columns: tuple[str, str] = ("series_id", "timestamp"),
) -> np.ndarray:
    """Validate a forecast artifact and return the aligned value array.

    ``test_keys=None`` is the array format (one value per test row, in test
    row order).  With ``test_keys`` the artifact must be a list of
    ``{<series id column>: ..., <time column>: "...", "prediction": number}``
    objects; missing, duplicate or extra keys are rejected and values are
    aligned to public test key order (interleaved rows stay safe).
    """
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
            if isinstance(item, bool):
                raise ValueError("predictions must not contain booleans")
            if not isinstance(item, (int, float)):
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

    series_column, time_column = key_columns
    by_key: dict[tuple[str, str], float] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(
                "key-mode predictions must be objects with series id, "
                "timestamp and prediction"
            )
        if set(item.keys()) != {series_column, time_column, "prediction"}:
            raise ValueError(
                "key-mode prediction objects must contain exactly "
                f"{series_column!r}, {time_column!r} and 'prediction'"
            )
        series_key = _series_key(item[series_column])
        time_key = _canonical_time(item[time_column])
        key = (series_key, time_key)
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
