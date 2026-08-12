"""Recommendation / matrix factorization data contract (R17).

Public files: ``train.csv`` (user_id, item_id, rating history) and
``test_features.csv`` (pairs to predict).  Host-only file:
``hidden_test_ratings.csv`` (same pairs + rating).

User/item ids follow the canonical string/finite-number key rules
(``1 == 1.0 == '1'``).  The prediction artifact is ``predictions.json``:
array mode aligned to test row order, or key mode with exact
``(user_id, item_id)`` coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class RecommendationDataContract:
    """Public recommendation input contract (never hidden values)."""

    user_id_column: str
    item_id_column: str
    rating_column: str
    row_order: str
    train_rows: int
    test_rows: int
    n_users: int
    n_items: int
    test_keys: tuple[tuple[str, str], ...] = field(
        default=(), repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id_column": self.user_id_column,
            "item_id_column": self.item_id_column,
            "rating_column": self.rating_column,
            "row_order": self.row_order,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "n_users": self.n_users,
            "n_items": self.n_items,
        }


def _key(value: Any) -> str:
    """Canonical user/item key (1, 1.0 and '1' are the same)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, str)
    ):
        raise ValueError(
            "user/item ids must be a scalar string or finite number, "
            f"got {type(value).__name__}"
        )
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("user/item ids must not be empty")
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("user/item ids must be finite")
    return _id_key(value)


def _validate_ids(
    frame: pd.DataFrame,
    id_column: str,
    source: str,
) -> list[str]:
    if id_column not in frame.columns:
        raise ValueError(
            f"{source} must contain id column {id_column!r}"
        )
    ids = frame[id_column]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        raise ValueError(f"{source} contains empty ids in {id_column!r}")
    return [_key(value) for value in ids]


def _validate_ratings(
    frame: pd.DataFrame,
    rating_column: str,
    source: str,
) -> np.ndarray:
    if rating_column not in frame.columns:
        raise ValueError(
            f"{source} must contain rating column {rating_column!r}"
        )
    try:
        ratings = frame[rating_column].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{source} rating column {rating_column!r} must be numeric"
        ) from exc
    if ratings.size == 0 or not np.isfinite(ratings).all():
        raise ValueError(
            f"{source} rating column {rating_column!r} must be non-empty "
            "and finite"
        )
    return ratings


def validate_recommendation_data(
    public_dir: Path,
    *,
    user_id_column: str = "user_id",
    item_id_column: str = "item_id",
    rating_column: str = "rating",
    row_order: str = "key",
) -> RecommendationDataContract:
    """Validate candidate-visible recommendation CSVs and return contract."""
    if row_order not in ROW_ORDERS:
        raise ValueError(
            f"row_order must be one of {ROW_ORDERS}, got {row_order!r}"
        )
    if not user_id_column.strip():
        raise ValueError("user_id_column must be non-empty")
    if not item_id_column.strip():
        raise ValueError("item_id_column must be non-empty")
    if not rating_column.strip():
        raise ValueError("rating_column must be non-empty")
    if len({user_id_column, item_id_column, rating_column}) != 3:
        raise ValueError(
            "user_id_column, item_id_column and rating_column must differ"
        )
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
    train_users = _validate_ids(train, user_id_column, "train.csv")
    train_items = _validate_ids(train, item_id_column, "train.csv")
    _validate_ratings(train, rating_column, "train.csv")
    test_users = _validate_ids(test, user_id_column, "test_features.csv")
    test_items = _validate_ids(test, item_id_column, "test_features.csv")
    # Every test user must appear with at least one pair (explicit contract
    # invariant; a user with zero test pairs cannot be evaluated).
    user_test_counts: dict[str, int] = {}
    for user_key in test_users:
        user_test_counts[user_key] = user_test_counts.get(user_key, 0) + 1
    if not user_test_counts or min(user_test_counts.values()) < 1:
        raise ValueError("every test user must have at least one test pair")
    train_user_set = set(train_users)
    train_item_set = set(train_items)
    for frame, source in ((train, "train.csv"), (test, "test_features.csv")):
        if frame.duplicated(
            subset=[user_id_column, item_id_column]
        ).any():
            raise ValueError(
                f"{source} contains duplicate (user, item) pairs"
            )
    test_keys = tuple(
        (user_key, item_key)
        for user_key, item_key in zip(test_users, test_items)
    )
    return RecommendationDataContract(
        user_id_column=user_id_column,
        item_id_column=item_id_column,
        rating_column=rating_column,
        row_order=row_order,
        train_rows=len(train),
        test_rows=len(test),
        n_users=len(train_user_set),
        n_items=len(train_item_set),
        test_keys=test_keys,
    )


def load_host_ratings(
    host_dir: Path, contract: RecommendationDataContract
) -> np.ndarray:
    """Load and validate hidden ratings, aligned to public test keys/order."""
    host_path = host_dir / "hidden_test_ratings.csv"
    _check_no_duplicate_headers(host_path, _raw_headers(host_path))
    host = pd.read_csv(host_path)
    ratings = _validate_ratings(host, contract.rating_column, "hidden")
    if ratings.size != contract.test_rows:
        raise ValueError(
            f"hidden ratings count {ratings.size} != test rows "
            f"{contract.test_rows}"
        )
    if contract.row_order == "input":
        return ratings
    host_users = _validate_ids(
        host, contract.user_id_column, "hidden_test_ratings.csv"
    )
    host_items = _validate_ids(
        host, contract.item_id_column, "hidden_test_ratings.csv"
    )
    host_keys = tuple(
        (user_key, item_key)
        for user_key, item_key in zip(host_users, host_items)
    )
    test_keys = contract.test_keys
    if len(set(host_keys)) != len(host_keys):
        raise ValueError("hidden ratings contain duplicate keys")
    if set(host_keys) != set(test_keys):
        raise ValueError(
            "hidden_test_ratings.csv keys must match public test keys exactly"
        )
    order = {host_key: index for index, host_key in enumerate(host_keys)}
    return np.asarray(
        [ratings[order[test_key]] for test_key in test_keys],
        dtype=np.float64,
    )


def validate_predictions(
    payload: dict[str, Any],
    *,
    expected_count: int,
    test_keys: tuple[tuple[str, str], ...] | None = None,
    key_columns: tuple[str, str] = ("user_id", "item_id"),
) -> np.ndarray:
    """Validate a recommendation artifact and return aligned values."""
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

    user_column, item_column = key_columns
    by_key: dict[tuple[str, str], float] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(
                "key-mode predictions must be objects with user id, item id "
                "and prediction"
            )
        if set(item.keys()) != {user_column, item_column, "prediction"}:
            raise ValueError(
                "key-mode prediction objects must contain exactly "
                f"{user_column!r}, {item_column!r} and 'prediction'"
            )
        user_key = _key(item[user_column])
        item_key = _key(item[item_column])
        key = (user_key, item_key)
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
