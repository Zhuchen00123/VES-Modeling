"""Classification data contract validation (R9).

Pre-execution checks on candidate-visible CSV inputs plus a shared prediction
artifact validator used by both the host verifier (search) and the application
path (apply).  Host labels never become part of the contract output.

Class encoding is host-fixed: the contract carries the class order (explicit
``classes`` or first appearance in train), and candidates must emit labels
matching that order plus per-class probabilities.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
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

PROB_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ClassificationDataContract:
    """Public classification input contract (never hidden values)."""

    label_column: str
    id_column: str | None
    row_order: str
    input_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    classes: tuple[Any, ...] = field(repr=False, compare=False)
    class_keys: tuple[str, ...] = field(repr=False, compare=False)
    class_counts: tuple[int, ...] = field(repr=False, compare=False)
    train_rows: int
    test_rows: int
    test_ids: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_column": self.label_column,
            "id_column": self.id_column,
            "row_order": self.row_order,
            "input_columns": list(self.input_columns),
            "feature_columns": list(self.feature_columns),
            "classes": list(self.classes),
            "class_counts": list(self.class_counts),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
        }


def _label_key(value: Any) -> str:
    """Canonical class key: non-empty str or finite number only.

    ``s:``/``n:`` prefixes keep ``"1"`` and ``1`` distinct classes while
    ``1`` and ``1.0`` canonicalize to the same number key.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("class labels must not be booleans")
    if isinstance(value, numbers.Integral):
        return f"n:{int(value)}"
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("class labels must be finite")
        if number.is_integer():
            return f"n:{int(number)}"
        return f"n:{number}"
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("class labels must not be empty")
        return f"s:{value}"
    raise ValueError(
        "class labels must be a non-empty string or finite number, "
        f"got {type(value).__name__}"
    )


def _validate_classes(
    classes: Sequence[Any] | None,
    train_labels: pd.Series,
    source: str,
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Resolve host-fixed class order and validate it against train labels."""
    if classes is not None:
        if not isinstance(classes, (list, tuple)):
            raise ValueError(
                "classes must be a list or tuple of class labels"
            )
        if len(classes) < 2:
            raise ValueError("at least two unique classes are required")
        if len(set(classes)) != len(classes):
            raise ValueError("classes must be unique")
        class_values = tuple(_json_scalar(value) for value in classes)
        class_keys = tuple(_label_key(value) for value in class_values)
    else:
        seen: list[Any] = []
        seen_keys: set[str] = set()
        for value in train_labels:
            key = _label_key(value)
            if key not in seen_keys:
                seen_keys.add(key)
                seen.append(_json_scalar(value))
        if len(seen) < 2:
            raise ValueError(
                f"{source} must contain at least two distinct classes"
            )
        class_values = tuple(seen)
        class_keys = tuple(_label_key(value) for value in seen)
    train_keys = {_label_key(value) for value in train_labels}
    missing = [key for key in class_keys if key not in train_keys]
    if missing:
        raise ValueError(
            "every declared class must appear in train "
            f"(missing in {source}: {missing})"
        )
    if classes is not None:
        outside = sorted(train_keys - set(class_keys))
        if outside:
            raise ValueError(
                "train contains labels outside the declared classes: "
                f"{outside}"
            )
    return class_values, class_keys


def _json_scalar(value: Any) -> Any:
    """Python scalar for JSON (numpy scalars -> int/float/str)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("class labels must not be booleans")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("class labels must be finite")
        return int(number) if number.is_integer() else number
    if isinstance(value, str):
        return value
    raise ValueError(
        f"class labels must be a string or finite number, "
        f"got {type(value).__name__}"
    )


def validate_classification_data(
    public_dir: Path,
    *,
    label_column: str = "target",
    id_column: str | None = None,
    row_order: str = "input",
    classes: Sequence[Any] | None = None,
) -> ClassificationDataContract:
    """Validate candidate-visible classification CSVs and return contract."""
    if row_order not in ROW_ORDERS:
        raise ValueError(
            f"row_order must be one of {ROW_ORDERS}, got {row_order!r}"
        )
    if row_order == "id" and not id_column:
        raise ValueError("row_order='id' requires id_column")
    if not label_column or not str(label_column).strip():
        raise ValueError("label_column must be non-empty")
    if id_column and id_column == label_column:
        raise ValueError("id_column must differ from label_column")

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
    if label_column not in train.columns:
        raise ValueError(
            f"train.csv must contain label column {label_column!r}"
        )
    if label_column in test.columns:
        raise ValueError(
            f"test_features.csv must not contain label column "
            f"{label_column!r}"
        )
    train_labels = train[label_column]
    if train_labels.isna().any():
        raise ValueError(
            f"train label column {label_column!r} must not contain nulls"
        )
    class_values, class_keys = _validate_classes(
        classes, train_labels, "train.csv"
    )
    input_columns = tuple(test.columns)
    train_input = tuple(
        column for column in train.columns if column != label_column
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
    train_key_counts = {
        key: 0 for key in class_keys
    }
    for value in train_labels:
        train_key_counts[_label_key(value)] += 1
    class_counts = tuple(
        train_key_counts[key] for key in class_keys
    )
    return ClassificationDataContract(
        label_column=label_column,
        id_column=id_column,
        row_order=row_order,
        input_columns=input_columns,
        feature_columns=feature_columns,
        classes=class_values,
        class_keys=tuple(class_keys),
        class_counts=class_counts,
        train_rows=len(train),
        test_rows=len(test),
        test_ids=test_ids,
    )


def load_host_labels(
    host_dir: Path, contract: ClassificationDataContract
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load host labels as class indices, aligned to public test order.

    Returns ``(label_indices, class_keys)``.  Host labels must cover every
    declared class (so AUROC/OVR metrics are computable).
    """
    host_path = host_dir / "hidden_test_labels.csv"
    host_headers = _raw_headers(host_path)
    _check_no_duplicate_headers(host_path, host_headers)
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
    key_to_index = {
        key: index for index, key in enumerate(contract.class_keys)
    }
    indices = np.asarray(
        [key_to_index[key] for key in raw_keys], dtype=np.int64
    )
    if indices.size == 0:
        raise ValueError("hidden labels must be non-empty")
    if set(indices.tolist()) != set(range(contract.n_classes)):
        raise ValueError(
            "hidden labels must cover every declared class"
        )
    if indices.size != contract.test_rows:
        raise ValueError(
            f"hidden labels count {indices.size} != test rows "
            f"{contract.test_rows}"
        )
    if contract.row_order == "input":
        return indices, contract.class_keys
    host_ids = _check_ids(
        host, contract.id_column, "hidden_test_labels.csv"
    )
    test_ids = contract.test_ids
    if test_ids is None:
        raise ValueError("test ids missing from contract")
    if set(host_ids) != set(test_ids):
        raise ValueError(
            "hidden_test_labels.csv ids must match public test ids exactly"
        )
    order = {host_id: index for index, host_id in enumerate(host_ids)}
    return (
        np.asarray(
            [indices[order[test_id]] for test_id in test_ids],
            dtype=np.int64,
        ),
        contract.class_keys,
    )


def _argmax_tie_first(probabilities: np.ndarray) -> int:
    """Index of the maximum; on ties the first (lowest index) wins."""
    return int(np.argmax(probabilities))


def validate_predictions(
    payload: dict[str, Any],
    *,
    expected_count: int,
    n_classes: int,
    class_keys: tuple[str, ...],
    test_ids: tuple[str, ...] | None = None,
    id_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a classification artifact and return aligned label indices
    plus the probability matrix (rows x classes).

    ``test_ids=None`` is the array format: one ``{"label", "probabilities"}``
    record per test row in test row order.  With ``test_ids`` records must be
    ``{"id", "label", "probabilities"}`` and align to public test id order.
    Top-level claimed fields (``claimed_accuracy`` etc.) are ignored.
    """
    if "predictions" not in payload:
        raise ValueError("missing required field 'predictions'")
    raw = payload["predictions"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError("'predictions' must be a JSON array")
    key_to_index = {
        key: index for index, key in enumerate(class_keys)
    }
    indices: list[int] = []
    probabilities: list[list[float]] = []
    if test_ids is None:
        if len(raw) != expected_count:
            raise ValueError(
                f"prediction count {len(raw)} != expected {expected_count}"
            )
        for item in raw:
            index, probs = _validate_record(
                item,
                n_classes=n_classes,
                key_to_index=key_to_index,
            )
            indices.append(index)
            probabilities.append(probs)
    else:
        by_id: dict[str, tuple[int, list[float]]] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(
                    "id-mode predictions must be objects with 'id', 'label' "
                    "and 'probabilities'"
                )
            if set(item.keys()) != {"id", "label", "probabilities"}:
                raise ValueError(
                    "id-mode prediction objects must contain exactly 'id', "
                    "'label' and 'probabilities'"
                )
            raw_id = item["id"]
            if not _is_valid_id(raw_id):
                raise ValueError("id must be a non-empty finite scalar")
            key = _id_key(raw_id)
            if key in by_id:
                raise ValueError(f"duplicate id in predictions: {key!r}")
            by_id[key] = _validate_record(
                item,
                n_classes=n_classes,
                key_to_index=key_to_index,
            )
        expected_ids = set(test_ids)
        missing = [test_id for test_id in test_ids if test_id not in by_id]
        extra = [key for key in by_id if key not in expected_ids]
        if missing or extra:
            raise ValueError(
                "id-mode predictions ids must match public test ids exactly "
                f"(missing={missing}, extra={extra})"
            )
        indices = [by_id[test_id][0] for test_id in test_ids]
        probabilities = [by_id[test_id][1] for test_id in test_ids]
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
    )


def _validate_record(
    item: Any,
    *,
    n_classes: int,
    key_to_index: dict[str, int],
) -> tuple[int, list[float]]:
    if not isinstance(item, dict):
        raise ValueError(
            "prediction records must be objects with 'label' and "
            "'probabilities'"
        )
    if "label" not in item or "probabilities" not in item:
        raise ValueError(
            "prediction records must contain 'label' and 'probabilities'"
        )
    label_key = _label_key(item["label"])
    if label_key not in key_to_index:
        raise ValueError(
            f"label {item['label']!r} is outside the declared classes"
        )
    label_index = key_to_index[label_key]
    probabilities = item["probabilities"]
    if isinstance(probabilities, (str, bytes)) or not isinstance(
        probabilities, list
    ):
        raise ValueError("'probabilities' must be a JSON array")
    if len(probabilities) != n_classes:
        raise ValueError(
            f"probabilities length {len(probabilities)} != n_classes "
            f"{n_classes}"
        )
    values: list[float] = []
    for probability in probabilities:
        if isinstance(probability, bool) or not isinstance(
            probability, (int, float)
        ):
            raise ValueError(
                "probabilities must be numbers, "
                f"got {type(probability).__name__}"
            )
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError(
                "probabilities must be finite (no NaN/Infinity)"
            )
        if value < 0.0 or value > 1.0:
            raise ValueError("probabilities must be within [0, 1]")
        values.append(value)
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=PROB_SUM_TOLERANCE):
        raise ValueError(
            "probabilities must sum to 1 within tolerance "
            f"{PROB_SUM_TOLERANCE}"
        )
    argmax_index = _argmax_tie_first(np.asarray(values, dtype=np.float64))
    if argmax_index != label_index:
        raise ValueError(
            "label must equal the class-order argmax of probabilities "
            "(ties: first class wins)"
        )
    return label_index, values
