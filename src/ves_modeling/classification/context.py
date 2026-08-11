"""Host-owned classification verification context (hidden truth stays host)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from ves.context import VerificationContext


class ClassificationVerificationContext(VerificationContext):
    """Holds hidden label indices plus the host-fixed class order.

    Only ``id`` and a one-way ``fingerprint()`` are exposed to records;
    labels are never serialized and never mounted into candidate containers.

    Invariant: hidden label indices are all in ``[0, n_classes)`` and cover
    every declared class; ``classes`` and ``class_keys`` have equal length
    with at least two unique classes.
    """

    def __init__(
        self,
        hidden_labels: np.ndarray,
        *,
        dataset_name: str = "classification",
        expected_count: int | None = None,
        classes: tuple[Any, ...] = (),
        class_keys: tuple[str, ...] | None = None,
        id_column: str | None = None,
        prediction_ids: tuple[str, ...] | None = None,
        row_order: str = "input",
    ) -> None:
        self._labels = np.asarray(hidden_labels, dtype=np.int64).reshape(-1)
        if self._labels.size == 0:
            raise ValueError("hidden labels must be non-empty")
        if expected_count is not None and expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if expected_count is not None and expected_count != self._labels.size:
            raise ValueError("expected_count must match hidden labels size")
        if len(set(classes)) < 2:
            raise ValueError("at least two unique classes are required")
        if class_keys is None:
            raise ValueError("class_keys are required")
        if len(classes) != len(class_keys):
            raise ValueError("classes and class_keys must have equal length")
        n_classes = len(classes)
        if self._labels.size and (
            self._labels.min() < 0 or self._labels.max() >= n_classes
        ):
            raise ValueError(
                "hidden label indices must be within [0, n_classes)"
            )
        if set(self._labels.tolist()) != set(range(n_classes)):
            raise ValueError(
                "hidden labels must cover every declared class"
            )
        if row_order not in ("input", "id"):
            raise ValueError("row_order must be 'input' or 'id'")
        self._dataset_name = dataset_name
        self._classes = tuple(classes)
        self._class_keys = tuple(class_keys)
        self._id_column = id_column
        self._row_order = row_order
        self._expected_count = (
            int(self._labels.size) if expected_count is None else expected_count
        )
        if row_order == "id":
            if id_column is None:
                raise ValueError(
                    "id_column is required when row_order='id'"
                )
            if prediction_ids is None:
                raise ValueError(
                    "prediction_ids are required when row_order='id'"
                )
            if len(prediction_ids) != self._labels.size:
                raise ValueError(
                    "prediction_ids must match hidden labels size"
                )
            self._prediction_ids = tuple(prediction_ids)
        else:
            if prediction_ids is not None:
                raise ValueError(
                    "prediction_ids are only used when row_order='id'"
                )
            self._prediction_ids = None

    @property
    def id(self) -> str:
        return f"classification:{self._dataset_name}"

    @property
    def expected_count(self) -> int:
        return self._expected_count

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    @property
    def class_keys(self) -> tuple[str, ...]:
        return self._class_keys

    @property
    def n_classes(self) -> int:
        return len(self._classes)

    @property
    def id_column(self) -> str | None:
        return self._id_column

    @property
    def row_order(self) -> str:
        return self._row_order

    @property
    def prediction_ids(self) -> tuple[str, ...] | None:
        return self._prediction_ids

    def hidden_labels(self) -> np.ndarray:
        """Host-only accessor; verifier uses this inside the host boundary."""
        return self._labels

    def fingerprint(self) -> str:
        """One-way digest of hidden labels + class order (reversible
        summaries forbidden)."""
        digest = hashlib.sha256(self._labels.tobytes()).hexdigest()
        prediction_ids_sha256 = None
        if self._prediction_ids is not None:
            canonical_ids = json.dumps(
                list(self._prediction_ids),
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            prediction_ids_sha256 = hashlib.sha256(
                canonical_ids
            ).hexdigest()
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "count": self._expected_count,
                "classes": list(self._classes),
                "id_column": self._id_column,
                "row_order": self._row_order,
                "prediction_ids_sha256": prediction_ids_sha256,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload + digest.encode("utf-8")).hexdigest()
