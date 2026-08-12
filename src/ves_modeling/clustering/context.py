"""Host-owned clustering verification context (reference labels stay host)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from ves.context import VerificationContext

from ves_modeling.clustering.data_contract import MIN_DISTINCT_LABELS


class ClusteringVerificationContext(VerificationContext):
    """Holds hidden reference labels plus optional public test features.

    Invariant: hidden labels are non-empty with at least two distinct
    classes; expected_count matches; id mode requires matching prediction
    ids; input mode forbids them.
    """

    def __init__(
        self,
        hidden_labels: list[str],
        *,
        dataset_name: str = "clustering",
        expected_count: int | None = None,
        id_column: str | None = None,
        prediction_ids: tuple[str, ...] | None = None,
        row_order: str = "input",
        test_features: np.ndarray | None = None,
    ) -> None:
        self._labels = tuple(hidden_labels)
        if not self._labels:
            raise ValueError("hidden labels must be non-empty")
        if len(set(self._labels)) < MIN_DISTINCT_LABELS:
            raise ValueError(
                "hidden labels must contain at least two distinct classes"
            )
        if expected_count is not None and expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if expected_count is not None and expected_count != len(self._labels):
            raise ValueError("expected_count must match hidden labels size")
        if row_order not in ("input", "id"):
            raise ValueError("row_order must be 'input' or 'id'")
        self._dataset_name = dataset_name
        self._row_order = row_order
        self._id_column = id_column
        self._expected_count = (
            len(self._labels) if expected_count is None else expected_count
        )
        if test_features is not None:
            features = np.asarray(test_features, dtype=np.float64)
            if features.ndim != 2 or features.shape[0] != self._expected_count:
                raise ValueError(
                    "test_features must be a (n_samples, n_features) matrix "
                    "matching the expected count"
                )
            if not np.isfinite(features).all():
                raise ValueError("test_features must be finite")
            self._test_features = features
        else:
            self._test_features = None
        if row_order == "id":
            if id_column is None:
                raise ValueError("id_column is required when row_order='id'")
            if prediction_ids is None:
                raise ValueError(
                    "prediction_ids are required when row_order='id'"
                )
            if len(prediction_ids) != self._expected_count:
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
        return f"clustering:{self._dataset_name}"

    @property
    def expected_count(self) -> int:
        return self._expected_count

    @property
    def hidden_labels(self) -> tuple[str, ...]:
        """Host-only accessor; verifier uses this inside the host boundary."""
        return self._labels

    @property
    def test_features(self) -> np.ndarray | None:
        return self._test_features

    @property
    def id_column(self) -> str | None:
        return self._id_column

    @property
    def row_order(self) -> str:
        return self._row_order

    @property
    def prediction_ids(self) -> tuple[str, ...] | None:
        return self._prediction_ids

    def fingerprint(self) -> str:
        """One-way digest of hidden labels (+ ids/features when present)."""
        labels_digest = hashlib.sha256(
            json.dumps(
                list(self._labels),
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        prediction_ids_sha256 = None
        if self._prediction_ids is not None:
            canonical = json.dumps(
                list(self._prediction_ids),
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            prediction_ids_sha256 = hashlib.sha256(canonical).hexdigest()
        features_sha256 = None
        if self._test_features is not None:
            features_sha256 = hashlib.sha256(
                self._test_features.tobytes()
            ).hexdigest()
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "count": self._expected_count,
                "id_column": self._id_column,
                "row_order": self._row_order,
                "prediction_ids_sha256": prediction_ids_sha256,
                "features_sha256": features_sha256,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(
            payload + labels_digest.encode("utf-8")
        ).hexdigest()
