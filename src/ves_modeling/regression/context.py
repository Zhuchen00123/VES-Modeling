"""Host-owned regression verification context (hidden truth never leaves host)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from ves.context import VerificationContext


class RegressionVerificationContext(VerificationContext):
    """Holds hidden test labels + expected prediction count.

    Only ``id`` and a one-way ``fingerprint()`` are exposed to records;
    labels are never serialized and never mounted into candidate containers.
    """

    def __init__(
        self,
        hidden_labels: np.ndarray,
        *,
        dataset_name: str = "regression",
        expected_count: int | None = None,
    ) -> None:
        self._labels = np.asarray(hidden_labels, dtype=np.float64).reshape(-1)
        if self._labels.size == 0:
            raise ValueError("hidden labels must be non-empty")
        if not np.isfinite(self._labels).all():
            raise ValueError("hidden labels must be finite")
        if expected_count is not None and expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if expected_count is not None and expected_count != self._labels.size:
            raise ValueError("expected_count must match hidden labels size")
        self._dataset_name = dataset_name
        self._expected_count = (
            int(self._labels.size) if expected_count is None else expected_count
        )

    @property
    def id(self) -> str:
        return f"regression:{self._dataset_name}"

    @property
    def expected_count(self) -> int:
        return self._expected_count

    def hidden_labels(self) -> np.ndarray:
        """Host-only accessor; verifier uses this inside the host boundary."""
        return self._labels

    def fingerprint(self) -> str:
        """One-way digest of hidden labels (reversible summaries forbidden)."""
        digest = hashlib.sha256(
            self._labels.tobytes()
        ).hexdigest()
        # Include the declared count so a count mismatch never fingerprints
        # identically to a different dataset slice.
        payload = json.dumps(
            {"dataset": self._dataset_name, "count": self._expected_count},
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload + digest.encode("utf-8")).hexdigest()
