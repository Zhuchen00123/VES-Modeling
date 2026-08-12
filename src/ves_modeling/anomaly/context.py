"""Host-owned anomaly verification context (hidden labels stay host)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from ves.context import VerificationContext

from ves_modeling.anomaly.data_contract import OUTPUT_MODES


class AnomalyVerificationContext(VerificationContext):
    """Holds hidden binary labels (1 = anomaly) and the artifact mode.

    Invariant: labels are non-empty with at least one of each class;
    expected_count matches; mode is 'score' or 'label'.
    """

    def __init__(
        self,
        hidden_labels: np.ndarray,
        *,
        dataset_name: str = "anomaly",
        expected_count: int | None = None,
        output_mode: str = "score",
    ) -> None:
        self._labels = np.asarray(hidden_labels, dtype=np.int64).reshape(-1)
        if self._labels.size == 0:
            raise ValueError("hidden labels must be non-empty")
        if set(self._labels.tolist()) != {0, 1}:
            raise ValueError(
                "hidden labels must contain at least one of each class"
            )
        if expected_count is not None and expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if expected_count is not None and expected_count != self._labels.size:
            raise ValueError("expected_count must match hidden labels size")
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {OUTPUT_MODES}")
        self._dataset_name = dataset_name
        self._output_mode = output_mode
        self._expected_count = (
            int(self._labels.size) if expected_count is None else expected_count
        )

    @property
    def id(self) -> str:
        return f"anomaly:{self._dataset_name}"

    @property
    def expected_count(self) -> int:
        return self._expected_count

    @property
    def output_mode(self) -> str:
        return self._output_mode

    def hidden_labels(self) -> np.ndarray:
        """Host-only accessor; verifier uses this inside the host boundary."""
        return self._labels

    def fingerprint(self) -> str:
        """One-way digest of hidden labels + mode (reversible summaries
        forbidden)."""
        digest = hashlib.sha256(self._labels.tobytes()).hexdigest()
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "count": self._expected_count,
                "output_mode": self._output_mode,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload + digest.encode("utf-8")).hexdigest()
