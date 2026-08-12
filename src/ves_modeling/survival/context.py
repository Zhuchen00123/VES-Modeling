"""Host-owned survival verification context (outcomes stay host)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from ves.context import VerificationContext


class SurvivalVerificationContext(VerificationContext):
    """Holds hidden test times/events plus expected count.

    Invariant: times finite positive; events are 0/1 with at least one
    event; expected_count matches; id mode requires matching prediction ids.
    """

    def __init__(
        self,
        hidden_times: np.ndarray,
        hidden_events: np.ndarray,
        *,
        dataset_name: str = "survival",
        expected_count: int | None = None,
        output_kind: str = "risk_score",
        id_column: str | None = None,
        prediction_ids: tuple[str, ...] | None = None,
        row_order: str = "input",
    ) -> None:
        self._times = np.asarray(hidden_times, dtype=np.float64).reshape(-1)
        self._events = np.asarray(hidden_events, dtype=np.int64).reshape(-1)
        if self._times.size == 0:
            raise ValueError("hidden outcomes must be non-empty")
        if self._times.size != self._events.size:
            raise ValueError("times and events must have the same length")
        if not np.isfinite(self._times).all() or (self._times <= 0).any():
            raise ValueError("hidden times must be finite and positive")
        if not set(self._events.tolist()) <= {0, 1}:
            raise ValueError("hidden events must be 0/1")
        if int(np.sum(self._events)) < 1:
            raise ValueError("hidden outcomes must contain at least one event")
        if expected_count is not None and expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if expected_count is not None and expected_count != self._times.size:
            raise ValueError("expected_count must match hidden outcomes size")
        if output_kind not in ("risk_score", "time"):
            raise ValueError("output_kind must be 'risk_score' or 'time'")
        if row_order not in ("input", "id"):
            raise ValueError("row_order must be 'input' or 'id'")
        self._dataset_name = dataset_name
        self._output_kind = output_kind
        self._row_order = row_order
        self._id_column = id_column
        self._expected_count = (
            int(self._times.size) if expected_count is None else expected_count
        )
        if row_order == "id":
            if id_column is None:
                raise ValueError("id_column is required when row_order='id'")
            if prediction_ids is None:
                raise ValueError(
                    "prediction_ids are required when row_order='id'"
                )
            if len(prediction_ids) != self._expected_count:
                raise ValueError(
                    "prediction_ids must match hidden outcomes size"
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
        return f"survival:{self._dataset_name}"

    @property
    def expected_count(self) -> int:
        return self._expected_count

    @property
    def output_kind(self) -> str:
        return self._output_kind

    @property
    def row_order(self) -> str:
        return self._row_order

    @property
    def id_column(self) -> str | None:
        return self._id_column

    @property
    def prediction_ids(self) -> tuple[str, ...] | None:
        return self._prediction_ids

    def hidden_times(self) -> np.ndarray:
        """Host-only accessor."""
        return self._times

    def hidden_events(self) -> np.ndarray:
        """Host-only accessor."""
        return self._events

    def fingerprint(self) -> str:
        """One-way digest of hidden outcomes + config."""
        digest = hashlib.sha256(
            self._times.tobytes() + self._events.tobytes()
        ).hexdigest()
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "count": self._expected_count,
                "output_kind": self._output_kind,
                "row_order": self._row_order,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload + digest.encode("utf-8")).hexdigest()
