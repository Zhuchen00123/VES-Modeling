"""Host-owned forecasting verification context (hidden truth never leaves host)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from ves.context import VerificationContext


class ForecastingVerificationContext(VerificationContext):
    """Holds hidden test labels plus expected forecast keys.

    Only ``id`` and a one-way ``fingerprint()`` are exposed to records;
    labels and key tuples are never serialized and never mounted into
    candidate containers.

    Invariant (key mode): ``series_keys`` / ``time_keys`` are required and
    must have the same length as the hidden labels; input mode forbids both.
    """

    def __init__(
        self,
        hidden_labels: np.ndarray,
        *,
        dataset_name: str = "forecasting",
        expected_count: int | None = None,
        series_keys: tuple[str, ...] | None = None,
        time_keys: tuple[str, ...] | None = None,
        series_id_column: str = "series_id",
        time_column: str = "timestamp",
        frequency: str = "D",
        row_order: str = "key",
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
        if row_order not in ("input", "key"):
            raise ValueError("row_order must be 'input' or 'key'")
        self._dataset_name = dataset_name
        self._frequency = frequency
        self._row_order = row_order
        self._series_id_column = series_id_column
        self._time_column = time_column
        self._expected_count = (
            int(self._labels.size) if expected_count is None else expected_count
        )
        if row_order == "key":
            if series_keys is None or time_keys is None:
                raise ValueError(
                    "series_keys and time_keys are required when "
                    "row_order='key'"
                )
            if len(series_keys) != self._labels.size:
                raise ValueError("series_keys must match hidden labels size")
            if len(time_keys) != self._labels.size:
                raise ValueError("time_keys must match hidden labels size")
            if not series_id_column or not time_column:
                raise ValueError(
                    "series_id_column and time_column are required when "
                    "row_order='key'"
                )
            self._series_keys = tuple(series_keys)
            self._time_keys = tuple(time_keys)
        else:
            if series_keys is not None or time_keys is not None:
                raise ValueError(
                    "series_keys/time_keys are only used when row_order='key'"
                )
            self._series_keys = None
            self._time_keys = None

    @property
    def id(self) -> str:
        return f"forecasting:{self._dataset_name}"

    @property
    def expected_count(self) -> int:
        return self._expected_count

    @property
    def series_keys(self) -> tuple[str, ...] | None:
        return self._series_keys

    @property
    def time_keys(self) -> tuple[str, ...] | None:
        return self._time_keys

    @property
    def series_id_column(self) -> str:
        return self._series_id_column

    @property
    def time_column(self) -> str:
        return self._time_column

    @property
    def frequency(self) -> str:
        return self._frequency

    @property
    def row_order(self) -> str:
        return self._row_order

    def hidden_labels(self) -> np.ndarray:
        """Host-only accessor; verifier uses this inside the host boundary."""
        return self._labels

    def fingerprint(self) -> str:
        """One-way digest of hidden labels + forecast keys (reversible
        summaries forbidden)."""
        digest = hashlib.sha256(self._labels.tobytes()).hexdigest()
        keys_sha256 = None
        if self._series_keys is not None:
            canonical = json.dumps(
                {
                    "series": list(self._series_keys),
                    "time": list(self._time_keys),
                },
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            keys_sha256 = hashlib.sha256(canonical).hexdigest()
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "count": self._expected_count,
                "frequency": self._frequency,
                "row_order": self._row_order,
                "keys_sha256": keys_sha256,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload + digest.encode("utf-8")).hexdigest()
