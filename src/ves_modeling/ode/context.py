"""Host-owned ODE verification context (hidden values never leave host)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from ves.context import VerificationContext


class OdeVerificationContext(VerificationContext):
    """Holds hidden test values plus expected forecast keys.

    Invariant (key mode): ``trajectory_keys`` / ``time_keys`` are required
    and must have the same length as the hidden values; input mode forbids
    both.
    """

    def __init__(
        self,
        hidden_values: np.ndarray,
        *,
        dataset_name: str = "ode",
        expected_count: int | None = None,
        trajectory_keys: tuple[str, ...] | None = None,
        time_keys: tuple[float, ...] | None = None,
        trajectory_id_column: str = "trajectory_id",
        time_column: str = "t",
        row_order: str = "input",
    ) -> None:
        self._values = np.asarray(hidden_values, dtype=np.float64).reshape(-1)
        if self._values.size == 0:
            raise ValueError("hidden values must be non-empty")
        if not np.isfinite(self._values).all():
            raise ValueError("hidden values must be finite")
        if expected_count is not None and expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if expected_count is not None and expected_count != self._values.size:
            raise ValueError("expected_count must match hidden values size")
        if row_order not in ("input", "key"):
            raise ValueError("row_order must be 'input' or 'key'")
        self._dataset_name = dataset_name
        self._row_order = row_order
        self._trajectory_id_column = trajectory_id_column
        self._time_column = time_column
        self._expected_count = (
            int(self._values.size) if expected_count is None else expected_count
        )
        if row_order == "key":
            if trajectory_keys is None or time_keys is None:
                raise ValueError(
                    "trajectory_keys and time_keys are required when "
                    "row_order='key'"
                )
            if len(trajectory_keys) != self._values.size:
                raise ValueError(
                    "trajectory_keys must match hidden values size"
                )
            if len(time_keys) != self._values.size:
                raise ValueError("time_keys must match hidden values size")
            self._trajectory_keys = tuple(trajectory_keys)
            self._time_keys = tuple(float(value) for value in time_keys)
        else:
            if trajectory_keys is not None or time_keys is not None:
                raise ValueError(
                    "trajectory_keys/time_keys are only used when "
                    "row_order='key'"
                )
            self._trajectory_keys = None
            self._time_keys = None

    @property
    def id(self) -> str:
        return f"ode:{self._dataset_name}"

    @property
    def expected_count(self) -> int:
        return self._expected_count

    @property
    def trajectory_keys(self) -> tuple[str, ...] | None:
        return self._trajectory_keys

    @property
    def time_keys(self) -> tuple[float, ...] | None:
        return self._time_keys

    @property
    def trajectory_id_column(self) -> str:
        return self._trajectory_id_column

    @property
    def time_column(self) -> str:
        return self._time_column

    @property
    def row_order(self) -> str:
        return self._row_order

    def hidden_values(self) -> np.ndarray:
        """Host-only accessor; verifier uses this inside the host boundary."""
        return self._values

    def fingerprint(self) -> str:
        """One-way digest of hidden values + keys (reversible summaries
        forbidden)."""
        digest = hashlib.sha256(self._values.tobytes()).hexdigest()
        keys_sha256 = None
        if self._trajectory_keys is not None:
            canonical = json.dumps(
                {
                    "trajectory": list(self._trajectory_keys),
                    "time": list(self._time_keys),
                },
                separators=(",", ":"),
            ).encode("utf-8")
            keys_sha256 = hashlib.sha256(canonical).hexdigest()
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "count": self._expected_count,
                "row_order": self._row_order,
                "keys_sha256": keys_sha256,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload + digest.encode("utf-8")).hexdigest()
