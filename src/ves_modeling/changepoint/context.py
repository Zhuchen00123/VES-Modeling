"""Host-owned change-point verification context (truth stays host)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from ves.context import VerificationContext


class ChangepointVerificationContext(VerificationContext):
    """Holds hidden true change points plus test-series length.

    Invariants: true points are non-empty, strictly increasing and lie in
    [1, n-2]; ``tolerance_window`` is a non-negative integer.
    """

    def __init__(
        self,
        hidden_changepoints: np.ndarray,
        *,
        dataset_name: str = "changepoint",
        n: int | None = None,
        tolerance_window: int = 3,
    ) -> None:
        self._true = np.asarray(
            hidden_changepoints, dtype=np.int64
        ).reshape(-1)
        if self._true.size == 0:
            raise ValueError("hidden changepoints must be non-empty")
        if isinstance(tolerance_window, bool) or not isinstance(
            tolerance_window, int
        ):
            raise ValueError("tolerance_window must be an integer")
        if tolerance_window < 0:
            raise ValueError("tolerance_window must be >= 0")
        if n is not None:
            if isinstance(n, bool) or not isinstance(n, int):
                raise ValueError("n must be an integer")
            if n < 3:
                raise ValueError("n must be at least 3")
            if (self._true < 1).any() or (self._true > n - 2).any():
                raise ValueError("hidden changepoints must lie in [1, n-2]")
        if (np.diff(self._true) <= 0).any():
            raise ValueError(
                "hidden changepoints must be strictly increasing"
            )
        self._dataset_name = dataset_name
        self._n = n
        self._tolerance_window = tolerance_window

    @property
    def id(self) -> str:
        return f"changepoint:{self._dataset_name}"

    @property
    def n(self) -> int | None:
        return self._n

    @property
    def tolerance_window(self) -> int:
        return self._tolerance_window

    def hidden_changepoints(self) -> np.ndarray:
        """Host-only accessor."""
        return self._true

    def fingerprint(self) -> str:
        """One-way digest of hidden truth + configuration."""
        digest = hashlib.sha256(self._true.tobytes()).hexdigest()
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "n": self._n,
                "tolerance_window": self._tolerance_window,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload + digest.encode("utf-8")).hexdigest()
