"""Host-owned bi-objective verification context (public problem + tolerance)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.multiobjective.data_contract import MooDataContract


class MooVerificationContext(VerificationContext):
    """Holds the canonical (public) problem plus tolerance.

    Records only see ``id`` and a one-way ``fingerprint()``.
    """

    def __init__(
        self,
        contract: MooDataContract,
        *,
        dataset_name: str = "multiobjective",
    ) -> None:
        if contract.n_variables < 1:
            raise ValueError("at least one variable is required")
        if len(contract.objectives) != 2:
            raise ValueError("exactly two objectives are required")
        if contract.tolerance <= 0.0 or not math.isfinite(
            contract.tolerance
        ):
            raise ValueError("tolerance must be a finite positive number")
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"multiobjective:{self._dataset_name}"

    @property
    def contract(self) -> MooDataContract:
        return self._contract

    @property
    def tolerance(self) -> float:
        return self._contract.tolerance

    def fingerprint(self) -> str:
        """One-way digest of the canonical problem + tolerance."""
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "problem": self._contract.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
