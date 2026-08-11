"""Host-owned optimization verification context (problem + tolerance)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.optimization.data_contract import (
    OptimizationDataContract,
)


class OptimizationVerificationContext(VerificationContext):
    """Holds the canonical (public) problem plus the feasibility tolerance.

    ``problem.json`` is the complete instance (no hidden truth), but it is
    still only exposed to records through ``id`` and a one-way
    ``fingerprint()``; candidate containers mount it read-only.
    """

    def __init__(
        self,
        contract: OptimizationDataContract,
        *,
        dataset_name: str = "optimization",
    ) -> None:
        if not contract.variables:
            raise ValueError("at least one variable is required")
        if contract.sense not in ("minimize", "maximize"):
            raise ValueError("sense must be 'minimize' or 'maximize'")
        if contract.tolerance <= 0.0 or not math.isfinite(contract.tolerance):
            raise ValueError("tolerance must be a finite positive number")
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"optimization:{self._dataset_name}"

    @property
    def contract(self) -> OptimizationDataContract:
        return self._contract

    @property
    def tolerance(self) -> float:
        return self._contract.tolerance

    @property
    def sense(self) -> str:
        return self._contract.sense

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
