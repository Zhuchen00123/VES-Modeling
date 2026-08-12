"""Host-owned Monte Carlo verification context (reference stays host)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.montecarlo.data_contract import (
    KINDS,
    MonteCarloDataContract,
)


class MonteCarloVerificationContext(VerificationContext):
    """Holds the analytic reference (host-only) and the public problem.

    The reference is deterministic from the public inputs but is never
    exposed to candidates or serialized into records.
    """

    def __init__(
        self,
        reference: float,
        contract: MonteCarloDataContract,
        *,
        dataset_name: str = "montecarlo",
    ) -> None:
        if not math.isfinite(reference):
            raise ValueError("reference must be finite")
        if contract.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self._reference = reference
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"montecarlo:{self._dataset_name}"

    @property
    def contract(self) -> MonteCarloDataContract:
        return self._contract

    @property
    def reference(self) -> float:
        """Host-only accessor; verifier uses this inside the host boundary."""
        return self._reference

    def fingerprint(self) -> str:
        """One-way digest of the public problem + reference."""
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "problem": self._contract.to_dict(),
                "reference": self._reference,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
