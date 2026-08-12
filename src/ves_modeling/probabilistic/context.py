"""Host-owned probabilistic verification context (parameters stay host)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.probabilistic.data_contract import (
    FAMILIES,
    QUANTITIES,
    ProbabilisticDataContract,
)


class ProbabilisticVerificationContext(VerificationContext):
    """Holds the exact reference (host-only) and the public problem."""

    def __init__(
        self,
        reference: float,
        contract: ProbabilisticDataContract,
        *,
        dataset_name: str = "probabilistic",
    ) -> None:
        if not math.isfinite(reference):
            raise ValueError("reference must be finite")
        if contract.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if contract.quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        self._reference = reference
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"probabilistic:{self._dataset_name}"

    @property
    def contract(self) -> ProbabilisticDataContract:
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
