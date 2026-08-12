"""Host-owned SIR verification context (reference stays host)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.sir.data_contract import SirDataContract


class SirVerificationContext(VerificationContext):
    """Holds the canonical SIR instance plus the host reference value."""

    def __init__(
        self,
        contract: SirDataContract,
        reference: float,
        *,
        dataset_name: str = "sir",
    ) -> None:
        if not math.isfinite(reference):
            raise ValueError("reference must be finite")
        self._contract = contract
        self._reference = float(reference)
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"sir:{self._dataset_name}"

    @property
    def contract(self) -> SirDataContract:
        return self._contract

    def reference_value(self) -> float:
        """Host-only accessor for the numerical SIR reference."""
        return self._reference

    def fingerprint(self) -> str:
        """One-way digest of public problem + host reference."""
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
