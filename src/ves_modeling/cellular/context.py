"""Host-owned cellular-automaton context (reference stays host)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.cellular.data_contract import CellularDataContract


class CellularVerificationContext(VerificationContext):
    """Holds the canonical CA instance plus the host reference value."""

    def __init__(
        self,
        contract: CellularDataContract,
        reference: float,
        *,
        dataset_name: str = "cellular",
    ) -> None:
        if not math.isfinite(reference):
            raise ValueError("reference must be finite")
        self._contract = contract
        self._reference = float(reference)
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"cellular:{self._dataset_name}"

    @property
    def contract(self) -> CellularDataContract:
        return self._contract

    def reference_value(self) -> float:
        """Host-only accessor for the exact CA reference."""
        return self._reference

    def fingerprint(self) -> str:
        """One-way digest of public problem + host reference."""
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "problem": self._contract.to_dict(),
                "initial": self._contract.initial.tolist(),
                "reference": self._reference,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
