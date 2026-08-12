"""Host-owned network-SIR context (reference stays host)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.networksir.data_contract import NetworkSirDataContract


class NetworkSirVerificationContext(VerificationContext):
    """Holds the canonical network instance plus the host reference value."""

    def __init__(
        self,
        contract: NetworkSirDataContract,
        reference: float,
        *,
        dataset_name: str = "networksir",
    ) -> None:
        if not math.isfinite(reference):
            raise ValueError("reference must be finite")
        self._contract = contract
        self._reference = float(reference)
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"networksir:{self._dataset_name}"

    @property
    def contract(self) -> NetworkSirDataContract:
        return self._contract

    def reference_value(self) -> float:
        """Host-only accessor for the averaged network-SIR reference."""
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
