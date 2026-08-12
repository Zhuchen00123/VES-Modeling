"""Host-owned queueing verification context (reference stays host)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.queueing.data_contract import (
    KINDS,
    QUANTITIES,
    QueueingDataContract,
)


class QueueingVerificationContext(VerificationContext):
    """Holds the analytic reference (host-only) and the public problem."""

    def __init__(
        self,
        reference: float,
        contract: QueueingDataContract,
        *,
        dataset_name: str = "queueing",
    ) -> None:
        if not math.isfinite(reference):
            raise ValueError("reference must be finite")
        if contract.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if contract.quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        self._reference = reference
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"queueing:{self._dataset_name}"

    @property
    def contract(self) -> QueueingDataContract:
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
