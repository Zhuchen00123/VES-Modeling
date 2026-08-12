"""Host-owned Markov verification context (transition matrix stays host)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.markov.data_contract import (
    QUANTITIES,
    MarkovDataContract,
)


class MarkovVerificationContext(VerificationContext):
    """Holds the exact reference (host-only) and the public problem."""

    def __init__(
        self,
        reference: float,
        contract: MarkovDataContract,
        *,
        dataset_name: str = "markov",
    ) -> None:
        if not math.isfinite(reference):
            raise ValueError("reference must be finite")
        if contract.quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        if contract.n_states < 2:
            raise ValueError("at least two states are required")
        self._reference = reference
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"markov:{self._dataset_name}"

    @property
    def contract(self) -> MarkovDataContract:
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
