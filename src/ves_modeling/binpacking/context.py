"""Host-owned bin packing verification context (public problem)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.binpacking.data_contract import BinDataContract


class BinVerificationContext(VerificationContext):
    """Holds the canonical (public) bin packing problem."""

    def __init__(
        self,
        contract: BinDataContract,
        *,
        dataset_name: str = "binpacking",
    ) -> None:
        if contract.n_items < 1:
            raise ValueError("at least one item is required")
        if not math.isfinite(contract.capacity) or contract.capacity <= 0.0:
            raise ValueError("capacity must be a finite positive number")
        for size in contract.items:
            if not math.isfinite(size) or size <= 0.0 or size > contract.capacity:
                raise ValueError("item sizes must be finite, positive and <= capacity")
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"binpacking:{self._dataset_name}"

    @property
    def contract(self) -> BinDataContract:
        return self._contract

    def fingerprint(self) -> str:
        """One-way digest of the canonical problem."""
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "problem": self._contract.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
