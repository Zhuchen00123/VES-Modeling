"""Host-owned association verification context (hidden transactions stay host)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.association.data_contract import (
    AssociationDataContract,
)


class AssociationVerificationContext(VerificationContext):
    """Holds hidden test transactions (host-only) and the public contract."""

    def __init__(
        self,
        hidden_transactions: list[frozenset[str]],
        contract: AssociationDataContract,
        *,
        dataset_name: str = "association",
        lift_cap: float = 1e6,
    ) -> None:
        if not hidden_transactions:
            raise ValueError("hidden transactions must be non-empty")
        if contract.n_transactions < 2:
            raise ValueError("at least two train transactions are required")
        if not math.isfinite(lift_cap) or lift_cap <= 0.0:
            raise ValueError("lift_cap must be a finite positive number")
        self._hidden_transactions = tuple(hidden_transactions)
        self._contract = contract
        self._dataset_name = dataset_name
        self._lift_cap = lift_cap

    @property
    def id(self) -> str:
        return f"association:{self._dataset_name}"

    @property
    def contract(self) -> AssociationDataContract:
        return self._contract

    @property
    def lift_cap(self) -> float:
        return self._lift_cap

    def hidden_transactions(self) -> tuple[frozenset[str], ...]:
        """Host-only accessor; verifier uses this inside the host boundary."""
        return self._hidden_transactions

    def fingerprint(self) -> str:
        """One-way digest of the public contract + hidden transactions."""
        canonical = json.dumps(
            {
                "dataset": self._dataset_name,
                "problem": self._contract.to_dict(),
                "hidden": [
                    sorted(transaction) for transaction in self._hidden_transactions
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
