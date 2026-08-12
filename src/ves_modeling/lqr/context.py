"""Host-owned LQR verification context (public instance only)."""

from __future__ import annotations

import hashlib
import json

from ves.context import VerificationContext

from ves_modeling.lqr.data_contract import LqrDataContract


class LqrVerificationContext(VerificationContext):
    """Holds the canonical (public) LQR instance."""

    def __init__(
        self,
        contract: LqrDataContract,
        *,
        dataset_name: str = "lqr",
    ) -> None:
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"lqr:{self._dataset_name}"

    @property
    def contract(self) -> LqrDataContract:
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
        matrices = b"".join(
            matrix.tobytes()
            for matrix in (
                self._contract.A,
                self._contract.B,
                self._contract.Q,
                self._contract.R,
                self._contract.QN,
                self._contract.x0,
            )
        )
        return hashlib.sha256(
            payload + hashlib.sha256(matrices).hexdigest().encode("utf-8")
        ).hexdigest()
