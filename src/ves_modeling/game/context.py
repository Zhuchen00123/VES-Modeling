"""Host-owned game verification context (public instance only)."""

from __future__ import annotations

import hashlib
import json

from ves.context import VerificationContext

from ves_modeling.game.data_contract import GameDataContract


class GameVerificationContext(VerificationContext):
    """Holds the canonical (public) LQ game instance."""

    def __init__(
        self,
        contract: GameDataContract,
        *,
        dataset_name: str = "game",
    ) -> None:
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"game:{self._dataset_name}"

    @property
    def contract(self) -> GameDataContract:
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
                self._contract.C,
                self._contract.Q,
                self._contract.R,
                self._contract.S,
                self._contract.QN,
                self._contract.x0,
            )
        )
        return hashlib.sha256(
            payload + hashlib.sha256(matrices).hexdigest().encode("utf-8")
        ).hexdigest()
