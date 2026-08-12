"""Host-owned recommendation verification context (hidden ratings stay host)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from ves.context import VerificationContext


class RecommendationVerificationContext(VerificationContext):
    """Holds hidden ratings plus expected test keys.

    Invariant (key mode): ``user_keys`` / ``item_keys`` are required and
    must have the same length as the hidden ratings; input mode forbids both.
    """

    def __init__(
        self,
        hidden_ratings: np.ndarray,
        *,
        dataset_name: str = "recommendation",
        expected_count: int | None = None,
        user_keys: tuple[str, ...] | None = None,
        item_keys: tuple[str, ...] | None = None,
        user_id_column: str = "user_id",
        item_id_column: str = "item_id",
        row_order: str = "key",
    ) -> None:
        self._ratings = np.asarray(
            hidden_ratings, dtype=np.float64
        ).reshape(-1)
        if self._ratings.size == 0:
            raise ValueError("hidden ratings must be non-empty")
        if not np.isfinite(self._ratings).all():
            raise ValueError("hidden ratings must be finite")
        if expected_count is not None and expected_count <= 0:
            raise ValueError("expected_count must be positive")
        if expected_count is not None and expected_count != self._ratings.size:
            raise ValueError("expected_count must match hidden ratings size")
        if row_order not in ("input", "key"):
            raise ValueError("row_order must be 'input' or 'key'")
        self._dataset_name = dataset_name
        self._row_order = row_order
        self._user_id_column = user_id_column
        self._item_id_column = item_id_column
        self._expected_count = (
            int(self._ratings.size)
            if expected_count is None
            else expected_count
        )
        if row_order == "key":
            if user_keys is None or item_keys is None:
                raise ValueError(
                    "user_keys and item_keys are required when "
                    "row_order='key'"
                )
            if len(user_keys) != self._ratings.size:
                raise ValueError("user_keys must match hidden ratings size")
            if len(item_keys) != self._ratings.size:
                raise ValueError("item_keys must match hidden ratings size")
            self._user_keys = tuple(user_keys)
            self._item_keys = tuple(item_keys)
        else:
            if user_keys is not None or item_keys is not None:
                raise ValueError(
                    "user_keys/item_keys are only used when row_order='key'"
                )
            self._user_keys = None
            self._item_keys = None

    @property
    def id(self) -> str:
        return f"recommendation:{self._dataset_name}"

    @property
    def expected_count(self) -> int:
        return self._expected_count

    @property
    def user_keys(self) -> tuple[str, ...] | None:
        return self._user_keys

    @property
    def item_keys(self) -> tuple[str, ...] | None:
        return self._item_keys

    @property
    def user_id_column(self) -> str:
        return self._user_id_column

    @property
    def item_id_column(self) -> str:
        return self._item_id_column

    @property
    def row_order(self) -> str:
        return self._row_order

    def hidden_ratings(self) -> np.ndarray:
        """Host-only accessor; verifier uses this inside the host boundary."""
        return self._ratings

    def fingerprint(self) -> str:
        """One-way digest of hidden ratings + keys (reversible summaries
        forbidden)."""
        digest = hashlib.sha256(self._ratings.tobytes()).hexdigest()
        keys_sha256 = None
        if self._user_keys is not None:
            canonical = json.dumps(
                {
                    "user": list(self._user_keys),
                    "item": list(self._item_keys),
                },
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            keys_sha256 = hashlib.sha256(canonical).hexdigest()
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "count": self._expected_count,
                "row_order": self._row_order,
                "keys_sha256": keys_sha256,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload + digest.encode("utf-8")).hexdigest()
