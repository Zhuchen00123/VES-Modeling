"""Host-owned sequential-pattern context (truth stays host)."""

from __future__ import annotations

import hashlib
import json

from ves.context import VerificationContext


class SeqPatternVerificationContext(VerificationContext):
    """Holds hidden test sequences plus the train event set."""

    def __init__(
        self,
        hidden_sequences: tuple[tuple[str, ...], ...],
        *,
        event_set: tuple[str, ...],
        dataset_name: str = "seqpattern",
    ) -> None:
        if not hidden_sequences:
            raise ValueError("hidden sequences must be non-empty")
        self._hidden = tuple(
            tuple(sequence) for sequence in hidden_sequences
        )
        self._event_set = tuple(event_set)
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"seqpattern:{self._dataset_name}"

    @property
    def event_set(self) -> tuple[str, ...]:
        return self._event_set

    def hidden_sequences(self) -> tuple[tuple[str, ...], ...]:
        """Host-only accessor."""
        return self._hidden

    def fingerprint(self) -> str:
        """One-way digest of hidden truth + configuration."""
        digest = hashlib.sha256(
            json.dumps(self._hidden, sort_keys=True).encode("utf-8")
        ).hexdigest()
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "event_set": list(self._event_set),
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload + digest.encode("utf-8")).hexdigest()
