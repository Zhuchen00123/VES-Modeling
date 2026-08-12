"""Host-owned assignment/TSP verification context (public problem)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.assignment.data_contract import (
    MIN_SIZE,
    PROBLEM_TYPES,
    AssignDataContract,
)


class AssignVerificationContext(VerificationContext):
    """Holds the canonical (public) assignment/TSP problem."""

    def __init__(
        self,
        contract: AssignDataContract,
        *,
        dataset_name: str = "assignment",
    ) -> None:
        if contract.problem_type not in PROBLEM_TYPES:
            raise ValueError(
                f"problem_type must be one of {PROBLEM_TYPES}"
            )
        if contract.size < MIN_SIZE:
            raise ValueError(f"size must be >= {MIN_SIZE}")
        if len(contract.costs) != contract.size:
            raise ValueError("costs must be a size x size matrix")
        if not (0 <= contract.start < contract.size):
            raise ValueError("start must be within [0, size)")
        for row in contract.costs:
            if len(row) != contract.size:
                raise ValueError("costs rows must have exactly size entries")
            for value in row:
                if not math.isfinite(value):
                    raise ValueError("costs must be finite")
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"assignment:{self._dataset_name}"

    @property
    def contract(self) -> AssignDataContract:
        return self._contract

    @property
    def problem_type(self) -> str:
        return self._contract.problem_type

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
