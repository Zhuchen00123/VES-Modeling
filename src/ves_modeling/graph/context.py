"""Host-owned graph verification context (public problem + tolerance)."""

from __future__ import annotations

import hashlib
import json
import math

from ves.context import VerificationContext

from ves_modeling.graph.data_contract import (
    PROBLEM_TYPES,
    GraphDataContract,
)


class GraphVerificationContext(VerificationContext):
    """Holds the canonical (public) graph problem plus tolerance.

    ``problem.json`` is the complete instance (no hidden truth); records only
    see ``id`` and a one-way ``fingerprint()``.
    """

    def __init__(
        self,
        contract: GraphDataContract,
        *,
        dataset_name: str = "graph",
    ) -> None:
        if contract.problem_type not in PROBLEM_TYPES:
            raise ValueError(
                f"problem_type must be one of {PROBLEM_TYPES}"
            )
        if contract.n_nodes < 2:
            raise ValueError("at least two nodes are required")
        if contract.tolerance <= 0.0 or not math.isfinite(
            contract.tolerance
        ):
            raise ValueError("tolerance must be a finite positive number")
        self._contract = contract
        self._dataset_name = dataset_name

    @property
    def id(self) -> str:
        return f"graph:{self._dataset_name}"

    @property
    def contract(self) -> GraphDataContract:
        return self._contract

    @property
    def tolerance(self) -> float:
        return self._contract.tolerance

    @property
    def problem_type(self) -> str:
        return self._contract.problem_type

    def fingerprint(self) -> str:
        """One-way digest of the canonical problem + tolerance."""
        payload = json.dumps(
            {
                "dataset": self._dataset_name,
                "problem": self._contract.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
