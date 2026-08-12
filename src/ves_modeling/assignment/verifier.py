"""Host-computed assignment/TSP facts; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.assignment.context import AssignVerificationContext
from ves_modeling.assignment.data_contract import validate_solution


class AssignVerifier:
    """EvidenceVerifier for assignment/TSP solution artifacts.

    The host recomputes the total cost from the public problem after
    structural validation; candidate self-reported costs are never read and
    no global-optimality fact is produced.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, AssignVerificationContext):
            raise TypeError("AssignVerifier requires AssignVerificationContext")
        payload = self._parse(raw_artifact)
        permutation = validate_solution(payload, context.contract)
        costs = context.contract.costs
        if context.contract.problem_type == "assignment":
            total_cost = sum(
                costs[i][permutation[i]] for i in range(context.contract.size)
            )
        else:
            total_cost = sum(
                costs[permutation[i]][permutation[(i + 1) % len(permutation)]]
                for i in range(len(permutation))
            )
        total_cost = float(total_cost)
        if not np.isfinite(total_cost):
            raise ValueError("total_cost must be finite")
        return Evidence(
            observations=(
                Observation(
                    value=total_cost,
                    uncertainty=0.0,
                    provenance="host:problem",
                    name="total_cost",
                ),
            )
        )

    @staticmethod
    def _parse(raw_artifact: RawArtifact) -> dict[str, Any]:
        text = (
            raw_artifact.content.decode("utf-8")
            if isinstance(raw_artifact.content, bytes)
            else raw_artifact.content
        )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ValueError("solution.json root must be an object")
        return data
