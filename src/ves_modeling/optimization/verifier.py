"""Host-computed optimization facts; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.optimization.context import OptimizationVerificationContext
from ves_modeling.optimization.data_contract import validate_solution


class OptimizationVerifier:
    """EvidenceVerifier for ``solution.json`` artifacts.

    The host recomputes the maximum bound residual, maximum constraint
    residual, maximum integrality violation and the objective value from the
    public problem.  Candidate-reported objective/feasibility/optimality/gap
    fields are never read; no global-optimality fact is ever produced.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, OptimizationVerificationContext):
            raise TypeError(
                "OptimizationVerifier requires OptimizationVerificationContext"
            )
        payload = self._parse(raw_artifact)
        values = validate_solution(payload, context.contract)
        max_bound = _max_bound_violation(values, context.contract)
        max_constraint = _max_constraint_violation(values, context.contract)
        max_integrality = _max_integrality_violation(values, context.contract)
        objective = _objective_value(values, context.contract)
        metrics = (max_bound, max_constraint, max_integrality, objective)
        for value in metrics:
            if not np.isfinite(value):
                raise ValueError("optimization metrics must be finite")
        return Evidence(
            observations=(
                Observation(
                    value=max_bound,
                    uncertainty=0.0,
                    provenance="host:problem",
                    name="max_bound_violation",
                ),
                Observation(
                    value=max_constraint,
                    uncertainty=0.0,
                    provenance="host:problem",
                    name="max_constraint_violation",
                ),
                Observation(
                    value=max_integrality,
                    uncertainty=0.0,
                    provenance="host:problem",
                    name="integrality_violation",
                ),
                Observation(
                    value=objective,
                    uncertainty=0.0,
                    provenance="host:problem",
                    name="objective",
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


def _max_bound_violation(
    values: dict[str, float], contract
) -> float:
    worst = 0.0
    for name, _type, lower, upper in contract.variables:
        value = values[name]
        worst = max(worst, max(0.0, lower - value, value - upper))
    return float(worst)


def _max_constraint_violation(
    values: dict[str, float], contract
) -> float:
    worst = 0.0
    for coefficients, sense, rhs in contract.constraints:
        lhs = sum(coefficient * values[name] for name, coefficient in coefficients)
        if sense == "<=":
            residual = max(0.0, lhs - rhs)
        elif sense == ">=":
            residual = max(0.0, rhs - lhs)
        else:
            residual = abs(lhs - rhs)
        worst = max(worst, residual)
    return float(worst)


def _max_integrality_violation(
    values: dict[str, float], contract
) -> float:
    worst = 0.0
    for name, variable_type, _lower, _upper in contract.variables:
        if variable_type == "continuous":
            continue
        value = values[name]
        worst = max(worst, abs(value - round(value)))
    return float(worst)


def _objective_value(values: dict[str, float], contract) -> float:
    objective = contract.objective_constant
    for name, coefficient in contract.objective_coefficients:
        objective += coefficient * values[name]
    return float(objective)
