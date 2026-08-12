"""Host-computed bi-objective facts; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.multiobjective.context import MooVerificationContext
from ves_modeling.multiobjective.data_contract import (
    MooDataContract,
    validate_solution,
)


class MooVerifier:
    """EvidenceVerifier for bi-objective solution sets.

    The host recomputes feasibility, integrality and both objective values
    per solution, keeps feasible solutions, drops dominated ones, and
    reports the exact 2D hypervolume against the reference point plus
    audit counts.  No global-optimality fact is ever produced.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, MooVerificationContext):
            raise TypeError("MooVerifier requires MooVerificationContext")
        payload = self._parse(raw_artifact)
        solutions = validate_solution(payload, context.contract)
        metrics = _compute_metrics(solutions, context.contract)
        for value in metrics.values():
            if not np.isfinite(value):
                raise ValueError("multi-objective metrics must be finite")
        return Evidence(
            observations=tuple(
                Observation(
                    value=value,
                    uncertainty=0.0,
                    provenance="host:problem",
                    name=name,
                )
                for name, value in metrics.items()
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


def _objective_value(values: dict[str, float], objective, contract) -> float:
    value = objective.constant
    for name, coefficient in objective.coefficients:
        value += coefficient * values[name]
    return float(value)


def _solution_violations(
    values: dict[str, float], contract: MooDataContract
) -> tuple[float, float, float]:
    max_bound = 0.0
    for name, _t, lower, upper in contract.variables:
        value = values[name]
        max_bound = max(max_bound, max(0.0, lower - value, value - upper))
    max_constraint = 0.0
    for coefficients, sense, rhs in contract.constraints:
        lhs = sum(
            coefficient * values[name] for name, coefficient in coefficients
        )
        if sense == "<=":
            residual = max(0.0, lhs - rhs)
        elif sense == ">=":
            residual = max(0.0, rhs - lhs)
        else:
            residual = abs(lhs - rhs)
        max_constraint = max(max_constraint, residual)
    max_integrality = 0.0
    for name, variable_type, _l, _u in contract.variables:
        if variable_type == "continuous":
            continue
        max_integrality = max(
            max_integrality, abs(values[name] - round(values[name]))
        )
    return (
        float(max_bound),
        float(max_constraint),
        float(max_integrality),
    )


def _compute_metrics(
    solutions: list[dict[str, float]], contract: MooDataContract
) -> dict[str, float]:
    tolerance = contract.tolerance
    feasible_points: list[tuple[float, float]] = []
    for values in solutions:
        max_bound, max_constraint, max_integrality = _solution_violations(
            values, contract
        )
        feasible = (
            max_bound <= tolerance
            and max_constraint <= tolerance
            and max_integrality <= tolerance
        )
        if not feasible:
            continue
        objective_1 = _objective_value(values, contract.objectives[0], contract)
        objective_2 = _objective_value(values, contract.objectives[1], contract)
        feasible_points.append((objective_1, objective_2))
    non_dominated: list[tuple[float, float]] = []
    for point in feasible_points:
        dominated = False
        for other in feasible_points:
            if other is point:
                continue
            if (
                other[0] >= point[0]
                and other[1] >= point[1]
                and (other[0] > point[0] or other[1] > point[1])
            ):
                dominated = True
                break
        if not dominated:
            non_dominated.append(point)
    hypervolume = (
        _hypervolume(non_dominated, contract.reference_point)
        if non_dominated
        else 0.0
    )
    return {
        "hypervolume": float(hypervolume),
        "non_dominated_count": float(len(non_dominated)),
        "feasible_count": float(len(feasible_points)),
        "total_count": float(len(solutions)),
    }


def _hypervolume(
    points: list[tuple[float, float]], reference: tuple[float, float]
) -> float:
    """Exact 2D quality measure against the reference corner.

    Sort non-dominated points by objective-1 ascending, take the prefix max
    of objective-2, and sum the covered rectangles between consecutive
    frontier x's up to the reference point (the reference is the fixed
    objective upper bound + 1 corner, above-right of every feasible point).
    """
    reference_x, reference_y = reference
    ordered = sorted(points, key=lambda point: point[0])
    area = 0.0
    prefix_max_y = float("-inf")
    for index, (x, y) in enumerate(ordered):
        prefix_max_y = max(prefix_max_y, y)
        next_x = (
            ordered[index + 1][0]
            if index + 1 < len(ordered)
            else reference_x
        )
        area += (next_x - x) * (reference_y - prefix_max_y)
    return float(area)
