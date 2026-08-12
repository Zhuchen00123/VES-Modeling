"""Bi-objective Pareto optimization data contract (R16).

``problem.json`` is the complete public instance (no hidden truth):
version, variables (continuous/integer/binary with finite bounds, same rules
as the optimization slice), exactly two linear objectives, linear constraints
and an optional reference point (defaults to host-computed objective box
upper bounds + 1).

The artifact ``solution.json`` is ``{"solutions": [{"variables": {...}}, ...]}``
with at least one solution (or a single ``{"variables": {...}}`` treated as
one solution); every solution carries all and only the declared variables.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VARIABLE_TYPES = ("continuous", "integer", "binary")
CONSTRAINT_SENSES = ("<=", ">=", "==")

Coefficients = tuple[tuple[str, float], ...]
Constraint = tuple[Coefficients, str, float]


@dataclass(frozen=True)
class MooObjective:
    coefficients: Coefficients
    constant: float = 0.0


@dataclass(frozen=True)
class MooDataContract:
    """Canonical public bi-objective problem."""

    version: int
    variables: tuple[tuple[str, str, float, float], ...] = field(
        repr=False, compare=False
    )
    objectives: tuple[MooObjective, MooObjective] = field(
        repr=False, compare=False
    )
    constraints: tuple[Constraint, ...] = field(repr=False, compare=False)
    reference_point: tuple[float, float] = field(repr=False, compare=False)
    tolerance: float = 1e-6

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(name for name, _t, _l, _u in self.variables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "variables": [
                {
                    "name": name,
                    "type": variable_type,
                    "lower": lower,
                    "upper": upper,
                }
                for name, variable_type, lower, upper in self.variables
            ],
            "objectives": [
                {
                    "coefficients": dict(objective.coefficients),
                    "constant": objective.constant,
                }
                for objective in self.objectives
            ],
            "constraints": [
                {
                    "coefficients": dict(coefficients),
                    "sense": sense,
                    "rhs": rhs,
                }
                for coefficients, sense, rhs in self.constraints
            ],
            "reference_point": list(self.reference_point),
            "tolerance": self.tolerance,
            "n_variables": self.n_variables,
            "n_constraints": self.n_constraints,
        }


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in JSON object: {key!r}")
        result[key] = value
    return result


def load_problem(path: Path) -> dict[str, Any]:
    """Load problem.json with duplicate-key rejection."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from None
    except OSError as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} root must be an object")
    return data


def _finite_number(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float)
    ):
        raise ValueError(f"{what} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite (no NaN/Infinity)")
    return number


def _validate_variables(
    raw: Any,
) -> tuple[tuple[str, str, float, float], ...]:
    if not isinstance(raw, dict):
        raise ValueError("'variables' must be an object")
    if not raw:
        raise ValueError("at least one variable is required")
    variables: list[tuple[str, str, float, float]] = []
    for name, spec in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("variable names must be non-empty strings")
        if not isinstance(spec, dict):
            raise ValueError(f"variable {name!r} must be an object")
        unknown = set(spec) - {"type", "lower", "upper"}
        if unknown:
            raise ValueError(
                f"variable {name!r} has unknown fields: {sorted(unknown)}"
            )
        if "type" not in spec:
            raise ValueError(f"variable {name!r} is missing 'type'")
        variable_type = spec["type"]
        if variable_type not in VARIABLE_TYPES:
            raise ValueError(
                f"variable {name!r} type must be one of {VARIABLE_TYPES}"
            )
        if "lower" not in spec or "upper" not in spec:
            raise ValueError(
                f"variable {name!r} must declare finite lower and upper"
            )
        lower = _finite_number(spec["lower"], f"variable {name!r} lower")
        upper = _finite_number(spec["upper"], f"variable {name!r} upper")
        if lower > upper:
            raise ValueError(
                f"variable {name!r} lower must not exceed upper"
            )
        if variable_type == "binary" and (lower != 0.0 or upper != 1.0):
            raise ValueError(
                f"binary variable {name!r} must have bounds [0, 1]"
            )
        variables.append((name, variable_type, lower, upper))
    return tuple(variables)


def _validate_coefficients(
    raw: Any,
    *,
    declared: set[str],
    what: str,
) -> Coefficients:
    if not isinstance(raw, dict):
        raise ValueError(f"'{what}' coefficients must be an object")
    coefficients: list[tuple[str, float]] = []
    for name, coefficient in raw.items():
        if name not in declared:
            raise ValueError(
                f"{what} references undeclared variable {name!r}"
            )
        coefficients.append(
            (
                name,
                _finite_number(coefficient, f"{what} coefficient {name!r}"),
            )
        )
    return tuple(coefficients)


def _validate_objectives(
    raw: Any,
    *,
    declared: set[str],
) -> tuple[MooObjective, MooObjective]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(
            "exactly two linear objectives are required"
        )
    objectives: list[MooObjective] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"objective {index} must be an object")
        unknown = set(entry) - {"coefficients", "constant"}
        if unknown:
            raise ValueError(
                f"objective {index} has unknown fields: {sorted(unknown)}"
            )
        if "coefficients" not in entry:
            raise ValueError(
                f"objective {index} is missing 'coefficients'"
            )
        coefficients = _validate_coefficients(
            entry["coefficients"],
            declared=declared,
            what=f"objective {index}",
        )
        constant = (
            _finite_number(entry["constant"], f"objective {index} constant")
            if "constant" in entry
            else 0.0
        )
        objectives.append(MooObjective(coefficients, constant))
    return objectives[0], objectives[1]


def _validate_constraints(
    raw: Any,
    *,
    declared: set[str],
) -> tuple[Constraint, ...]:
    if not isinstance(raw, list):
        raise ValueError("'constraints' must be a JSON array")
    constraints: list[Constraint] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"constraint {index} must be an object")
        unknown = set(entry) - {"coefficients", "sense", "rhs"}
        if unknown:
            raise ValueError(
                f"constraint {index} has unknown fields: {sorted(unknown)}"
            )
        if "coefficients" not in entry or "sense" not in entry or "rhs" not in entry:
            raise ValueError(
                f"constraint {index} must contain 'coefficients', 'sense' "
                "and 'rhs'"
            )
        coefficients = _validate_coefficients(
            entry["coefficients"],
            declared=declared,
            what=f"constraint {index}",
        )
        sense = entry["sense"]
        if sense not in CONSTRAINT_SENSES:
            raise ValueError(
                f"constraint {index} sense must be one of "
                f"{CONSTRAINT_SENSES}"
            )
        rhs = _finite_number(entry["rhs"], f"constraint {index} rhs")
        constraints.append((coefficients, sense, rhs))
    return tuple(constraints)


def _default_reference_point(
    variables: tuple[tuple[str, str, float, float], ...],
    objectives: tuple[MooObjective, MooObjective],
) -> tuple[float, float]:
    """Objective box upper bounds + 1 (dominates any feasible point)."""
    upper = {name: upper for name, _t, _l, upper in variables}
    lower = {name: lower for name, _t, lower, _u in variables}
    bounds: list[float] = []
    for objective in objectives:
        value = objective.constant
        for name, coefficient in objective.coefficients:
            if coefficient >= 0.0:
                value += coefficient * upper[name]
            else:
                value += coefficient * lower[name]
        bounds.append(value + 1.0)
    return float(bounds[0]), float(bounds[1])


def validate_moo_data(
    public_dir: Path,
    *,
    tolerance: float = 1e-6,
) -> MooDataContract:
    """Validate the public problem.json and return the canonical contract."""
    tolerance_value = _finite_number(tolerance, "tolerance")
    if tolerance_value <= 0.0:
        raise ValueError("tolerance must be positive")
    problem = load_problem(Path(public_dir) / "problem.json")
    unknown_top = set(problem) - {
        "version",
        "variables",
        "objectives",
        "constraints",
        "reference_point",
    }
    if unknown_top:
        raise ValueError(
            f"problem.json has unknown top-level fields: {sorted(unknown_top)}"
        )
    if "version" not in problem:
        raise ValueError("problem.json is missing 'version'")
    version = problem["version"]
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, int):
        raise ValueError("problem.json 'version' must be an integer")
    if version < 1:
        raise ValueError("problem.json 'version' must be >= 1")
    if "variables" not in problem:
        raise ValueError("problem.json is missing 'variables'")
    variables = _validate_variables(problem["variables"])
    declared = {name for name, _t, _l, _u in variables}
    if "objectives" not in problem:
        raise ValueError("problem.json is missing 'objectives'")
    objectives = _validate_objectives(problem["objectives"], declared=declared)
    constraints = _validate_constraints(
        problem.get("constraints", []), declared=declared
    )
    reference_point: tuple[float, float]
    if "reference_point" in problem:
        raw_reference = problem["reference_point"]
        if (
            not isinstance(raw_reference, list)
            or len(raw_reference) != 2
            or isinstance(raw_reference[0], bool)
            or isinstance(raw_reference[1], bool)
        ):
            raise ValueError(
                "'reference_point' must be a [r1, r2] pair of numbers"
            )
        reference_point = (
            _finite_number(raw_reference[0], "reference_point r1"),
            _finite_number(raw_reference[1], "reference_point r2"),
        )
    else:
        reference_point = _default_reference_point(variables, objectives)
    return MooDataContract(
        version=version,
        variables=variables,
        objectives=objectives,
        constraints=constraints,
        reference_point=reference_point,
        tolerance=tolerance_value,
    )


def _validate_single_solution(
    raw: Any,
    *,
    declared: tuple[str, ...],
) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("solutions must be objects with 'variables'")
    if "variables" not in raw:
        raise ValueError("solution is missing 'variables'")
    variables = raw["variables"]
    if isinstance(variables, (str, bytes)) or not isinstance(variables, dict):
        raise ValueError("'variables' must be a JSON object")
    expected = set(declared)
    actual = set(variables)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "solution variables must match declared variables exactly "
            f"(missing={missing}, extra={extra})"
        )
    values: dict[str, float] = {}
    for name in declared:
        values[name] = _finite_number(variables[name], f"variable {name!r}")
    return values


def validate_solution(
    payload: dict[str, Any], contract: MooDataContract
) -> list[dict[str, float]]:
    """Validate a solution artifact; returns per-solution variable values."""
    declared = contract.variable_names
    if "solutions" in payload:
        raw_solutions = payload["solutions"]
        if isinstance(raw_solutions, (str, bytes)) or not isinstance(
            raw_solutions, list
        ):
            raise ValueError("'solutions' must be a JSON array")
        if not raw_solutions:
            raise ValueError("'solutions' must contain at least one solution")
        return [
            _validate_single_solution(item, declared=declared)
            for item in raw_solutions
        ]
    if "variables" in payload:
        return [_validate_single_solution(payload, declared=declared)]
    raise ValueError(
        "solution.json must contain 'solutions' or a single 'variables'"
    )
