"""Bounded linear/MILP optimization data contract (R10).

The public instance is a read-only ``problem.json`` (no hidden truth):
version, sense, variables with type and finite bounds, a finite linear
objective (plus optional constant) and finite linear constraints.  The
candidate artifact ``solution.json`` is exactly
``{"variables": {"name": finite_number, ...}}`` with all and only the
declared variables.

This module validates the problem schema (with duplicate-key detection) and
the solution structure; the host metric computation lives in ``verifier.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SENSES = ("minimize", "maximize")
VARIABLE_TYPES = ("continuous", "integer", "binary")
CONSTRAINT_SENSES = ("<=", ">=", "==")

Coefficients = tuple[tuple[str, float], ...]
Constraint = tuple[Coefficients, str, float]


@dataclass(frozen=True)
class OptimizationDataContract:
    """Canonical public optimization problem (never hidden values)."""

    version: int
    sense: str
    variables: tuple[tuple[str, str, float, float], ...]
    objective_coefficients: Coefficients
    objective_constant: float
    constraints: tuple[Constraint, ...]
    tolerance: float

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(name for name, _type, _lower, _upper in self.variables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sense": self.sense,
            "variables": [
                {
                    "name": name,
                    "type": variable_type,
                    "lower": lower,
                    "upper": upper,
                }
                for name, variable_type, lower, upper in self.variables
            ],
            "objective": {
                "coefficients": dict(self.objective_coefficients),
                "constant": self.objective_constant,
            },
            "constraints": [
                {
                    "coefficients": dict(coefficients),
                    "sense": sense,
                    "rhs": rhs,
                }
                for coefficients, sense, rhs in self.constraints
            ],
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


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object root with duplicate-key rejection."""
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


def _validate_coefficient_map(
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
            (name, _finite_number(coefficient, f"{what} coefficient {name!r}"))
        )
    return tuple(coefficients)


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
        coefficients = _validate_coefficient_map(
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


def validate_optimization_data(
    public_dir: Path,
    *,
    tolerance: float = 1e-6,
) -> OptimizationDataContract:
    """Validate the public problem.json and return the canonical contract."""
    tolerance_value = _finite_number(tolerance, "tolerance")
    if tolerance_value <= 0.0:
        raise ValueError("tolerance must be positive")
    problem = load_json_object(Path(public_dir) / "problem.json")
    unknown_top = set(problem) - {
        "version",
        "sense",
        "variables",
        "objective",
        "constraints",
    }
    if unknown_top:
        raise ValueError(
            f"problem.json has unknown top-level fields: {sorted(unknown_top)}"
        )
    if "version" not in problem:
        raise ValueError("problem.json is missing 'version'")
    version = problem["version"]
    if isinstance(version, (bool, np.bool_)) or not isinstance(
        version, int
    ):
        raise ValueError("problem.json 'version' must be an integer")
    if version < 1:
        raise ValueError("problem.json 'version' must be >= 1")
    if "sense" not in problem:
        raise ValueError("problem.json is missing 'sense'")
    sense = problem["sense"]
    if sense not in SENSES:
        raise ValueError(f"problem.json 'sense' must be one of {SENSES}")
    if "variables" not in problem:
        raise ValueError("problem.json is missing 'variables'")
    variables = _validate_variables(problem["variables"])
    declared = {name for name, _type, _lower, _upper in variables}
    if "objective" not in problem:
        raise ValueError("problem.json is missing 'objective'")
    objective_raw = problem["objective"]
    if not isinstance(objective_raw, dict):
        raise ValueError("'objective' must be an object")
    unknown_objective = set(objective_raw) - {"coefficients", "constant"}
    if unknown_objective:
        raise ValueError(
            f"'objective' has unknown fields: {sorted(unknown_objective)}"
        )
    if "coefficients" not in objective_raw:
        raise ValueError("'objective' is missing 'coefficients'")
    objective_coefficients = _validate_coefficient_map(
        objective_raw["coefficients"],
        declared=declared,
        what="objective",
    )
    objective_constant = (
        _finite_number(objective_raw["constant"], "objective constant")
        if "constant" in objective_raw
        else 0.0
    )
    constraints = _validate_constraints(
        problem.get("constraints", []),
        declared=declared,
    )
    return OptimizationDataContract(
        version=version,
        sense=sense,
        variables=variables,
        objective_coefficients=objective_coefficients,
        objective_constant=objective_constant,
        constraints=constraints,
        tolerance=tolerance_value,
    )


def validate_solution(
    payload: dict[str, Any],
    contract: OptimizationDataContract,
) -> dict[str, float]:
    """Validate a solution artifact; returns variable -> value in order.

    The artifact must be exactly ``{"variables": {...}}`` with all and only
    the declared variables and finite numeric values.  Feasibility against
    bounds/constraints/integrality is computed by the host verifier.
    """
    if "variables" not in payload:
        raise ValueError("missing required field 'variables'")
    raw = payload["variables"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, dict):
        raise ValueError("'variables' must be a JSON object")
    declared = contract.variable_names
    expected = set(declared)
    actual = set(raw)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "solution variables must match declared variables exactly "
            f"(missing={missing}, extra={extra})"
        )
    values: dict[str, float] = {}
    for name in declared:
        value = _finite_number(raw[name], f"solution variable {name!r}")
        values[name] = value
    return values
