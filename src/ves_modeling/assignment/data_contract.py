"""Assignment / TSP combinatorial optimization data contract (R22).

``problem.json`` is the complete public instance (no hidden truth):
version, problem_type (assignment | tsp), size n >= 3, and a finite cost
matrix (assignment: n x n; tsp: n x n symmetric with zero diagonal), plus an
optional start for tsp (default 0).

Artifact ``solution.json``:
- assignment: ``{"assignment": [j0, ..., j_{n-1}]}`` (a permutation);
- tsp: ``{"tour": [node0, ..., node_{n-1}]}`` (a permutation starting at
  start, representing a cyclic tour).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROBLEM_TYPES = ("assignment", "tsp")
MIN_SIZE = 3


@dataclass(frozen=True)
class AssignDataContract:
    """Canonical public assignment/TSP problem."""

    version: int
    problem_type: str
    size: int
    costs: tuple[tuple[float, ...], ...] = field(repr=False, compare=False)
    start: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "problem_type": self.problem_type,
            "size": self.size,
            "costs": [list(row) for row in self.costs],
            "start": self.start,
        }


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in JSON object: {key!r}")
        result[key] = value
    return result


def _finite_number(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float)
    ):
        raise ValueError(f"{what} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite (no NaN/Infinity)")
    return number


def validate_assign_data(
    public_dir: Path,
) -> AssignDataContract:
    """Validate the public problem.json and return the canonical contract."""
    try:
        with (Path(public_dir) / "problem.json").open(
            encoding="utf-8"
        ) as handle:
            problem = json.load(
                handle, object_pairs_hook=_reject_duplicates
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in problem.json: {exc}") from None
    except OSError as exc:
        raise ValueError(f"cannot read problem.json: {exc}") from None
    if not isinstance(problem, dict):
        raise ValueError("problem.json root must be an object")
    unknown_top = set(problem) - {
        "version",
        "problem_type",
        "size",
        "costs",
        "start",
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
    if "problem_type" not in problem:
        raise ValueError("problem.json is missing 'problem_type'")
    problem_type = problem["problem_type"]
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(
            f"problem_type must be one of {PROBLEM_TYPES}"
        )
    if "size" not in problem:
        raise ValueError("problem.json is missing 'size'")
    size = problem["size"]
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, int):
        raise ValueError("'size' must be an integer")
    if size < MIN_SIZE:
        raise ValueError(f"'size' must be >= {MIN_SIZE}")
    if "costs" not in problem:
        raise ValueError("problem.json is missing 'costs'")
    costs_raw = problem["costs"]
    if not isinstance(costs_raw, list) or len(costs_raw) != size:
        raise ValueError("'costs' must be a size x size matrix")
    costs: list[tuple[float, ...]] = []
    for i, row in enumerate(costs_raw):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError(f"costs row {i} must have exactly {size} entries")
        parsed = tuple(
            _finite_number(value, f"costs[{i}][{j}]")
            for j, value in enumerate(row)
        )
        costs.append(parsed)
    if problem_type == "tsp":
        for i in range(size):
            if costs[i][i] != 0.0:
                raise ValueError("tsp diagonal costs must be zero")
            for j in range(i + 1, size):
                if costs[i][j] != costs[j][i]:
                    raise ValueError("tsp costs must be symmetric")
    start = 0
    if problem_type == "tsp":
        if "start" in problem:
            start = problem["start"]
            if isinstance(start, (bool, np.bool_)) or not isinstance(
                start, int
            ):
                raise ValueError("'start' must be an integer")
            if not (0 <= start < size):
                raise ValueError("'start' must be within [0, size)")
    elif "start" in problem:
        raise ValueError("assignment must not declare 'start'")
    return AssignDataContract(
        version=version,
        problem_type=problem_type,
        size=size,
        costs=tuple(costs),
        start=start,
    )


def _validate_permutation(raw: Any, size: int, what: str) -> list[int]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError(f"'{what}' must be a JSON array of node indices")
    if len(raw) != size:
        raise ValueError(f"'{what}' must have exactly {size} entries")
    values: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
            raise ValueError(f"'{what}' entry {index} must be an integer")
        if not (0 <= value < size):
            raise ValueError(f"'{what}' entry {index} is out of range")
        values.append(value)
    if len(set(values)) != size:
        raise ValueError(f"'{what}' must be a permutation (no repeats)")
    return values


def validate_solution(
    payload: dict[str, Any], contract: AssignDataContract
) -> list[int]:
    """Validate a solution artifact; returns the permutation."""
    if contract.problem_type == "assignment":
        if "assignment" not in payload:
            raise ValueError("missing required field 'assignment'")
        return _validate_permutation(
            payload["assignment"], contract.size, "assignment"
        )
    if "tour" not in payload:
        raise ValueError("missing required field 'tour'")
    tour = _validate_permutation(payload["tour"], contract.size, "tour")
    if tour[0] != contract.start:
        raise ValueError(f"tour must start at {contract.start}")
    return tour
