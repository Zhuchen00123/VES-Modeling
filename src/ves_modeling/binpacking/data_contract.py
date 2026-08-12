"""One-dimensional bin packing data contract (R24).

``problem.json`` is the complete public instance (no hidden truth):
version, capacity (finite > 0), items (finite positive sizes, each <=
capacity) and n_items.

Artifact ``solution.json``: ``{"assignment": [bin_index, ...]}`` with one
entry per item; bin indices are non-negative integers and the used bins are
contiguous ``0..k-1`` (no gaps).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BinDataContract:
    """Canonical public bin packing problem."""

    version: int
    capacity: float
    items: tuple[float, ...] = field(repr=False, compare=False)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "capacity": self.capacity,
            "items": list(self.items),
            "n_items": self.n_items,
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


def validate_bin_data(
    public_dir: Path,
) -> BinDataContract:
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
        "capacity",
        "items",
        "n_items",
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
    if "capacity" not in problem:
        raise ValueError("problem.json is missing 'capacity'")
    capacity = _finite_number(problem["capacity"], "capacity")
    if capacity <= 0.0:
        raise ValueError("capacity must be positive")
    if "items" not in problem:
        raise ValueError("problem.json is missing 'items'")
    items_raw = problem["items"]
    if not isinstance(items_raw, list) or not items_raw:
        raise ValueError("'items' must be a non-empty JSON array")
    items = []
    for index, value in enumerate(items_raw):
        size = _finite_number(value, f"items[{index}]")
        if size <= 0.0:
            raise ValueError(f"items[{index}] must be positive")
        if size > capacity:
            raise ValueError(f"items[{index}] must not exceed capacity")
        items.append(size)
    if "n_items" in problem:
        n_items = problem["n_items"]
        if isinstance(n_items, (bool, np.bool_)) or not isinstance(
            n_items, int
        ):
            raise ValueError("'n_items' must be an integer")
        if n_items != len(items):
            raise ValueError("'n_items' must match the items list length")
    return BinDataContract(
        version=version, capacity=capacity, items=tuple(items)
    )


def validate_solution(
    payload: dict[str, Any], contract: BinDataContract
) -> list[int]:
    """Validate a solution artifact; returns the per-item bin assignment."""
    if "assignment" not in payload:
        raise ValueError("missing required field 'assignment'")
    raw = payload["assignment"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError("'assignment' must be a JSON array")
    if len(raw) != contract.n_items:
        raise ValueError(
            f"'assignment' must have exactly {contract.n_items} entries"
        )
    assignment: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
            raise ValueError(f"assignment[{index}] must be an integer")
        if value < 0:
            raise ValueError(f"assignment[{index}] must be non-negative")
        assignment.append(value)
    used = sorted(set(assignment))
    if used and used != list(range(len(used))):
        raise ValueError("used bins must be contiguous 0..k-1 (no gaps)")
    return assignment
