"""One-dimensional cellular automaton data contract (R29).

``problem.json`` is the complete public instance (no hidden truth):
``version``, ``rule`` (int in [0, 255]), ``width`` (int in [20, 200]),
``steps`` (int in [1, 200]), ``initial`` (exactly ``width`` binary values,
at least one 1) and ``quantity``: ``final_density`` |
``cell_state`` (with ``index`` in [0, width)) | ``persistent_ones``.

Artifact ``solution.json``: ``{"estimate": finite_number,
"confidence_interval": [lo, hi]}`` — CI optional, lo <= estimate <= hi.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

QUANTITIES = ("final_density", "cell_state", "persistent_ones")


@dataclass(frozen=True)
class CellularDataContract:
    """Canonical public cellular-automaton instance."""

    rule: int
    width: int
    steps: int
    initial: np.ndarray = field(repr=False)
    quantity: str
    index: int | None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rule": self.rule,
            "width": self.width,
            "steps": self.steps,
            "initial_ones": int(np.count_nonzero(self.initial)),
            "quantity": self.quantity,
        }
        if self.index is not None:
            result["index"] = self.index
        return result


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in JSON object: {key!r}")
        result[key] = value
    return result


def _strict_int(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer")
    return int(value)


def validate_cellular_data(public_dir: Path) -> CellularDataContract:
    """Validate the public problem.json and return the canonical contract."""
    try:
        with (Path(public_dir) / "problem.json").open(
            encoding="utf-8"
        ) as handle:
            problem = json.load(handle, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in problem.json: {exc}") from None
    except OSError as exc:
        raise ValueError(f"cannot read problem.json: {exc}") from None
    if not isinstance(problem, dict):
        raise ValueError("problem.json root must be an object")
    unknown_top = set(problem) - {
        "version",
        "rule",
        "width",
        "steps",
        "initial",
        "quantity",
        "index",
    }
    if unknown_top:
        raise ValueError(
            f"problem.json has unknown top-level fields: {sorted(unknown_top)}"
        )
    if "version" not in problem:
        raise ValueError("problem.json is missing 'version'")
    version = _strict_int(problem["version"], "problem.json 'version'")
    if version < 1:
        raise ValueError("problem.json 'version' must be >= 1")
    for key in ("rule", "width", "steps", "initial", "quantity"):
        if key not in problem:
            raise ValueError(f"problem.json is missing {key!r}")
    rule = _strict_int(problem["rule"], "rule")
    if not 0 <= rule <= 255:
        raise ValueError("rule must be an integer in [0, 255]")
    width = _strict_int(problem["width"], "width")
    if not 20 <= width <= 200:
        raise ValueError("width must be an integer in [20, 200]")
    steps = _strict_int(problem["steps"], "steps")
    if not 1 <= steps <= 200:
        raise ValueError("steps must be an integer in [1, 200]")
    initial_raw = problem["initial"]
    if isinstance(initial_raw, (str, bytes)) or not isinstance(
        initial_raw, list
    ):
        raise ValueError("'initial' must be a JSON array")
    if len(initial_raw) != width:
        raise ValueError(f"'initial' must have exactly {width} entries")
    initial_values: list[bool] = []
    for index, value in enumerate(initial_raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
            raise ValueError(f"initial[{index}] must be an integer")
        if value not in (0, 1):
            raise ValueError(f"initial[{index}] must be 0 or 1")
        initial_values.append(bool(value))
    initial = np.asarray(initial_values, dtype=bool)
    if not initial.any():
        raise ValueError("'initial' must contain at least one 1")
    quantity = problem["quantity"]
    if not isinstance(quantity, str) or quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}")
    index: int | None = None
    if quantity == "cell_state":
        if "index" not in problem:
            raise ValueError("quantity 'cell_state' requires 'index'")
        index = _strict_int(problem["index"], "index")
        if not 0 <= index < width:
            raise ValueError("index must lie in [0, width)")
    elif "index" in problem:
        raise ValueError("'index' is only valid for quantity 'cell_state'")
    return CellularDataContract(
        rule=rule,
        width=width,
        steps=steps,
        initial=initial,
        quantity=quantity,
        index=index,
    )


def _finite_number(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float)
    ):
        raise ValueError(f"{what} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite (no NaN/Infinity)")
    return number


def validate_estimate(
    payload: dict[str, Any],
) -> tuple[float, tuple[float, float] | None]:
    """Validate a solution artifact; returns (estimate, CI or None)."""
    if "estimate" not in payload:
        raise ValueError("missing required field 'estimate'")
    estimate = _finite_number(payload["estimate"], "estimate")
    confidence_interval: tuple[float, float] | None = None
    if "confidence_interval" in payload:
        raw = payload["confidence_interval"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
            raise ValueError("'confidence_interval' must be a JSON array")
        if len(raw) != 2:
            raise ValueError(
                "'confidence_interval' must have exactly 2 entries"
            )
        lo = _finite_number(raw[0], "confidence_interval[0]")
        hi = _finite_number(raw[1], "confidence_interval[1]")
        if lo > hi:
            raise ValueError("confidence_interval lo must not exceed hi")
        if estimate < lo or estimate > hi:
            raise ValueError("estimate must lie within confidence_interval")
        confidence_interval = (lo, hi)
    return estimate, confidence_interval


def ca_step(state: np.ndarray, rule: int, width: int) -> np.ndarray:
    """One elementary-CA step with periodic boundary (exact, bool)."""
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    triplets = (left.astype(np.uint8) << 2) | (
        state.astype(np.uint8) << 1
    ) | right.astype(np.uint8)
    lookup = np.asarray(
        [(rule >> bit) & 1 for bit in range(8)], dtype=np.uint8
    )
    return lookup[triplets].astype(bool)
