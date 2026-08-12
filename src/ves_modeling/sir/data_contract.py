"""SIR epidemic simulation data contract (R28).

``problem.json`` is the complete public instance (no hidden truth):
``version``, ``model: "sir"``, ``beta`` (> 0), ``gamma`` (> 0), ``N``
(int >= 100 population), ``i0`` (int >= 1 initial infected), ``r0``
(int >= 0 initial recovered, default 0), ``t_end`` (> 0) and ``quantity``:
``final_size`` | ``peak_infected`` | ``infected_at`` (with ``t`` in
(0, t_end]).

Artifact ``solution.json``: ``{"estimate": finite_number,
"confidence_interval": [lo, hi]}`` — CI optional, lo <= estimate <= hi.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

QUANTITIES = ("final_size", "peak_infected", "infected_at")


@dataclass(frozen=True)
class SirDataContract:
    """Canonical public SIR instance (never host secrets)."""

    beta: float
    gamma: float
    N: int
    i0: int
    r0: int
    t_end: float
    quantity: str
    t: float | None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": "sir",
            "beta": self.beta,
            "gamma": self.gamma,
            "N": self.N,
            "i0": self.i0,
            "r0": self.r0,
            "t_end": self.t_end,
            "quantity": self.quantity,
        }
        if self.t is not None:
            result["t"] = self.t
        return result


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


def _positive_number(value: Any, what: str) -> float:
    number = _finite_number(value, what)
    if number <= 0.0:
        raise ValueError(f"{what} must be positive")
    return number


def _strict_int(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer")
    return int(value)


def validate_sir_data(public_dir: Path) -> SirDataContract:
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
        "model",
        "beta",
        "gamma",
        "N",
        "i0",
        "r0",
        "t_end",
        "quantity",
        "t",
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
    if "model" not in problem:
        raise ValueError("problem.json is missing 'model'")
    if problem["model"] != "sir":
        raise ValueError("problem.json 'model' must be 'sir'")
    for key in ("beta", "gamma", "N", "i0", "t_end", "quantity"):
        if key not in problem:
            raise ValueError(f"problem.json is missing {key!r}")
    beta = _positive_number(problem["beta"], "beta")
    gamma = _positive_number(problem["gamma"], "gamma")
    N = _strict_int(problem["N"], "N")
    if N < 100:
        raise ValueError("N must be an integer >= 100")
    i0 = _strict_int(problem["i0"], "i0")
    if i0 < 1:
        raise ValueError("i0 must be an integer >= 1")
    r0 = _strict_int(problem.get("r0", 0), "r0")
    if r0 < 0:
        raise ValueError("r0 must be an integer >= 0")
    if i0 + r0 > N:
        raise ValueError("i0 + r0 must not exceed N")
    t_end = _positive_number(problem["t_end"], "t_end")
    quantity = problem["quantity"]
    if not isinstance(quantity, str) or quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}")
    t: float | None = None
    if quantity == "infected_at":
        if "t" not in problem:
            raise ValueError("quantity 'infected_at' requires 't'")
        t = _finite_number(problem["t"], "t")
        if t <= 0.0 or t > t_end:
            raise ValueError("t must lie in (0, t_end]")
    elif "t" in problem:
        raise ValueError("'t' is only valid for quantity 'infected_at'")
    return SirDataContract(
        beta=beta,
        gamma=gamma,
        N=N,
        i0=i0,
        r0=r0,
        t_end=t_end,
        quantity=quantity,
        t=t,
    )


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
            raise ValueError(
                "confidence_interval lo must not exceed hi"
            )
        if estimate < lo or estimate > hi:
            raise ValueError("estimate must lie within confidence_interval")
        confidence_interval = (lo, hi)
    return estimate, confidence_interval
