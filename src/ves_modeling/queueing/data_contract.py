"""Queueing theory data contract (R19).

``problem.json`` is the complete public instance (no hidden truth): queue
kind (mm1 | mmc) with arrival/service rates and a quantity to estimate.  The
host holds the analytic reference values (M/M/1 closed form, M/M/c
Erlang-C via scipy) and never exposes them to candidates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

KINDS = ("mm1", "mmc")
QUANTITIES = (
    "mean_wait",
    "mean_queue",
    "mean_utilization",
    "prob_wait_gt",
)


@dataclass(frozen=True)
class QueueingDataContract:
    """Canonical public queueing problem (reference kept host-only)."""

    version: int
    kind: str
    lambda_: float = field(repr=False, compare=False)
    mu: float = field(repr=False, compare=False)
    quantity: str = field(repr=False, compare=False)
    c: int = field(default=1, repr=False, compare=False)
    threshold: float | None = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "lambda": self.lambda_,
            "mu": self.mu,
            "c": self.c,
            "quantity": self.quantity,
            "threshold": self.threshold,
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


def validate_queueing_data(
    public_dir: Path,
) -> QueueingDataContract:
    """Validate the public problem.json and return the contract."""
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
        "kind",
        "lambda",
        "mu",
        "c",
        "quantity",
        "threshold",
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
    if "kind" not in problem:
        raise ValueError("problem.json is missing 'kind'")
    kind = problem["kind"]
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if "lambda" not in problem or "mu" not in problem:
        raise ValueError("problem.json requires 'lambda' and 'mu'")
    lambda_ = _finite_number(problem["lambda"], "lambda")
    mu = _finite_number(problem["mu"], "mu")
    if lambda_ <= 0.0 or mu <= 0.0:
        raise ValueError("lambda and mu must be positive")
    c = 1
    if kind == "mmc":
        if "c" not in problem:
            raise ValueError("kind 'mmc' requires 'c'")
        c = problem["c"]
        if isinstance(c, (bool, np.bool_)) or not isinstance(c, int) or c < 1:
            raise ValueError("'c' must be an integer >= 1")
    elif "c" in problem:
        raise ValueError("kind 'mm1' must not declare 'c'")
    rho = lambda_ / (mu * c)
    if rho >= 1.0:
        raise ValueError("queue must be stable (rho < 1)")
    if "quantity" not in problem:
        raise ValueError("problem.json is missing 'quantity'")
    quantity = problem["quantity"]
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}")
    threshold: float | None = None
    if quantity == "prob_wait_gt":
        if "threshold" not in problem:
            raise ValueError(
                "quantity 'prob_wait_gt' requires 'threshold'"
            )
        threshold = _finite_number(problem["threshold"], "threshold")
        if threshold < 0.0:
            raise ValueError("threshold must be >= 0")
    elif "threshold" in problem:
        raise ValueError(
            f"quantity {quantity!r} must not declare 'threshold'"
        )
    return QueueingDataContract(
        version=version,
        kind=kind,
        lambda_=lambda_,
        mu=mu,
        c=c,
        quantity=quantity,
        threshold=threshold,
    )


def compute_reference(contract: QueueingDataContract) -> float:
    """Exact analytic reference value (host-only, never exposed)."""
    lambda_ = contract.lambda_
    mu = contract.mu
    c = contract.c
    rho = lambda_ / (mu * c)
    if contract.kind == "mm1":
        if contract.quantity == "mean_wait":
            value = rho / (mu * (1.0 - rho))
        elif contract.quantity == "mean_queue":
            value = rho**2 / (1.0 - rho)
        elif contract.quantity == "mean_utilization":
            value = rho
        else:
            value = rho * math.exp(-mu * (1.0 - rho) * contract.threshold)
    else:
        # M/M/c Erlang-C.
        term = (c * rho) ** c / (math.factorial(c) * (1.0 - rho))
        series = sum(
            (c * rho) ** k / math.factorial(k)
            for k in range(c)
        )
        p_wait = term / (series + term)
        if contract.quantity == "mean_wait":
            value = p_wait / (c * mu * (1.0 - rho))
        elif contract.quantity == "mean_queue":
            value = lambda_ * p_wait / (c * mu * (1.0 - rho))
        elif contract.quantity == "mean_utilization":
            value = rho
        else:
            value = p_wait * math.exp(
                -c * mu * (1.0 - rho) * contract.threshold
            )
    reference = float(value)
    if not math.isfinite(reference):
        raise ValueError("analytic reference must be finite")
    return reference


def validate_solution(
    payload: dict[str, Any], contract: QueueingDataContract
) -> tuple[float, tuple[float, float] | None]:
    """Validate a solution artifact; returns (estimate, CI or None)."""
    if "estimate" not in payload:
        raise ValueError("missing required field 'estimate'")
    estimate = _finite_number(payload["estimate"], "estimate")
    confidence_interval: tuple[float, float] | None = None
    if "confidence_interval" in payload:
        raw_ci = payload["confidence_interval"]
        if (
            not isinstance(raw_ci, list)
            or len(raw_ci) != 2
            or isinstance(raw_ci[0], bool)
            or isinstance(raw_ci[1], bool)
        ):
            raise ValueError(
                "'confidence_interval' must be a [lo, hi] pair of numbers"
            )
        lo = _finite_number(raw_ci[0], "confidence_interval lo")
        hi = _finite_number(raw_ci[1], "confidence_interval hi")
        if lo > hi:
            raise ValueError("confidence_interval lo must not exceed hi")
        if not (lo <= estimate <= hi):
            raise ValueError(
                "estimate must lie within the confidence_interval"
            )
        confidence_interval = (lo, hi)
    return estimate, confidence_interval
