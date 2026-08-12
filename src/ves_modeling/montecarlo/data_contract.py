"""Monte Carlo / stochastic simulation data contract (R15).

``problem.json`` is the complete public instance (no hidden truth).  The
host computes an exact/analytic reference value from the public inputs and
never exposes it to candidates, who are asked to estimate the quantity by
simulation.

Problem kinds:
- expectation: discrete distribution (outcomes + probabilities) with a
  target (mean | second_moment | variance | prob_ge | prob_le);
- integral: polynomial coefficients over [a, b] (closed-form integral);
- probability: binomial n/p with event (ge | le | eq) and threshold.

No eval / arbitrary function strings are allowed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

KINDS = ("expectation", "integral", "probability")
EXPECTATION_TARGETS = (
    "mean",
    "second_moment",
    "variance",
    "prob_ge",
    "prob_le",
)
PROBABILITY_EVENTS = ("ge", "le", "eq")
PROB_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class MonteCarloDataContract:
    """Canonical public Monte Carlo problem (reference kept host-only)."""

    version: int
    kind: str
    params: dict[str, Any] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "params": json.loads(json.dumps(self.params, allow_nan=False)),
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


def _validate_expectation(raw: Any) -> dict[str, Any]:
    unknown = set(raw) - {"outcomes", "probabilities", "target", "threshold"}
    if unknown:
        raise ValueError(
            f"expectation has unknown fields: {sorted(unknown)}"
        )
    if "outcomes" not in raw or "probabilities" not in raw:
        raise ValueError(
            "expectation requires 'outcomes' and 'probabilities'"
        )
    if "target" not in raw:
        raise ValueError("expectation requires 'target'")
    outcomes_raw = raw["outcomes"]
    probabilities_raw = raw["probabilities"]
    if not isinstance(outcomes_raw, list) or not isinstance(
        probabilities_raw, list
    ):
        raise ValueError(
            "'outcomes' and 'probabilities' must be JSON arrays"
        )
    if len(outcomes_raw) < 2:
        raise ValueError("expectation needs at least two outcomes")
    if len(outcomes_raw) != len(probabilities_raw):
        raise ValueError(
            "outcomes and probabilities must have the same length"
        )
    outcomes = [
        _finite_number(value, "outcome") for value in outcomes_raw
    ]
    probabilities = [
        _finite_number(value, "probability") for value in probabilities_raw
    ]
    if any(value < 0.0 for value in probabilities):
        raise ValueError("probabilities must be non-negative")
    if not math.isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=PROB_SUM_TOLERANCE
    ):
        raise ValueError(
            "probabilities must sum to 1 within "
            f"{PROB_SUM_TOLERANCE}"
        )
    target = raw["target"]
    if target not in EXPECTATION_TARGETS:
        raise ValueError(
            f"target must be one of {EXPECTATION_TARGETS}"
        )
    threshold: float | None = None
    if target in ("prob_ge", "prob_le"):
        if "threshold" not in raw:
            raise ValueError(f"target {target!r} requires 'threshold'")
        threshold = _finite_number(raw["threshold"], "threshold")
    else:
        if "threshold" in raw:
            raise ValueError(
                f"target {target!r} must not declare 'threshold'"
            )
    return {
        "outcomes": outcomes,
        "probabilities": probabilities,
        "target": target,
        "threshold": threshold,
    }


def _validate_integral(raw: Any) -> dict[str, Any]:
    unknown = set(raw) - {"integrand", "coefficients", "interval"}
    if unknown:
        raise ValueError(f"integral has unknown fields: {sorted(unknown)}")
    if raw.get("integrand") != "polynomial":
        raise ValueError("integral integrand must be 'polynomial'")
    if "coefficients" not in raw or "interval" not in raw:
        raise ValueError(
            "integral requires 'coefficients' and 'interval'"
        )
    coefficients_raw = raw["coefficients"]
    if not isinstance(coefficients_raw, list) or not coefficients_raw:
        raise ValueError("'coefficients' must be a non-empty JSON array")
    coefficients = [
        _finite_number(value, "coefficient")
        for value in coefficients_raw
    ]
    interval_raw = raw["interval"]
    if (
        not isinstance(interval_raw, list)
        or len(interval_raw) != 2
        or isinstance(interval_raw[0], bool)
        or isinstance(interval_raw[1], bool)
    ):
        raise ValueError("'interval' must be a [a, b] pair of numbers")
    a = _finite_number(interval_raw[0], "interval a")
    b = _finite_number(interval_raw[1], "interval b")
    if a >= b:
        raise ValueError("interval requires a < b")
    return {
        "integrand": "polynomial",
        "coefficients": coefficients,
        "interval": [a, b],
    }


def _validate_probability(raw: Any) -> dict[str, Any]:
    unknown = set(raw) - {
        "distribution",
        "n",
        "p",
        "event",
        "threshold",
    }
    if unknown:
        raise ValueError(f"probability has unknown fields: {sorted(unknown)}")
    if raw.get("distribution") != "binomial":
        raise ValueError("probability distribution must be 'binomial'")
    if "n" not in raw or "p" not in raw or "event" not in raw:
        raise ValueError(
            "probability requires 'n', 'p' and 'event'"
        )
    n = raw["n"]
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, int) or n < 1:
        raise ValueError("binomial 'n' must be an integer >= 1")
    p = _finite_number(raw["p"], "binomial p")
    if not (0.0 < p < 1.0):
        raise ValueError("binomial 'p' must be within (0, 1)")
    event = raw["event"]
    if event not in PROBABILITY_EVENTS:
        raise ValueError(
            f"event must be one of {PROBABILITY_EVENTS}"
        )
    threshold = raw.get("threshold")
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, int)
        or not (0 <= threshold <= n)
    ):
        raise ValueError(
            "binomial 'threshold' must be an integer in [0, n]"
        )
    return {
        "distribution": "binomial",
        "n": n,
        "p": p,
        "event": event,
        "threshold": threshold,
    }


def validate_montecarlo_data(
    public_dir: Path,
) -> MonteCarloDataContract:
    """Validate the public problem.json and return the contract (no ref)."""
    problem = load_problem(Path(public_dir) / "problem.json")
    unknown_top = set(problem) - {"version", "kind", "params"}
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
    if "params" not in problem:
        raise ValueError("problem.json is missing 'params'")
    params_raw = problem["params"]
    if not isinstance(params_raw, dict):
        raise ValueError("'params' must be an object")
    if kind == "expectation":
        params = _validate_expectation(params_raw)
    elif kind == "integral":
        params = _validate_integral(params_raw)
    else:
        params = _validate_probability(params_raw)
    return MonteCarloDataContract(
        version=version, kind=kind, params=params
    )


def compute_reference(contract: MonteCarloDataContract) -> float:
    """Exact/analytic reference value (host-only, never exposed)."""
    if contract.kind == "expectation":
        params = contract.params
        outcomes = np.asarray(params["outcomes"], dtype=np.float64)
        probabilities = np.asarray(
            params["probabilities"], dtype=np.float64
        )
        target = params["target"]
        if target == "mean":
            value = float(np.sum(probabilities * outcomes))
        elif target == "second_moment":
            value = float(np.sum(probabilities * outcomes**2))
        elif target == "variance":
            mean = float(np.sum(probabilities * outcomes))
            second = float(np.sum(probabilities * outcomes**2))
            value = second - mean**2
        elif target == "prob_ge":
            threshold = params["threshold"]
            value = float(
                np.sum(probabilities[outcomes >= threshold])
            )
        else:
            threshold = params["threshold"]
            value = float(
                np.sum(probabilities[outcomes <= threshold])
            )
    elif contract.kind == "integral":
        coefficients = contract.params["coefficients"]
        a, b = contract.params["interval"]
        value = 0.0
        for degree, coefficient in enumerate(coefficients):
            value += coefficient * (
                b ** (degree + 1) - a ** (degree + 1)
            ) / (degree + 1)
        value = float(value)
    else:
        n = contract.params["n"]
        p = contract.params["p"]
        threshold = contract.params["threshold"]
        event = contract.params["event"]
        from scipy.stats import binom

        if event == "ge":
            value = float(binom.sf(threshold - 1, n, p))
        elif event == "le":
            value = float(binom.cdf(threshold, n, p))
        else:
            value = float(binom.pmf(threshold, n, p))
    if not math.isfinite(value):
        raise ValueError("analytic reference must be finite")
    return value


def validate_solution(
    payload: dict[str, Any], contract: MonteCarloDataContract
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
