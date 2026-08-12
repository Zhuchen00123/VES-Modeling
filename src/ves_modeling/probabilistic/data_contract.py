"""Probabilistic inference (parameter estimation) data contract (R18).

Public files: ``problem.json`` (distribution family + quantity to estimate)
and ``train.csv`` (observed samples, n >= 20, finite).  Host-only file:
``hidden_parameters.json`` (the true parameters used to generate the
samples; never public).  The host computes the exact reference from the
hidden parameters analytically.

Problem families: normal (mean/std), gamma (shape/scale), beta
(alpha/beta).  Quantities: mean | variance | quantile (with q in (0,1)) |
probability_ge (with threshold).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FAMILIES = ("normal", "gamma", "beta")
QUANTITIES = ("mean", "variance", "quantile", "probability_ge")
MIN_SAMPLES = 20


@dataclass(frozen=True)
class ProbabilisticDataContract:
    """Canonical public probabilistic inference problem."""

    version: int
    family: str
    quantity: str
    q: float | None = field(default=None, repr=False, compare=False)
    threshold: float | None = field(default=None, repr=False, compare=False)
    sample_column: str = "value"
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "family": self.family,
            "quantity": self.quantity,
            "q": self.q,
            "threshold": self.threshold,
            "sample_column": self.sample_column,
            "n_samples": self.n_samples,
        }


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in JSON object: {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {name}: {exc}") from None
    except OSError as exc:
        raise ValueError(f"cannot read {name}: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{name} root must be an object")
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


def validate_probabilistic_data(
    public_dir: Path,
    *,
    sample_column: str = "value",
) -> ProbabilisticDataContract:
    """Validate problem.json and train.csv; returns the public contract."""
    if not sample_column.strip():
        raise ValueError("sample_column must be non-empty")
    problem = _load_json_object(
        Path(public_dir) / "problem.json", "problem.json"
    )
    unknown_top = set(problem) - {
        "version",
        "family",
        "quantity",
        "q",
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
    if "family" not in problem:
        raise ValueError("problem.json is missing 'family'")
    family = problem["family"]
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if "quantity" not in problem:
        raise ValueError("problem.json is missing 'quantity'")
    quantity = problem["quantity"]
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}")
    q: float | None = None
    threshold: float | None = None
    if quantity == "quantile":
        if "q" not in problem:
            raise ValueError("quantity 'quantile' requires 'q'")
        q = _finite_number(problem["q"], "q")
        if not (0.0 < q < 1.0):
            raise ValueError("q must be within (0, 1)")
        if "threshold" in problem:
            raise ValueError("quantity 'quantile' must not declare 'threshold'")
    elif quantity == "probability_ge":
        if "threshold" not in problem:
            raise ValueError(
                "quantity 'probability_ge' requires 'threshold'"
            )
        threshold = _finite_number(problem["threshold"], "threshold")
        if "q" in problem:
            raise ValueError(
                "quantity 'probability_ge' must not declare 'q'"
            )
    else:
        if "q" in problem or "threshold" in problem:
            raise ValueError(
                f"quantity {quantity!r} must not declare 'q'/'threshold'"
            )
    train_path = Path(public_dir) / "train.csv"
    try:
        train = pd.read_csv(train_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read train.csv: {exc}") from None
    if sample_column not in train.columns:
        raise ValueError(
            f"train.csv must contain sample column {sample_column!r}"
        )
    samples = train[sample_column]
    if len(samples) < MIN_SAMPLES:
        raise ValueError(
            f"train.csv needs at least {MIN_SAMPLES} samples"
        )
    try:
        values = samples.to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"train sample column {sample_column!r} must be numeric"
        ) from exc
    if not np.isfinite(values).all():
        raise ValueError("train samples must be finite (no NaN/Infinity)")
    return ProbabilisticDataContract(
        version=version,
        family=family,
        quantity=quantity,
        q=q,
        threshold=threshold,
        sample_column=sample_column,
        n_samples=int(values.size),
    )


def load_hidden_parameters(
    host_dir: Path, contract: ProbabilisticDataContract
) -> dict[str, float]:
    """Load the true parameters (host-only, never exposed to candidates)."""
    params = _load_json_object(
        Path(host_dir) / "hidden_parameters.json", "hidden_parameters.json"
    )
    if contract.family == "normal":
        required = {"mean", "std"}
        positive = {"std"}
    elif contract.family == "gamma":
        required = {"shape", "scale"}
        positive = {"shape", "scale"}
    else:
        required = {"alpha", "beta"}
        positive = {"alpha", "beta"}
    if set(params) != required:
        raise ValueError(
            f"hidden_parameters.json must contain exactly {sorted(required)} "
            f"for family {contract.family!r}"
        )
    values: dict[str, float] = {}
    for name in required:
        value = _finite_number(params[name], f"hidden parameter {name}")
        if name in positive and value <= 0.0:
            raise ValueError(f"hidden parameter {name} must be positive")
        values[name] = value
    return values


def compute_reference(
    contract: ProbabilisticDataContract,
    parameters: dict[str, float],
) -> float:
    """Exact reference value from the hidden parameters (host-only)."""
    from scipy.stats import beta, gamma, norm

    family = contract.family
    quantity = contract.quantity
    if family == "normal":
        mean, std = parameters["mean"], parameters["std"]
        if quantity == "mean":
            value = mean
        elif quantity == "variance":
            value = std**2
        elif quantity == "quantile":
            value = norm.ppf(contract.q, mean, std)
        else:
            value = norm.sf(contract.threshold, mean, std)
    elif family == "gamma":
        shape, scale = parameters["shape"], parameters["scale"]
        if quantity == "mean":
            value = shape * scale
        elif quantity == "variance":
            value = shape * scale**2
        elif quantity == "quantile":
            value = gamma.ppf(contract.q, shape, scale=scale)
        else:
            value = gamma.sf(contract.threshold, shape, scale=scale)
    else:
        alpha, b = parameters["alpha"], parameters["beta"]
        if quantity == "mean":
            value = alpha / (alpha + b)
        elif quantity == "variance":
            value = alpha * b / (
                (alpha + b) ** 2 * (alpha + b + 1.0)
            )
        elif quantity == "quantile":
            value = beta.ppf(contract.q, alpha, b)
        else:
            value = beta.sf(contract.threshold, alpha, b)
    reference = float(value)
    if not math.isfinite(reference):
        raise ValueError("analytic reference must be finite")
    return reference


def validate_solution(
    payload: dict[str, Any], contract: ProbabilisticDataContract
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
