"""Host-computed SIR metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.sir.context import SirVerificationContext
from ves_modeling.sir.data_contract import (
    SirDataContract,
    validate_estimate,
)


class SirVerifier:
    """EvidenceVerifier for SIR simulation artifacts."""

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, SirVerificationContext):
            raise TypeError("SirVerifier requires SirVerificationContext")
        payload = self._parse(raw_artifact)
        estimate, confidence_interval = validate_estimate(payload)
        metrics = compute_sir_metrics(
            estimate,
            confidence_interval,
            context.reference_value(),
        )
        observations = (
            Observation(
                value=metrics["absolute_error"],
                uncertainty=0.0,
                provenance="host:reference",
                name="absolute_error",
            ),
            Observation(
                value=metrics["relative_error"],
                uncertainty=0.0,
                provenance="host:reference",
                name="relative_error",
            ),
            Observation(
                value=metrics["ci_coverage"],
                uncertainty=0.0,
                provenance="host:reference",
                name="ci_coverage",
            ),
        )
        for observation in observations:
            if not np.isfinite(observation.value):
                raise ValueError("SIR metrics must be finite")
        return Evidence(observations=observations)

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


def compute_sir_metrics(
    estimate: float,
    confidence_interval: tuple[float, float] | None,
    reference: float,
) -> dict[str, float]:
    """Host error metrics against the SIR reference (all finite)."""
    absolute_error = abs(estimate - reference)
    if reference != 0.0:
        relative_error = absolute_error / abs(reference)
    else:
        relative_error = absolute_error
    if confidence_interval is not None:
        lo, hi = confidence_interval
        ci_coverage = 1.0 if lo <= reference <= hi else 0.0
    else:
        ci_coverage = 0.0
    return {
        "absolute_error": float(absolute_error),
        "relative_error": float(relative_error),
        "ci_coverage": float(ci_coverage),
    }


def reference_sir_value(contract: SirDataContract) -> float:
    """Host-held numerical SIR reference via scipy RK45 (never public)."""
    from scipy.integrate import solve_ivp

    n = contract.N
    i0 = contract.i0
    r0 = contract.r0
    s0 = n - i0 - r0
    beta = contract.beta
    gamma = contract.gamma
    t_end = contract.t_end

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        s, i, _ = y
        return [
            -beta * s * i / n,
            beta * s * i / n - gamma * i,
            gamma * i,
        ]

    t_eval = None
    if contract.quantity == "infected_at":
        t_eval = [float(contract.t)]
    solution = solve_ivp(
        rhs,
        (0.0, t_end),
        [float(s0), float(i0), float(r0)],
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
        t_eval=t_eval,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"SIR reference failed: {solution.message}")
    if contract.quantity == "final_size":
        s_end = float(solution.y[0, -1])
        return (n - s_end) / n
    if contract.quantity == "peak_infected":
        times = np.linspace(0.0, t_end, 20001)
        values = solution.sol(times)
        return float(np.max(values[1])) / n
    return float(solution.y[1, 0]) / n
