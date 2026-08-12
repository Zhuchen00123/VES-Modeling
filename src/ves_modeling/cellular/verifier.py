"""Host-computed CA metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.cellular.context import CellularVerificationContext
from ves_modeling.cellular.data_contract import (
    CellularDataContract,
    ca_step,
    validate_estimate,
)


class CellularVerifier:
    """EvidenceVerifier for cellular-automaton artifacts."""

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, CellularVerificationContext):
            raise TypeError(
                "CellularVerifier requires CellularVerificationContext"
            )
        payload = self._parse(raw_artifact)
        estimate, confidence_interval = validate_estimate(payload)
        metrics = compute_cellular_metrics(
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
                raise ValueError("cellular metrics must be finite")
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


def compute_cellular_metrics(
    estimate: float,
    confidence_interval: tuple[float, float] | None,
    reference: float,
) -> dict[str, float]:
    """Host error metrics against the exact CA reference (all finite)."""
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


def reference_ca_value(contract: CellularDataContract) -> float:
    """Host-held exact deterministic CA reference (never public)."""
    state = contract.initial.copy()
    persistent = state.copy()
    for _ in range(contract.steps):
        state = ca_step(state, contract.rule, contract.width)
        persistent = persistent & state
    if contract.quantity == "final_density":
        return float(np.count_nonzero(state)) / contract.width
    if contract.quantity == "cell_state":
        assert contract.index is not None
        return float(state[contract.index])
    return float(np.count_nonzero(persistent))
