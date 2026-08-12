"""Host-computed network-SIR metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.networksir.context import NetworkSirVerificationContext
from ves_modeling.networksir.data_contract import (
    NetworkSirDataContract,
    validate_estimate,
)

REFERENCE_SEED = 20260812
REFERENCE_REPLICATIONS = 2000


class NetworkSirVerifier:
    """EvidenceVerifier for network-SIR simulation artifacts."""

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, NetworkSirVerificationContext):
            raise TypeError(
                "NetworkSirVerifier requires NetworkSirVerificationContext"
            )
        payload = self._parse(raw_artifact)
        estimate, confidence_interval = validate_estimate(payload)
        metrics = compute_networksir_metrics(
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
                raise ValueError("network-SIR metrics must be finite")
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


def compute_networksir_metrics(
    estimate: float,
    confidence_interval: tuple[float, float] | None,
    reference: float,
) -> dict[str, float]:
    """Host error metrics against the averaged reference (all finite)."""
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


def _simulate_once(
    rng: np.random.Generator,
    contract: NetworkSirDataContract,
    steps: int,
    target_step: int,
) -> float:
    """One discrete-time network-SIR run; returns the requested quantity."""
    n = contract.n_nodes
    state = np.zeros(n, dtype=np.int8)
    state[: contract.i0] = 1
    cumulative = state == 1
    peak = int(contract.i0)
    infected_at_step: int | None = None
    adjacency = contract.adjacency
    beta = contract.beta
    gamma = contract.gamma
    for step in range(1, steps + 1):
        infected_indices = np.where(state == 1)[0]
        for node in infected_indices:
            for neighbor in adjacency[node]:
                if state[neighbor] == 0 and rng.random() < beta:
                    state[neighbor] = 1
                    cumulative[neighbor] = True
        infected_mask = state == 1
        recover = rng.random(n) < gamma
        state[recover & infected_mask] = 2
        infected_count = int(np.count_nonzero(state == 1))
        peak = max(peak, infected_count)
        if step == target_step:
            infected_at_step = infected_count
    if contract.quantity == "final_size":
        return float(np.count_nonzero(cumulative)) / n
    if contract.quantity == "peak_infected":
        return peak / n
    assert infected_at_step is not None
    return infected_at_step / n


def reference_networksir_value(contract: NetworkSirDataContract) -> float:
    """Host-held averaged stochastic network-SIR reference (never public)."""
    steps = max(1, round(contract.t_end))
    target_step = max(1, round(contract.t or contract.t_end))
    rng = np.random.default_rng(REFERENCE_SEED)
    values = [
        _simulate_once(rng, contract, steps, target_step)
        for _ in range(REFERENCE_REPLICATIONS)
    ]
    return float(np.mean(values))
