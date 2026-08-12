"""Host-computed LQR cost; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.lqr.context import LqrVerificationContext
from ves_modeling.lqr.data_contract import LqrDataContract, validate_control


class LqrVerifier:
    """EvidenceVerifier for LQR control-sequence artifacts."""

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, LqrVerificationContext):
            raise TypeError("LqrVerifier requires LqrVerificationContext")
        payload = self._parse(raw_artifact)
        control = validate_control(payload, context.contract)
        total_cost = simulate_total_cost(context.contract, control)
        reference = reference_optimal_cost(context.contract)
        observations = (
            Observation(
                value=total_cost,
                uncertainty=0.0,
                provenance="host:problem",
                name="total_cost",
            ),
            Observation(
                value=reference,
                uncertainty=0.0,
                provenance="host:problem",
                name="reference_optimal_cost",
            ),
        )
        for observation in observations:
            if not np.isfinite(observation.value):
                raise ValueError("LQR metrics must be finite")
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


def simulate_total_cost(
    contract: LqrDataContract, control: np.ndarray
) -> float:
    """Simulate x_{k+1}=A x_k + B u_k and recompute the total cost J."""
    x = contract.x0.astype(np.float64)
    cost = 0.0
    for step in range(contract.horizon):
        u = control[step]
        cost += float(x @ contract.Q @ x + u @ contract.R @ u)
        x = contract.A @ x + contract.B @ u
    cost += float(x @ contract.QN @ x)
    return float(cost)


def reference_optimal_cost(contract: LqrDataContract) -> float:
    """Discrete finite-horizon Riccati recursion; audit-only reference."""
    A = contract.A
    B = contract.B
    Q = contract.Q
    R = contract.R
    P = contract.QN
    for _ in range(contract.horizon):
        K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A + B @ K)
    return float(contract.x0 @ P @ contract.x0)
