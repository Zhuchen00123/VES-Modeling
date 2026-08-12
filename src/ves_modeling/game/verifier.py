"""Host-computed game cost; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.game.context import GameVerificationContext
from ves_modeling.game.data_contract import (
    GameDataContract,
    validate_control,
)


class GameVerifier:
    """EvidenceVerifier for LQ game control-sequence artifacts."""

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, GameVerificationContext):
            raise TypeError("GameVerifier requires GameVerificationContext")
        payload = self._parse(raw_artifact)
        control = validate_control(payload, context.contract)
        total_cost = simulate_game_total_cost(context.contract, control)
        reference = reference_game_optimal_cost(context.contract)
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
                raise ValueError("game metrics must be finite")
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


def _game_riccati_backward(
    contract: GameDataContract,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Game Riccati recursion; returns (P_0, controller, disturbance gains)."""
    A = contract.A
    B = contract.B
    C = contract.C
    Q = contract.Q
    R = contract.R
    S = contract.S
    H = np.concatenate([B, C], axis=1)
    D = np.zeros((contract.m + contract.p, contract.m + contract.p))
    D[: contract.m, : contract.m] = R
    D[contract.m :, contract.m :] = -S
    P = contract.QN
    controller_gains: list[np.ndarray] = []
    disturbance_gains: list[np.ndarray] = []
    for _ in range(contract.horizon):
        try:
            inverse = np.linalg.inv(D + H.T @ P @ H)
        except np.linalg.LinAlgError as exc:
            raise ValueError("game Riccati recursion is not well-posed") from exc
        feedback = -(inverse @ H.T @ P @ A)
        controller_gains.append(feedback[: contract.m, :])
        disturbance_gains.append(feedback[contract.m :, :])
        P = Q + A.T @ P @ A - A.T @ P @ H @ inverse @ H.T @ P @ A
    controller_gains.reverse()
    disturbance_gains.reverse()
    return P, controller_gains, disturbance_gains


def reference_game_optimal_cost(contract: GameDataContract) -> float:
    """Game-Riccati optimal cost J*; audit-only reference."""
    p0, _, _ = _game_riccati_backward(contract)
    return float(contract.x0 @ p0 @ contract.x0)


def simulate_game_total_cost(
    contract: GameDataContract, control: np.ndarray
) -> float:
    """Simulate with the candidate's control and the worst-case disturbance."""
    _, _, disturbance_gains = _game_riccati_backward(contract)
    x = contract.x0.astype(np.float64)
    cost = 0.0
    for step in range(contract.horizon):
        u = control[step]
        w = disturbance_gains[step] @ x
        cost += float(
            x @ contract.Q @ x
            + u @ contract.R @ u
            - w @ contract.S @ w
        )
        x = contract.A @ x + contract.B @ u + contract.C @ w
    cost += float(x @ contract.QN @ x)
    return float(cost)
