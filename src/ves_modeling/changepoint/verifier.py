"""Host-computed change-point metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.changepoint.context import ChangepointVerificationContext
from ves_modeling.changepoint.data_contract import validate_changepoints


class ChangepointVerifier:
    """EvidenceVerifier for change-point detection artifacts."""

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, ChangepointVerificationContext):
            raise TypeError(
                "ChangepointVerifier requires "
                "ChangepointVerificationContext"
            )
        payload = self._parse(raw_artifact)
        if context.n is None:
            raise ValueError("context n is required for verification")
        detected = validate_changepoints(payload, n=context.n)
        metrics = compute_changepoint_metrics(
            detected,
            context.hidden_changepoints(),
            context.tolerance_window,
        )
        observations = (
            Observation(
                value=metrics["precision"],
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="precision",
            ),
            Observation(
                value=metrics["recall"],
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="recall",
            ),
            Observation(
                value=metrics["f1"],
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="f1",
            ),
            Observation(
                value=metrics["mean_distance"],
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="mean_distance",
            ),
        )
        for observation in observations:
            if not np.isfinite(observation.value):
                raise ValueError("change-point metrics must be finite")
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
            raise ValueError("changepoints.json root must be an object")
        return data


def compute_changepoint_metrics(
    detected: np.ndarray,
    true: np.ndarray,
    tolerance_window: int,
) -> dict[str, float]:
    """Greedy one-to-one matching within a tolerance window."""
    detected = np.asarray(detected, dtype=np.int64)
    true = np.asarray(true, dtype=np.int64)
    matches: list[int] = []
    used = [False] * int(true.size)
    for raw_detected in detected:
        detected_index = int(raw_detected)
        best_index: int | None = None
        best_distance: int | None = None
        for index, raw_true in enumerate(true):
            if used[index]:
                continue
            distance = abs(int(raw_true) - detected_index)
            if distance <= tolerance_window and (
                best_distance is None or distance < best_distance
            ):
                best_index = index
                best_distance = distance
        if best_index is not None:
            used[best_index] = True
            assert best_distance is not None
            matches.append(best_distance)
    matched = len(matches)
    precision = matched / detected.size if detected.size else 0.0
    recall = matched / true.size if true.size else 0.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    mean_distance = float(np.mean(matches)) if matches else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_distance": mean_distance,
    }
