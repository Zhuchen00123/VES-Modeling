"""Host-computed survival metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.survival.context import SurvivalVerificationContext
from ves_modeling.survival.data_contract import validate_predictions


class SurvivalVerifier:
    """EvidenceVerifier for survival prediction artifacts.

    Risk-score mode reports Harrell C-index; time mode additionally reports
    MAE over uncensored test rows.  Candidate self-reported metrics are never
    read.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, SurvivalVerificationContext):
            raise TypeError(
                "SurvivalVerifier requires SurvivalVerificationContext"
            )
        payload = self._parse(raw_artifact)
        predictions = validate_predictions(
            payload,
            expected_count=context.expected_count,
            test_ids=(
                context.prediction_ids
                if context.row_order == "id"
                else None
            ),
            id_column=context.id_column,
        )
        times = context.hidden_times()
        events = context.hidden_events()
        if len(set(predictions.tolist())) < 2:
            raise ValueError(
                "c-index requires at least two distinct prediction scores"
            )
        c_index = _harrell_c_index(times, events, predictions)
        observations = [
            Observation(
                value=float(c_index),
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="c_index",
            )
        ]
        if context.output_kind == "time":
            uncensored = events == 1
            if not uncensored.any():
                raise ValueError("time mode requires at least one uncensored row")
            mae = float(
                np.mean(np.abs(predictions[uncensored] - times[uncensored]))
            )
            observations.append(
                Observation(
                    value=mae,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="mae",
                )
            )
        for observation in observations:
            if not np.isfinite(observation.value):
                raise ValueError("survival metrics must be finite")
        return Evidence(observations=tuple(observations))

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
            raise ValueError("predictions.json root must be an object")
        return data


def _harrell_c_index(
    times: np.ndarray, events: np.ndarray, predictions: np.ndarray
) -> float:
    concordant = 0.0
    allowable = 0
    n = times.size
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = times[i], times[j]
            ei, ej = events[i], events[j]
            if (ti < tj and ei == 1) or (ti == tj and ei == 1 and ej == 1) or (
                ti > tj and ej == 1
            ):
                allowable += 1
                if predictions[i] == predictions[j]:
                    concordant += 0.5
                elif (predictions[i] > predictions[j] and ti < tj) or (
                    predictions[i] < predictions[j] and ti > tj
                ):
                    concordant += 1.0
    if allowable == 0:
        return 0.5
    return float(concordant / allowable)
