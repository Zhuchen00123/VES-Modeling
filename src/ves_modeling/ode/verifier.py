"""Host-computed ODE metrics; candidate self-reports are never trusted."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.ode.context import OdeVerificationContext
from ves_modeling.ode.data_contract import validate_predictions


class OdeVerifier:
    """EvidenceVerifier for ODE prediction artifacts.

    The host aligns predictions to hidden values by ``(trajectory_id, t)``
    (key mode) or row order (input mode) and recomputes RMSE and MAE.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, OdeVerificationContext):
            raise TypeError("OdeVerifier requires OdeVerificationContext")
        payload = self._parse(raw_artifact)
        test_keys = (
            tuple(zip(context.trajectory_keys, context.time_keys))
            if context.row_order == "key"
            else None
        )
        predictions = validate_predictions(
            payload,
            expected_count=context.expected_count,
            test_keys=test_keys,
            key_columns=(
                context.trajectory_id_column,
                context.time_column,
            ),
        )
        values = context.hidden_values()
        rmse = float(np.sqrt(np.mean((predictions - values) ** 2)))
        mae = float(np.mean(np.abs(predictions - values)))
        if not np.isfinite(rmse) or not np.isfinite(mae):
            raise ValueError("ODE metrics must be finite")
        return Evidence(
            observations=(
                Observation(
                    value=rmse,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="rmse",
                ),
                Observation(
                    value=mae,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="mae",
                ),
            )
        )

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
