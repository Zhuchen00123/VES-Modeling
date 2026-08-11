"""Host-computed forecasting metrics; candidate self-reports are never trusted."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.forecasting.context import ForecastingVerificationContext
from ves_modeling.forecasting.data_contract import validate_predictions


class ForecastingVerifier:
    """EvidenceVerifier for keyed time-series forecast artifacts.

    The host recomputes RMSE, MAE and sMAPE from hidden labels; candidate
    fields like ``claimed_rmse`` / ``score`` are never read.  All metrics are
    finite by construction and guarded (a non-finite result raises).
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, ForecastingVerificationContext):
            raise TypeError(
                "ForecastingVerifier requires ForecastingVerificationContext"
            )
        payload = self._parse(raw_artifact)
        test_keys = (
            tuple(zip(context.series_keys, context.time_keys))
            if context.row_order == "key"
            else None
        )
        predictions = validate_predictions(
            payload,
            expected_count=context.expected_count,
            test_keys=test_keys,
            key_columns=(context.series_id_column, context.time_column),
        )
        labels = context.hidden_labels()
        rmse = float(np.sqrt(np.mean((predictions - labels) ** 2)))
        mae = float(np.mean(np.abs(predictions - labels)))
        smape = _smape(predictions, labels)
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
                Observation(
                    value=smape,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="smape",
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


def _smape(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Symmetric MAPE (percent).  Both-zero pairs contribute 0."""
    denominator = np.abs(predictions) + np.abs(labels)
    numerator = 2.0 * np.abs(predictions - labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator == 0.0, 0.0, numerator / denominator)
    value = float(100.0 * np.mean(ratio))
    if not np.isfinite(value):
        raise ValueError("smape must be finite")
    return value
