"""Host-computed regression metrics; candidate self-reports are never trusted."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.regression.context import RegressionVerificationContext


class RegressionVerifier:
    """EvidenceVerifier for tabular regression predictions.

    Responsibilities (idea.md #11):
    1. parse predictions.json; 2. predictions must be an array;
    3. count must exactly equal hidden labels; 4. reject bool; 5. reject NaN;
    6. reject Infinity; 7. reject non-numeric; 8. host computes RMSE and MAE.
    Candidate fields like ``claimed_rmse`` / ``claimed_mae`` / ``score`` are
    never read.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, RegressionVerificationContext):
            raise TypeError("RegressionVerifier requires RegressionVerificationContext")
        payload = self._parse(raw_artifact)
        predictions = self._validate_predictions(payload, context)
        labels = context.hidden_labels()
        rmse = float(np.sqrt(np.mean((predictions - labels) ** 2)))
        mae = float(np.mean(np.abs(predictions - labels)))
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

    @staticmethod
    def _validate_predictions(
        payload: dict[str, Any], context: RegressionVerificationContext
    ) -> np.ndarray:
        if "predictions" not in payload:
            raise ValueError("missing required field 'predictions'")
        raw = payload["predictions"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
            raise ValueError("'predictions' must be a JSON array")
        expected = context.expected_count
        if len(raw) != expected:
            raise ValueError(
                f"prediction count {len(raw)} != expected {expected}"
            )
        values: list[float] = []
        for item in raw:
            if isinstance(item, bool):
                raise ValueError("predictions must not contain booleans")
            if not isinstance(item, (int, float)):
                raise ValueError(
                    f"predictions must be numbers, got {type(item).__name__}"
                )
            value = float(item)
            if not math.isfinite(value):
                raise ValueError("predictions must be finite (no NaN/Infinity)")
            values.append(value)
        return np.asarray(values, dtype=np.float64)
