"""Host-computed anomaly metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.anomaly.context import AnomalyVerificationContext
from ves_modeling.anomaly.data_contract import validate_predictions


class AnomalyVerifier:
    """EvidenceVerifier for anomaly score/label artifacts.

    Score mode: AUROC and Average Precision against hidden binary labels
    (higher score = more anomalous).  Label mode: F1 and balanced accuracy
    with ``anomaly``/``1`` as the positive class.  All metrics finite.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, AnomalyVerificationContext):
            raise TypeError(
                "AnomalyVerifier requires AnomalyVerificationContext"
            )
        payload = self._parse(raw_artifact)
        predictions = validate_predictions(
            payload,
            expected_count=context.expected_count,
            mode=context.output_mode,
        )
        labels = context.hidden_labels()
        if context.output_mode == "score":
            auroc = float(roc_auc_score(labels, predictions))
            average_precision = float(
                average_precision_score(labels, predictions)
            )
            metrics = (auroc, average_precision)
            names = ("auroc", "average_precision")
        else:
            f1 = float(f1_score(labels, predictions, zero_division=0))
            balanced_accuracy = float(
                balanced_accuracy_score(labels, predictions)
            )
            metrics = (f1, balanced_accuracy)
            names = ("f1", "balanced_accuracy")
        for value in metrics:
            if not np.isfinite(value):
                raise ValueError("anomaly metrics must be finite")
        return Evidence(
            observations=tuple(
                Observation(
                    value=value,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name=name,
                )
                for value, name in zip(metrics, names)
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
