"""Host-computed classification metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.classification.context import (
    ClassificationVerificationContext,
)
from ves_modeling.classification.data_contract import validate_predictions


class ClassificationVerifier:
    """EvidenceVerifier for class-label + probability artifacts.

    The host recomputes accuracy, macro F1, log loss, AUROC (binary or
    OVR-macro), multiclass Brier score, 10-bin calibration ECE and every
    confusion-matrix cell from hidden labels.  All metrics are finite or the
    verification raises.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, ClassificationVerificationContext):
            raise TypeError(
                "ClassificationVerifier requires "
                "ClassificationVerificationContext"
            )
        payload = self._parse(raw_artifact)
        predictions, probabilities = validate_predictions(
            payload,
            expected_count=context.expected_count,
            n_classes=context.n_classes,
            class_keys=context.class_keys,
            test_ids=(
                context.prediction_ids
                if context.row_order == "id"
                else None
            ),
            id_column=context.id_column,
        )
        labels = context.hidden_labels()
        observations = self._metrics(labels, predictions, probabilities)
        return Evidence(observations=tuple(observations))

    def _metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> list[Observation]:
        n_classes = probabilities.shape[1]
        accuracy = float(accuracy_score(labels, predictions))
        macro_f1 = float(
            f1_score(
                labels, predictions, average="macro", zero_division=0
            )
        )
        log_loss_value = float(log_loss(labels, probabilities))
        if n_classes == 2:
            # Binary: classes[1] is the positive class.
            auroc = float(roc_auc_score(labels, probabilities[:, 1]))
        else:
            auroc = float(
                roc_auc_score(
                    labels,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            )
        brier = float(
            np.mean((_one_hot(labels, n_classes) - probabilities) ** 2)
        )
        ece = _calibration_ece(labels, predictions, probabilities, n_bins=10)
        values: list[float] = [
            accuracy,
            macro_f1,
            log_loss_value,
            auroc,
            brier,
            ece,
        ]
        for value in values:
            if not np.isfinite(value):
                raise ValueError(
                    "classification metrics must be finite"
                )
        observations = [
            Observation(
                value=accuracy,
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="accuracy",
            ),
            Observation(
                value=macro_f1,
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="macro_f1",
            ),
            Observation(
                value=log_loss_value,
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="log_loss",
            ),
            Observation(
                value=auroc,
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="auroc",
            ),
            Observation(
                value=brier,
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="multiclass_brier",
            ),
            Observation(
                value=ece,
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="calibration_ece",
            ),
        ]
        for i in range(n_classes):
            for j in range(n_classes):
                count = float(
                    np.sum((labels == i) & (predictions == j))
                )
                observations.append(
                    Observation(
                        value=count,
                        uncertainty=0.0,
                        provenance="host:hidden-test",
                        name=f"confusion_{i}_{j}",
                    )
                )
        return observations

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


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((labels.size, n_classes), dtype=np.float64)
    matrix[np.arange(labels.size), labels] = 1.0
    return matrix


def _calibration_ece(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Expected calibration error over n equal-width confidence bins.

    ``confidence == 1.0`` falls into the last bin (``min(int(conf*n), n-1)``),
    so the final bin covers exactly 1.0 instead of losing it to an open edge.
    """
    confidences = probabilities.max(axis=1)
    bin_index = np.minimum(
        (confidences * n_bins).astype(np.int64), n_bins - 1
    )
    total = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        if not mask.any():
            continue
        accuracy = float(np.mean(labels[mask] == predictions[mask]))
        confidence = float(np.mean(confidences[mask]))
        total += mask.sum() * abs(accuracy - confidence)
    value = float(total / labels.size)
    if not np.isfinite(value):
        raise ValueError("calibration_ece must be finite")
    return value
