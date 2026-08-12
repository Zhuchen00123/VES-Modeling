"""Host-computed clustering metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.clustering.context import ClusteringVerificationContext
from ves_modeling.clustering.data_contract import validate_predictions


class ClusteringVerifier:
    """EvidenceVerifier for clustering label artifacts.

    ARI, NMI and V-measure compare candidate labels against the hidden
    reference partition and are permutation-invariant (cluster names never
    need to match).  Silhouette is an internal metric computed from the
    public test features and candidate labels when computable (else 0.0).
    All metrics are finite or the verification raises.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, ClusteringVerificationContext):
            raise TypeError(
                "ClusteringVerifier requires ClusteringVerificationContext"
            )
        payload = self._parse(raw_artifact)
        candidate_labels = validate_predictions(
            payload,
            expected_count=context.expected_count,
            test_ids=(
                context.prediction_ids
                if context.row_order == "id"
                else None
            ),
            id_column=context.id_column,
        )
        host_labels = list(context.hidden_labels)
        ari = float(adjusted_rand_score(host_labels, candidate_labels))
        nmi = float(
            normalized_mutual_info_score(host_labels, candidate_labels)
        )
        v_measure = float(v_measure_score(host_labels, candidate_labels))
        silhouette = _silhouette_or_zero(
            context.test_features, candidate_labels
        )
        metrics = (ari, nmi, v_measure, silhouette)
        for value in metrics:
            if not np.isfinite(value):
                raise ValueError("clustering metrics must be finite")
        return Evidence(
            observations=(
                Observation(
                    value=ari,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="ari",
                ),
                Observation(
                    value=nmi,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="nmi",
                ),
                Observation(
                    value=v_measure,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="v_measure",
                ),
                Observation(
                    value=silhouette,
                    uncertainty=0.0,
                    provenance="host:test-features",
                    name="silhouette",
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


def _silhouette_or_zero(
    features: np.ndarray | None, labels: list[str]
) -> float:
    if features is None:
        return 0.0
    n_clusters = len(set(labels))
    if n_clusters < 2 or features.shape[0] <= n_clusters:
        return 0.0
    try:
        return float(silhouette_score(features, labels))
    except (ValueError, TypeError):
        return 0.0
