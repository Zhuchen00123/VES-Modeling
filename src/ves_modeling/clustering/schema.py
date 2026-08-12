"""JSON-facing schema for the Clustering API (R12)."""

from __future__ import annotations

from typing import Any

from ves_modeling.clustering.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the supported Clustering surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_clustering_search",
            "apply_clustering_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "predictions.json",
            "format": {
                "input": {"labels": ["cluster label"]},
                "id": [{"id": "scalar", "label": "cluster label"}],
            },
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; mounts only train.csv and "
                "test_features.csv; network none, read-only, cap-drop ALL, "
                "non-root"
            ),
        },
        "verified_metrics": ["ari", "nmi", "v_measure", "silhouette"],
        "data_contract": {
            "label_column": {"default": "label", "customizable": True},
            "id_column": {"default": None},
            "row_order": ["input", "id"],
            "labels": (
                "non-empty strings or finite numbers; at least two distinct "
                "cluster labels; host reference covers all test rows with "
                "at least two classes"
            ),
            "metrics_note": (
                "ARI/NMI/V-measure are permutation-invariant; cluster names "
                "never need to match the reference"
            ),
            "artifact_formats": {
                "input": {"labels": ["cluster label"]},
                "id": [{"id": "scalar", "label": "cluster label"}],
            },
        },
    }
