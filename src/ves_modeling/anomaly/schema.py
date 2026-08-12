"""JSON-facing schema for the Anomaly API (R13)."""

from __future__ import annotations

from typing import Any

from ves_modeling.anomaly.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the supported Anomaly surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_anomaly_search",
            "apply_anomaly_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "predictions.json",
            "format": {
                "score": {"scores": ["number"]},
                "label": {
                    "labels": ["'normal'|'anomaly'|0|1"]
                },
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
        "verified_metrics": {
            "score": ["auroc", "average_precision"],
            "label": ["f1", "balanced_accuracy"],
        },
        "data_contract": {
            "label_column": {"default": "label", "customizable": True},
            "output_modes": ["score", "label"],
            "score_semantics": "higher score = more anomalous",
            "labels": (
                "'normal'/'anomaly' or 0/1 (1 = anomaly), at least one of "
                "each class, no mixed encodings"
            ),
            "host_labels": (
                "binary normal/anomaly or 0/1, at least one of each class "
                "(degenerate splits fail fast)"
            ),
        },
    }
