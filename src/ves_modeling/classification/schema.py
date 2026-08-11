"""JSON-facing schema for the Classification API (R9)."""

from __future__ import annotations

from typing import Any

from ves_modeling.classification.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the supported Classification surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_classification_search",
            "apply_classification_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "predictions.json",
            "format": {
                "input": [
                    {
                        "label": "class scalar",
                        "probabilities": ["number"],
                    }
                ],
                "id": [
                    {
                        "id": "scalar",
                        "label": "class scalar",
                        "probabilities": ["number"],
                    }
                ],
            },
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; network none, read-only, "
                "cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": [
            "accuracy",
            "macro_f1",
            "log_loss",
            "auroc",
            "multiclass_brier",
            "calibration_ece",
            "confusion_*",
        ],
        "data_contract": {
            "label_column": {"default": "target", "customizable": True},
            "id_column": {"default": None},
            "row_order": ["input", "id"],
            "classes": (
                "host-fixed order: explicit classes or train first "
                "appearance; >=2 unique; every declared class must appear "
                "in train and host must cover all classes"
            ),
            "probabilities": {
                "shape": "n_classes per row",
                "range": "[0, 1]",
                "sum": "1 within 1e-6",
                "label": "class-order argmax, ties first",
            },
            "artifact_formats": {
                "input": [
                    {
                        "label": "class scalar",
                        "probabilities": ["number"],
                    }
                ],
                "id": [
                    {
                        "id": "scalar",
                        "label": "class scalar",
                        "probabilities": ["number"],
                    }
                ],
            },
        },
    }
