"""JSON-facing schema for the Regression API (R7.3 Batch A)."""

from __future__ import annotations

from typing import Any

from ves_modeling.regression.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the supported Regression surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": ["run_regression_search", "apply_regression_solution"],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "predictions.json",
            "format": {"predictions": ["number"]},
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; network none, read-only, "
                "cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": ["rmse", "mae"],
        "data_contract": {
            "target_column": {"default": "target", "customizable": True},
            "id_column": {"default": None},
            "row_order": ["input", "id"],
            "artifact_formats": {
                "input": {"predictions": ["number"]},
                "id": {
                    "predictions": [
                        {"id": "scalar", "prediction": "number"}
                    ]
                },
            },
        },
    }
