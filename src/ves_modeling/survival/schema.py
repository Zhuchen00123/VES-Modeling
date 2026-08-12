"""JSON-facing schema for the Survival API (R21)."""

from __future__ import annotations

from typing import Any

from ves_modeling.survival.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Survival surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": ["run_survival_search", "apply_survival_solution"],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "predictions.json",
            "format": {
                "input": {"predictions": ["number"]},
                "id": [{"id": "scalar", "prediction": "number"}],
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
            "risk_score": ["c_index"],
            "time": ["c_index", "mae"],
        },
        "data_contract": {
            "time_column": {"default": "time", "customizable": True},
            "event_column": {"default": "event", "customizable": True},
            "output_kinds": ["risk_score", "time"],
            "row_order": ["input", "id"],
            "c_index": (
                "Harrell C-index on hidden outcomes; requires >=1 event "
                "and >=2 distinct scores"
            ),
            "mae": "time mode only; over uncensored test rows",
        },
    }
