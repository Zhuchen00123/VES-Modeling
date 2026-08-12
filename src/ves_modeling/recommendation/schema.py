"""JSON-facing schema for the Recommendation API (R17)."""

from __future__ import annotations

from typing import Any

from ves_modeling.recommendation.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Recommendation surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_recommendation_search",
            "apply_recommendation_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "predictions.json",
            "format": {
                "input": {"predictions": ["number"]},
                "key": [
                    {
                        "user_id": "scalar",
                        "item_id": "scalar",
                        "prediction": "number",
                    }
                ],
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
        "verified_metrics": ["rmse", "mae", "ndcg@5"],
        "data_contract": {
            "user_id_column": {"default": "user_id", "customizable": True},
            "item_id_column": {"default": "item_id", "customizable": True},
            "rating_column": {"default": "rating", "customizable": True},
            "row_order": ["input", "key"],
            "ids": "canonical string/finite-number keys (1 == 1.0 == '1')",
            "ndcg_at_5": (
                "audit only; per-user predicted-order DCG over ideal order, "
                "k=min(5,count); <2 items contributes 1.0; input mode has "
                "no user keys so ndcg@5 is always 1.0"
            ),
        },
    }
