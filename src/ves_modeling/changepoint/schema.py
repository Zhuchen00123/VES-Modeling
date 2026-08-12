"""JSON-facing schema for the Change-point API (R25)."""

from __future__ import annotations

from typing import Any

from ves_modeling.changepoint.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Change-point surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_changepoint_search",
            "apply_changepoint_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "changepoints.json",
            "format": {"changepoints": ["index", "..."]},
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; mounts only train.csv and "
                "test_features.csv read-only; network none, read-only, "
                "cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": ["precision", "recall", "f1", "mean_distance"],
        "data_contract": {
            "train_csv": "t strictly increasing, y; n >= 40",
            "test_features_csv": "same shape; detection target",
            "changepoint_indices": "integers in [1, n-2]",
            "tolerance_window": {"default": 3, "customizable": True},
            "matching": "greedy one-to-one |detected - true| <= w",
            "optimality": "never claimed; host recomputes all metrics",
        },
    }
