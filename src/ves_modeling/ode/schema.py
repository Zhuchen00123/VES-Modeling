"""JSON-facing schema for the ODE API (R11)."""

from __future__ import annotations

from typing import Any

from ves_modeling.ode.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the supported ODE surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": ["run_ode_search", "apply_ode_solution"],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "predictions.json",
            "format": {
                "input": {"predictions": ["number"]},
                "key": [
                    {
                        "trajectory_id": "scalar",
                        "t": "finite number",
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
        "verified_metrics": ["rmse", "mae"],
        "data_contract": {
            "time_column": {"default": "t", "customizable": True},
            "value_column": {"default": "y", "customizable": True},
            "trajectory_id_column": {
                "default": None,
                "customizable": True,
                "note": "per-trajectory t strictly increasing; "
                "single trajectory when absent",
            },
            "row_order": ["input", "key"],
            "min_rows": {
                "single_trajectory": 16,
                "per_trajectory": 8,
            },
            "artifact_formats": {
                "input": {"predictions": ["number"]},
                "key": [
                    {
                        "trajectory_id": "scalar",
                        "t": "finite number",
                        "prediction": "number",
                    }
                ],
            },
        },
    }
