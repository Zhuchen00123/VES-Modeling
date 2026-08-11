"""JSON-facing schema for the Forecasting API (R8)."""

from __future__ import annotations

from typing import Any

from ves_modeling.forecasting.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the supported Forecasting surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_forecasting_search",
            "apply_forecasting_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "predictions.json",
            "format": {
                "key": [
                    {
                        "series_id": "scalar",
                        "timestamp": "strict ISO 8601 string",
                        "prediction": "number",
                    }
                ],
                "input": {"predictions": ["number"]},
            },
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; network none, read-only, "
                "cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": ["rmse", "mae", "smape"],
        "data_contract": {
            "time_column": {"default": "timestamp", "customizable": True},
            "series_id_column": {"default": "series_id", "customizable": True},
            "target_column": {"default": "target", "customizable": True},
            "frequency": {
                "default": "D",
                "note": "validated through pandas real offsets",
            },
            "row_order": ["input", "key"],
            "horizon": "uniform future steps per series",
            "exogenous": "optional feature columns derived from test_features.csv",
            "artifact_formats": {
                "input": {"predictions": ["number"]},
                "key": [
                    {
                        "series_id": "scalar",
                        "timestamp": "strict ISO 8601 string",
                        "prediction": "number",
                    }
                ],
            },
        },
    }
