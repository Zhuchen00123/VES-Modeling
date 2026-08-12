"""JSON-facing schema for the Probabilistic API (R18)."""

from __future__ import annotations

from typing import Any

from ves_modeling.probabilistic.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Probabilistic surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_probabilistic_search",
            "apply_probabilistic_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "solution.json",
            "format": {
                "estimate": "finite number",
                "confidence_interval": "[lo, hi] (optional)",
            },
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; mounts only problem.json "
                "and train.csv; network none, read-only, cap-drop ALL, "
                "non-root"
            ),
        },
        "verified_metrics": [
            "absolute_error",
            "relative_error",
            "ci_coverage",
        ],
        "data_contract": {
            "problem_file": "problem.json (public, read-only)",
            "train_file": "train.csv (n >= 20 samples)",
            "families": ["normal", "gamma", "beta"],
            "quantities": [
                "mean",
                "variance",
                "quantile",
                "probability_ge",
            ],
            "hidden_parameters": (
                "host-only true parameters in hidden_parameters.json; never "
                "exposed"
            ),
            "reference": "exact analytic value from hidden parameters",
        },
    }
