"""JSON-facing schema for the Monte Carlo API (R15)."""

from __future__ import annotations

from typing import Any

from ves_modeling.montecarlo.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the supported Monte Carlo surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_montecarlo_search",
            "apply_montecarlo_solution",
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
                "read-only; network none, read-only, cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": [
            "absolute_error",
            "relative_error",
            "ci_coverage",
        ],
        "data_contract": {
            "problem_file": "problem.json (public, read-only)",
            "kinds": ["expectation", "integral", "probability"],
            "expectation": {
                "targets": [
                    "mean",
                    "second_moment",
                    "variance",
                    "prob_ge",
                    "prob_le",
                ]
            },
            "integral": "polynomial coefficients over [a, b]",
            "probability": "binomial n/p with ge|le|eq event",
            "reference": "host-only analytic value; never exposed",
        },
    }
