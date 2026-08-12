"""JSON-facing schema for the Markov API (R23)."""

from __future__ import annotations

from typing import Any

from ves_modeling.markov.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Markov surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": ["run_markov_search", "apply_markov_solution"],
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
            "train_file": "train.csv (>= 50 rows)",
            "quantities": [
                "transition_probability",
                "steady_state",
                "expected_recurrence_time",
            ],
            "hidden_parameters": (
                "host-only true transition matrix; rows sum to 1 and the "
                "chain is irreducible"
            ),
            "reference": "exact value from the hidden matrix",
        },
    }
