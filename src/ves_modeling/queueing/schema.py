"""JSON-facing schema for the Queueing API (R19)."""

from __future__ import annotations

from typing import Any

from ves_modeling.queueing.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Queueing surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": ["run_queueing_search", "apply_queueing_solution"],
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
            "kinds": ["mm1", "mmc"],
            "quantities": [
                "mean_wait",
                "mean_queue",
                "mean_utilization",
                "prob_wait_gt",
            ],
            "stability": "rho < 1 required",
            "reference": "host-only analytic M/M/1 / M/M/c Erlang-C value",
        },
    }
