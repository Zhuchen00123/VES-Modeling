"""JSON-facing schema for the Assignment/TSP API (R22)."""

from __future__ import annotations

from typing import Any

from ves_modeling.assignment.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Assignment/TSP surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": ["run_assignment_search", "apply_assignment_solution"],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "solution.json",
            "format": {
                "assignment": {"assignment": ["permutation"]},
                "tsp": {"tour": ["permutation starting at start"]},
            },
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; mounts only problem.json "
                "read-only; network none, read-only, cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": ["total_cost"],
        "data_contract": {
            "problem_file": "problem.json (public, read-only)",
            "problem_types": ["assignment", "tsp"],
            "size": "n >= 3",
            "costs": (
                "assignment n x n finite; tsp n x n symmetric zero diagonal"
            ),
            "start": "tsp only, default 0",
            "optimality": "never claimed; host recomputes total cost only",
        },
    }
