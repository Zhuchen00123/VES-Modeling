"""JSON-facing schema for the Multi-objective API (R16)."""

from __future__ import annotations

from typing import Any

from ves_modeling.multiobjective.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Multi-objective surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_multiobjective_search",
            "apply_multiobjective_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "solution.json",
            "format": {
                "solutions": [{"variables": {"<name>": "finite number"}}]
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
            "hypervolume",
            "non_dominated_count",
            "feasible_count",
            "total_count",
        ],
        "data_contract": {
            "problem_file": "problem.json (public, read-only)",
            "objectives": "exactly two linear objectives",
            "variables": [
                "continuous",
                "integer",
                "binary",
                "finite bounds",
            ],
            "constraints": ["<=", ">=", "=="],
            "reference_point": (
                "optional [r1, r2]; defaults to host objective box upper "
                "bounds + 1"
            ),
            "tolerance": {"default": "1e-6", "customizable": True},
            "optimality": (
                "never claimed; hypervolume is a relative quality measure "
                "of the delivered solution set"
            ),
        },
    }
