"""JSON-facing schema for the Optimization API (R10)."""

from __future__ import annotations

from typing import Any

from ves_modeling.optimization.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the supported Optimization surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_optimization_search",
            "apply_optimization_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "solution.json",
            "format": {"variables": {"<name>": "finite number"}},
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; mounts only problem.json "
                "read-only; network none, read-only, cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": [
            "max_bound_violation",
            "max_constraint_violation",
            "integrality_violation",
            "objective",
        ],
        "data_contract": {
            "problem_file": "problem.json (public, read-only)",
            "sense": ["minimize", "maximize"],
            "variable_types": ["continuous", "integer", "binary"],
            "variables": "finite lower/upper bounds; binary is [0, 1]",
            "objective": "linear coefficients + optional constant",
            "constraints": ["<=", ">=", "=="],
            "tolerance": {"default": "1e-6", "customizable": True},
            "optimality": "never claimed without a host reference solver",
        },
    }
