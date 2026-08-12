"""JSON-facing schema for the Cellular-automaton API (R29)."""

from __future__ import annotations

from typing import Any

from ves_modeling.cellular.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Cellular surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_cellular_search",
            "apply_cellular_solution",
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
        "verified_metrics": ["absolute_error", "relative_error"],
        "audit_observations": ["ci_coverage"],
        "data_contract": {
            "problem_file": "problem.json (public, read-only)",
            "rule": "elementary CA rule int in [0, 255]",
            "width": "int in [20, 200]",
            "steps": "int in [1, 200]",
            "initial": "exactly width binary values, at least one 1",
            "quantities": ["final_density", "cell_state", "persistent_ones"],
            "boundary": "periodic",
            "reference": "host-held exact deterministic iteration",
            "optimality": "never claimed; host recomputes all metrics",
        },
    }
