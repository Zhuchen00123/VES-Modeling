"""JSON-facing schema for the SIR API (R28)."""

from __future__ import annotations

from typing import Any

from ves_modeling.sir.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the SIR surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": ["run_sir_search", "apply_sir_solution"],
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
            "model": "sir",
            "parameters": "beta > 0, gamma > 0, N >= 100, i0 >= 1, r0 >= 0",
            "t_end": "> 0",
            "quantities": ["final_size", "peak_infected", "infected_at"],
            "reference": "host-held scipy RK45 solution, never public",
            "ci": "optional; lo <= estimate <= hi",
            "optimality": "never claimed; host recomputes all metrics",
        },
    }
