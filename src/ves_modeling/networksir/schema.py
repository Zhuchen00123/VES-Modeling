"""JSON-facing schema for the Network-SIR API (R30)."""

from __future__ import annotations

from typing import Any

from ves_modeling.networksir.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Network-SIR surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_networksir_search",
            "apply_networksir_solution",
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
            "model": "network_sir",
            "parameters": "beta > 0, gamma > 0, n_nodes in [10, 100]",
            "edges": "undirected, no self-loops, no duplicates",
            "initial_infected": "first i0 nodes (node ids 0..i0-1)",
            "time": "discrete unit steps; t_end/t round to whole steps",
            "quantities": ["final_size", "peak_infected", "infected_at"],
            "reference": (
                "host-held average over 2000 fixed-seed replications"
            ),
            "optimality": "never claimed; host recomputes all metrics",
        },
    }
