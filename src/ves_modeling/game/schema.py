"""JSON-facing schema for the Game API (R31)."""

from __future__ import annotations

from typing import Any

from ves_modeling.game.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Game surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": ["run_game_search", "apply_game_solution"],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "solution.json",
            "format": {"control": ["u_0", "...", "u_{N-1}"]},
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; mounts only problem.json "
                "read-only; network none, read-only, cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": ["total_cost"],
        "audit_observations": ["reference_optimal_cost"],
        "data_contract": {
            "problem_file": "problem.json (public, read-only)",
            "matrices": (
                "A n x n, B n x m, C n x p, Q PSD, R PD, S PD, Q_N PSD, x0"
            ),
            "horizon": "int >= 2",
            "terminal_weight": "defaults to Q",
            "worst_case": (
                "disturbance from the game-Riccati minimax strategy"
            ),
            "optimality": (
                "never claimed; reference optimal cost is audit-only"
            ),
        },
    }
