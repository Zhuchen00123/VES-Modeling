"""JSON-facing schema for the Bin Packing API (R24)."""

from __future__ import annotations

from typing import Any

from ves_modeling.binpacking.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Bin Packing surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": ["run_binpacking_search", "apply_binpacking_solution"],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "solution.json",
            "format": {"assignment": ["bin index", "..."]},
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; mounts only problem.json "
                "read-only; network none, read-only, cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": ["bin_count", "capacity_violation"],
        "data_contract": {
            "problem_file": "problem.json (public, read-only)",
            "capacity": "finite > 0",
            "items": "finite positive sizes, each <= capacity",
            "contiguity": "used bins are 0..k-1 with no gaps",
            "optimality": "never claimed; host recomputes bin count only",
        },
    }
