"""JSON-facing schema for the Graph API (R14)."""

from __future__ import annotations

from typing import Any

from ves_modeling.graph.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the supported Graph surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": ["run_graph_search", "apply_graph_solution"],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "solution.json",
            "format": {
                "shortest_path": {"path": ["node", "..."]},
                "max_flow": {"flow": {"<u>-><v>": "finite number"}},
                "min_spanning_tree": {"edges": [["u", "v"], "..."]},
            },
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; mounts only problem.json "
                "read-only; network none, read-only, cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": {
            "shortest_path": ["total_weight", "path_violation"],
            "max_flow": [
                "total_value",
                "capacity_violation",
                "conservation_violation",
            ],
            "min_spanning_tree": ["total_weight", "tree_violation"],
        },
        "data_contract": {
            "problem_file": "problem.json (public, read-only)",
            "problem_types": [
                "shortest_path",
                "max_flow",
                "min_spanning_tree",
            ],
            "nodes": "unique labels, at least two",
            "edges": "finite-weight, no self-loops, no duplicates",
            "source/target": "required for shortest_path and max_flow",
            "tolerance": {"default": "1e-6", "customizable": True},
            "optimality": "never claimed without a host reference solver",
        },
    }
