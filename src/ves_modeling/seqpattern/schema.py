"""JSON-facing schema for the Sequential-pattern API (R27)."""

from __future__ import annotations

from typing import Any

from ves_modeling.seqpattern.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Sequential-pattern surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_seqpattern_search",
            "apply_seqpattern_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "patterns.json",
            "format": {
                "patterns": [
                    {"prefix": ["event", "..."], "suffix": ["event", "..."]}
                ]
            },
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; mounts only train.csv "
                "read-only; network none, read-only, cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": ["mean_lift", "mean_confidence"],
        "evaluation_observations": [
            "evaluable_pattern_count",
            "pattern_count",
        ],
        "data_contract": {
            "train_csv": "sequence_id, step, event; >= 10 sequences",
            "min_steps_per_sequence": 3,
            "hidden": "hidden_test_sequences.csv (never mounted)",
            "matching": "contiguous prefix/suffix on hidden sequences",
            "lift_cap": 1_000_000.0,
            "optimality": "never claimed; host recomputes all metrics",
        },
    }
