"""JSON-facing schema for the Association API (R20)."""

from __future__ import annotations

from typing import Any

from ves_modeling.association.diagnostics import CANDIDATE_STATUSES

API_SCHEMA_VERSION = "1.0"


def capabilities() -> dict[str, Any]:
    """JSON-serializable declaration of the Association surface."""
    return {
        "api_schema_version": API_SCHEMA_VERSION,
        "operations": [
            "run_association_search",
            "apply_association_solution",
        ],
        "candidate_statuses": list(CANDIDATE_STATUSES),
        "apply_statuses": ["produced_unverified"],
        "artifact": {
            "filename": "rules.json",
            "format": {
                "rules": [
                    {
                        "antecedent": ["item", "..."],
                        "consequent": ["item", "..."],
                    }
                ]
            },
        },
        "trust_boundaries": {
            "local": "trusted code only (trusted_code=True)",
            "docker": (
                "default for untrusted/LLM code; mounts only train.csv; "
                "network none, read-only, cap-drop ALL, non-root"
            ),
        },
        "verified_metrics": [
            "mean_lift",
            "mean_confidence",
            "evaluable_rule_count",
            "rule_count",
        ],
        "data_contract": {
            "train_file": "train.csv (transaction long format)",
            "transaction_id_column": {
                "default": "transaction_id",
                "customizable": True,
            },
            "item_column": {"default": "item", "customizable": True},
            "hidden_file": (
                "hidden_test_transactions.csv (host-only, never mounted)"
            ),
            "lift_cap": {
                "default": "1e6",
                "customizable": True,
                "note": "clips lift to a finite value",
            },
            "evaluation": (
                "confidence/lift recomputed on hidden transactions; rules "
                "with absent antecedents are skipped"
            ),
        },
    }
