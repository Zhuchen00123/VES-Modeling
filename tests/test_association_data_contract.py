"""R20: association data contract (rules/confidence/lift)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.association.context import AssociationVerificationContext
from ves_modeling.association.data_contract import (
    load_hidden_transactions,
    validate_association_data,
    validate_rules,
)
from ves_modeling.association.problem import build_association_problem
from ves_modeling.association.verifier import AssociationVerifier


def _make_data(
    root: Path,
    *,
    numeric_items: bool = False,
    seed: int = 7,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    items = ["a", "b", "c", "d", "e"]
    train_rows: list[dict] = []
    hidden_rows: list[dict] = []
    for tid in range(1, 11):
        chosen = rng.choice(items, size=3, replace=False)
        for item in chosen:
            train_rows.append({"transaction_id": tid, "item": item})
    for tid in range(1, 8):
        chosen = rng.choice(items, size=2, replace=False)
        for item in chosen:
            hidden_rows.append({"transaction_id": tid, "item": item})
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(train_rows).to_csv(public / "train.csv", index=False)
    pd.DataFrame(hidden_rows).to_csv(
        host / "hidden_test_transactions.csv", index=False
    )
    return public, host


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="rules.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contract(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    contract = validate_association_data(public)
    assert contract.n_transactions == 10
    assert contract.n_items == 5
    hidden = load_hidden_transactions(host, contract)
    assert len(hidden) == 7


def test_long_table_attacks(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train.loc[0, "item"] = float("nan")
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="item keys"):
        validate_association_data(public)
    train = pd.read_csv(public / "train.csv")
    train.loc[0, "item"] = "a"
    train.loc[1, "transaction_id"] = float("nan")
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="empty transaction ids"):
        validate_association_data(public)
    train = pd.read_csv(public / "train.csv")
    train.loc[1, "transaction_id"] = 1
    train = train[train["transaction_id"] == 1].reset_index(drop=True)
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="at least two"):
        validate_association_data(public)


def test_validate_rules(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    contract = validate_association_data(public)
    rules = validate_rules(
        {
            "rules": [
                {"antecedent": ["a"], "consequent": ["b"]},
                {"antecedent": ["b", "a"], "consequent": ["c"]},
            ]
        },
        contract,
    )
    assert len(rules) == 2
    with pytest.raises(ValueError, match="at least one rule"):
        validate_rules({"rules": []}, contract)
    with pytest.raises(ValueError, match="missing required field"):
        validate_rules({}, contract)
    with pytest.raises(ValueError, match="disjoint"):
        validate_rules(
            {"rules": [{"antecedent": ["a"], "consequent": ["a"]}]},
            contract,
        )
    with pytest.raises(ValueError, match="outside the train item set"):
        validate_rules(
            {"rules": [{"antecedent": ["zz"], "consequent": ["a"]}]},
            contract,
        )
    with pytest.raises(ValueError, match="duplicates an earlier rule"):
        validate_rules(
            {
                "rules": [
                    {"antecedent": ["a"], "consequent": ["b"]},
                    {"antecedent": ["a"], "consequent": ["b"]},
                ]
            },
            contract,
        )


def test_verifier_metrics_and_claims_ignored(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    problem = build_association_problem(public, host)
    payload = {
        "rules": [
            {"antecedent": ["a"], "consequent": ["b"]},
            {"antecedent": ["c"], "consequent": ["d"]},
        ],
        "mean_lift": 0.0,
        "mean_confidence": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert np.isfinite(values["mean_lift"])
    assert np.isfinite(values["mean_confidence"])
    assert values["rule_count"] == 2.0
    assert values["evaluable_rule_count"] >= 1.0
    assert values["mean_lift"] >= 0.0


def test_verifier_skips_absent_antecedents(tmp_path: Path) -> None:
    public = tmp_path / "public"
    host = tmp_path / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(
        {
            "transaction_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "item": ["a", "b", "a", "b", "a", "b", "c", "d"],
        }
    ).to_csv(public / "train.csv", index=False)
    # Hidden transactions never contain item c.
    pd.DataFrame(
        {"transaction_id": [1, 1, 2, 2], "item": ["a", "b", "a", "b"]}
    ).to_csv(host / "hidden_test_transactions.csv", index=False)
    contract = validate_association_data(public)
    hidden = load_hidden_transactions(host, contract)
    context = AssociationVerificationContext(hidden, contract)
    verifier = AssociationVerifier()
    payload = {
        "rules": [
            {"antecedent": ["c"], "consequent": ["d"]},  # absent antecedent
            {"antecedent": ["a"], "consequent": ["b"]},  # evaluable
        ]
    }
    evidence = verifier.verify(_artifact(payload), context)
    values = {o.name: o.value for o in evidence.observations}
    assert values["rule_count"] == 2.0
    assert values["evaluable_rule_count"] == 1.0


def test_context_invariant(tmp_path: Path) -> None:
    from dataclasses import replace

    public, _host = _make_data(tmp_path / "data")
    contract = validate_association_data(public)
    with pytest.raises(ValueError, match="hidden transactions"):
        AssociationVerificationContext([], contract)
    with pytest.raises(ValueError, match="at least two train"):
        AssociationVerificationContext(
            [frozenset({"a"})], replace(contract, n_transactions=1)
        )
    with pytest.raises(ValueError, match="lift_cap"):
        AssociationVerificationContext(
            [frozenset({"a"})], contract, lift_cap=0.0
        )
    context = AssociationVerificationContext(
        [frozenset({"a", "b"})], contract
    )
    assert len(context.fingerprint()) == 64
