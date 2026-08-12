"""R17: recommendation data contract (ids/keys/ratings/ndcg)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.recommendation.context import (
    RecommendationVerificationContext,
)
from ves_modeling.recommendation.data_contract import (
    load_host_ratings,
    validate_predictions,
    validate_recommendation_data,
)
from ves_modeling.recommendation.problem import build_recommendation_problem
from ves_modeling.recommendation.verifier import RecommendationVerifier


def _make_recommendation_data(
    root: Path,
    *,
    n_users: int = 4,
    n_items: int = 6,
    train_per_user: int = 3,
    test_per_user: int = 2,
    numeric_ids: bool = False,
    host_reversed: bool = False,
    seed: int = 7,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    users = (
        [str(i) for i in range(1, n_users + 1)]
        if not numeric_ids
        else list(range(1, n_users + 1))
    )
    items = (
        [f"i{i}" for i in range(1, n_items + 1)]
        if not numeric_ids
        else list(range(1, n_items + 1))
    )
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    host_rows: list[dict] = []
    for user in users:
        base = float(rng.integers(1, 5))
        train_items = rng.choice(items, size=train_per_user, replace=False)
        for item in train_items:
            train_rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "rating": base + float(rng.integers(0, 2)),
                }
            )
        pool = [item for item in items if item not in set(train_items)]
        test_items = rng.choice(pool, size=test_per_user, replace=False)
        for item in test_items:
            rating = base + float(rng.integers(0, 3))
            test_rows.append({"user_id": user, "item_id": item})
            host_rows.append(
                {"user_id": user, "item_id": item, "rating": rating}
            )
    if host_reversed:
        host_rows = host_rows[::-1]
    pd.DataFrame(train_rows).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(
        public / "test_features.csv", index=False
    )
    pd.DataFrame(host_rows).to_csv(
        host / "hidden_test_ratings.csv", index=False
    )
    return public, host


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="predictions.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contract(tmp_path: Path) -> None:
    public, host = _make_recommendation_data(tmp_path / "data")
    contract = validate_recommendation_data(public)
    assert contract.n_users == 4
    assert contract.n_items >= 5
    assert contract.test_rows == 8
    assert len(contract.test_keys) == 8
    user_counts = Counter(key[0] for key in contract.test_keys)
    assert min(user_counts.values()) >= 1  # every test user has >=1 pair
    ratings = load_host_ratings(host, contract)
    assert ratings.shape == (8,)


def test_numeric_id_canonicalization(tmp_path: Path) -> None:
    public, host, = _make_recommendation_data(
        tmp_path / "data", numeric_ids=True
    )
    contract = validate_recommendation_data(public)
    keys = contract.test_keys
    assert all(user_key in {"1", "2", "3", "4"} for user_key, _ in keys)
    ratings = load_host_ratings(host, contract)
    assert ratings.shape == (8,)


def test_id_and_rating_attacks(tmp_path: Path) -> None:
    public, _host = _make_recommendation_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train.loc[0, "rating"] = float("nan")
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        validate_recommendation_data(public)
    train = pd.read_csv(public / "train.csv")
    train.loc[0, "rating"] = 3.0
    train = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="duplicate \\(user, item\\)"):
        validate_recommendation_data(public)


def test_host_alignment_reversed(tmp_path: Path) -> None:
    public, host = _make_recommendation_data(
        tmp_path / "data", host_reversed=True
    )
    contract = validate_recommendation_data(public)
    ratings = load_host_ratings(host, contract)
    assert ratings.shape == (8,)


def test_validate_predictions_input_and_key(tmp_path: Path) -> None:
    public, _host = _make_recommendation_data(tmp_path / "data")
    contract = validate_recommendation_data(public, row_order="input")
    values = list(np.arange(contract.test_rows, dtype=float))
    validate_predictions(
        {"predictions": values}, expected_count=contract.test_rows
    )
    with pytest.raises(ValueError, match="prediction count"):
        validate_predictions(
            {"predictions": values[:-1]},
            expected_count=contract.test_rows,
        )
    key_contract = validate_recommendation_data(public, row_order="key")
    test_keys = key_contract.test_keys
    rows = [
        {
            "user_id": user_key,
            "item_id": item_key,
            "prediction": 3.0,
        }
        for user_key, item_key in test_keys
    ]
    validate_predictions(
        {"predictions": rows},
        expected_count=len(test_keys),
        test_keys=test_keys,
        key_columns=("user_id", "item_id"),
    )
    with pytest.raises(ValueError, match="missing="):
        validate_predictions(
            {"predictions": rows[:-1]},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    with pytest.raises(ValueError, match="duplicate key"):
        validate_predictions(
            {"predictions": [rows[0], *rows]},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    with pytest.raises(ValueError, match="exactly"):
        validate_predictions(
            {"predictions": [{**rows[0], "extra": 1}, *rows[1:]]},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    with pytest.raises(ValueError, match="finite"):
        bad = [dict(row) for row in rows]
        bad[0]["prediction"] = float("nan")
        validate_predictions(
            {"predictions": bad},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )


def test_ndcg_audit_and_claims_ignored(tmp_path: Path) -> None:
    public = tmp_path / "public"
    host = tmp_path / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_id": ["u0", "u0", "u0", "u1"],
            "item_id": ["a", "b", "c", "d"],
            "rating": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(public / "train.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u0", "u0", "u0", "u1"],
            "item_id": ["a", "b", "c", "d"],
        }
    ).to_csv(public / "test_features.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u0", "u0", "u0", "u1"],
            "item_id": ["a", "b", "c", "d"],
            "rating": [5.0, 1.0, 4.0, 3.0],
        }
    ).to_csv(host / "hidden_test_ratings.csv", index=False)
    problem = build_recommendation_problem(public, host, row_order="key")
    # Predictions rank item b (true 1) first for u0 -> ndcg < 1.
    payload = {
        "predictions": [
            {"user_id": "u0", "item_id": "a", "prediction": 1.0},
            {"user_id": "u0", "item_id": "b", "prediction": 5.0},
            {"user_id": "u0", "item_id": "c", "prediction": 4.0},
            {"user_id": "u1", "item_id": "d", "prediction": 3.0},
        ],
        "claimed_rmse": 0.000001,
        "score": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["rmse"] > 0.0
    assert values["ndcg@5"] < 1.0
    assert values["ndcg@5"] > 0.0
    assert np.isfinite(values["mae"])


def test_verifier_perfect_ndcg(tmp_path: Path) -> None:
    public, host = _make_recommendation_data(tmp_path / "data")
    contract = validate_recommendation_data(public, row_order="key")
    ratings = load_host_ratings(host, contract)
    context = RecommendationVerificationContext(
        ratings,
        expected_count=int(ratings.size),
        user_keys=tuple(key[0] for key in contract.test_keys),
        item_keys=tuple(key[1] for key in contract.test_keys),
        row_order="key",
    )
    verifier = RecommendationVerifier()
    rows = [
        {
            "user_id": user_key,
            "item_id": item_key,
            "prediction": float(value),
        }
        for (user_key, item_key), value in zip(contract.test_keys, ratings)
    ]
    evidence = verifier.verify(_artifact({"predictions": rows}), context)
    values = {o.name: o.value for o in evidence.observations}
    assert values["rmse"] == pytest.approx(0.0)
    assert values["ndcg@5"] == pytest.approx(1.0)


def test_context_invariant() -> None:
    ratings = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="expected_count"):
        RecommendationVerificationContext(
            ratings, expected_count=2, row_order="input"
        )
    with pytest.raises(ValueError, match="user_keys and item_keys"):
        RecommendationVerificationContext(ratings, row_order="key")
    with pytest.raises(ValueError, match="only used when"):
        RecommendationVerificationContext(
            ratings,
            user_keys=("a",),
            item_keys=("b",),
            row_order="input",
        )
    with pytest.raises(ValueError, match="non-empty"):
        RecommendationVerificationContext(np.array([]))
    context = RecommendationVerificationContext(
        ratings,
        user_keys=("u0", "u0", "u1"),
        item_keys=("a", "b", "c"),
        row_order="key",
    )
    assert len(context.fingerprint()) == 64
