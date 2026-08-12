"""R12: clustering data contract (features/labels/id/artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.clustering.context import ClusteringVerificationContext
from ves_modeling.clustering.data_contract import (
    load_host_labels,
    validate_clustering_data,
    validate_predictions,
)
from ves_modeling.clustering.problem import build_clustering_problem
from ves_modeling.clustering.verifier import ClusteringVerifier


def _make_clustering_data(
    root: Path,
    *,
    n_clusters: int = 3,
    samples_per_cluster: int = 40,
    test_per_cluster: int = 10,
    id_col: str | None = None,
    host_order: str = "same",
    string_labels: bool = False,
    seed: int = 7,
) -> tuple[Path, Path]:
    X, y = make_blobs(
        n_samples=(samples_per_cluster + test_per_cluster) * n_clusters,
        centers=n_clusters,
        n_features=4,
        cluster_std=1.0,
        random_state=seed,
    )
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cluster in range(n_clusters):
        indices = np.where(y == cluster)[0]
        train_idx.extend(indices[:samples_per_cluster].tolist())
        test_idx.extend(indices[samples_per_cluster:].tolist())
    train = pd.DataFrame(X[train_idx], columns=[f"f{i}" for i in range(4)])
    test = pd.DataFrame(X[test_idx], columns=[f"f{i}" for i in range(4)])
    if string_labels:
        labels = [f"c{int(value)}" for value in y[test_idx]]
    else:
        labels = [int(value) for value in y[test_idx]]
    host_frame = pd.DataFrame({"label": labels})
    if id_col:
        train[id_col] = np.arange(1, len(train) + 1)
        test[id_col] = np.arange(1, len(test) + 1)
        host_frame[id_col] = np.arange(1, len(test) + 1)
        if host_order == "reversed":
            host_frame = host_frame.iloc[::-1].reset_index(drop=True)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test_features.csv", index=False)
    host_frame.to_csv(host / "hidden_test_labels.csv", index=False)
    return public, host


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="predictions.json",
        content=json.dumps(payload),
        producer="test",
    )


def _input_payload(n_rows: int, n_clusters: int = 3) -> dict:
    return {
        "labels": [
            f"cluster_{index % n_clusters}" for index in range(n_rows)
        ]
    }


def test_valid_input_contract(tmp_path: Path) -> None:
    public, host = _make_clustering_data(tmp_path / "data")
    contract = validate_clustering_data(public)
    assert contract.test_rows == 30
    assert contract.train_rows == 120
    assert contract.input_columns == ("f0", "f1", "f2", "f3")
    host_keys, distinct = load_host_labels(host, contract)
    assert len(host_keys) == 30
    assert len(distinct) == 3


def test_valid_id_mode_alignment(tmp_path: Path) -> None:
    public, host = _make_clustering_data(
        tmp_path / "data", id_col="id_col", host_order="reversed"
    )
    contract = validate_clustering_data(
        public, id_column="id_col", row_order="id"
    )
    host_keys, _distinct = load_host_labels(host, contract)
    assert len(host_keys) == 30
    assert contract.test_ids is not None


def test_feature_attacks_rejected(tmp_path: Path) -> None:
    public, _host = _make_clustering_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train["bad"] = True
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="columns must match"):
        validate_clustering_data(public)
    # string column
    public2, _host2 = _make_clustering_data(tmp_path / "data2")
    train2 = pd.read_csv(public2 / "train.csv")
    train2["f0"] = "x"
    train2.to_csv(public2 / "train.csv", index=False)
    with pytest.raises(ValueError, match="numeric"):
        validate_clustering_data(public2)
    # nan feature
    public3, _host3 = _make_clustering_data(tmp_path / "data3")
    train3 = pd.read_csv(public3 / "train.csv")
    train3.loc[0, "f0"] = float("nan")
    train3.to_csv(public3 / "train.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        validate_clustering_data(public3)


def test_host_label_attacks_rejected(tmp_path: Path) -> None:
    public, host = _make_clustering_data(tmp_path / "data")
    frame = pd.read_csv(host / "hidden_test_labels.csv")
    frame.loc[0, "label"] = float("nan")
    frame.to_csv(host / "hidden_test_labels.csv", index=False)
    contract = validate_clustering_data(public)
    with pytest.raises(ValueError, match="nulls"):
        load_host_labels(host, contract)
    # fewer than two distinct host labels
    frame = pd.DataFrame({"label": [0] * 30})
    frame.to_csv(host / "hidden_test_labels.csv", index=False)
    with pytest.raises(ValueError, match="at least two distinct"):
        load_host_labels(host, contract)
    # count mismatch
    frame = pd.DataFrame({"label": [0] * 15 + [1] * 14})
    frame.to_csv(host / "hidden_test_labels.csv", index=False)
    with pytest.raises(ValueError, match="hidden labels count"):
        load_host_labels(host, contract)


def test_validate_predictions_input_mode(tmp_path: Path) -> None:
    public, _host = _make_clustering_data(tmp_path / "data")
    contract = validate_clustering_data(public)
    good = _input_payload(30)
    keys = validate_predictions(good, expected_count=contract.test_rows)
    assert len(keys) == 30
    with pytest.raises(ValueError, match="missing required field"):
        validate_predictions({}, expected_count=30)
    with pytest.raises(ValueError, match="label count"):
        validate_predictions(_input_payload(29), expected_count=30)
    with pytest.raises(ValueError, match="at least two distinct"):
        validate_predictions(
            {"labels": ["c0"] * 30}, expected_count=30
        )
    with pytest.raises(ValueError, match="class labels"):
        bad = {"labels": ["c0", True] + ["c0"] * 28}
        validate_predictions(bad, expected_count=30)


def test_validate_predictions_id_mode_attacks(tmp_path: Path) -> None:
    public, _host = _make_clustering_data(
        tmp_path / "data", id_col="id_col"
    )
    contract = validate_clustering_data(
        public, id_column="id_col", row_order="id"
    )
    test_ids = contract.test_ids
    assert test_ids is not None
    rows = [
        {"id": test_id, "label": f"c{index % 3}"}
        for index, test_id in enumerate(test_ids)
    ]
    keys = validate_predictions(
        {"predictions": rows},
        expected_count=30,
        test_ids=test_ids,
        id_column="id_col",
    )
    assert len(keys) == 30
    with pytest.raises(ValueError, match="missing="):
        validate_predictions(
            {"predictions": rows[:-1]},
            expected_count=30,
            test_ids=test_ids,
        )
    with pytest.raises(ValueError, match="duplicate id"):
        validate_predictions(
            {"predictions": [rows[0], *rows]},
            expected_count=30,
            test_ids=test_ids,
        )
    with pytest.raises(ValueError, match="exactly"):
        validate_predictions(
            {"predictions": [{**rows[0], "extra": 1}, *rows[1:]]},
            expected_count=30,
            test_ids=test_ids,
        )
    bad_id = [dict(row) for row in rows]
    bad_id[0]["id"] = True
    with pytest.raises(ValueError, match="id must"):
        validate_predictions(
            {"predictions": bad_id},
            expected_count=30,
            test_ids=test_ids,
        )


def test_permutation_invariance_and_claims_ignored(tmp_path: Path) -> None:
    public, host = _make_clustering_data(
        tmp_path / "data", string_labels=True
    )
    problem = build_clustering_problem(public, host)
    host_labels = pd.read_csv(host / "hidden_test_labels.csv")[
        "label"
    ].tolist()
    # Permuted cluster names (renamed c0<->c1) must score perfectly.
    permuted = {
        "labels": [
            "c1" if value == "c0" else "c0" if value == "c1" else value
            for value in host_labels
        ],
        "claimed_ari": -1.0,
        "claimed_nmi": -1.0,
        "score": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(permuted))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["ari"] == pytest.approx(1.0)
    assert values["nmi"] == pytest.approx(1.0)
    assert values["v_measure"] == pytest.approx(1.0)
    assert np.isfinite(values["silhouette"])


def test_context_invariant() -> None:
    labels = ["a", "a", "b"]
    with pytest.raises(ValueError, match="non-empty"):
        ClusteringVerificationContext([])
    with pytest.raises(ValueError, match="at least two distinct"):
        ClusteringVerificationContext(["a", "a", "a"])
    with pytest.raises(ValueError, match="expected_count"):
        ClusteringVerificationContext(labels, expected_count=2)
    with pytest.raises(ValueError, match="prediction_ids"):
        ClusteringVerificationContext(
            labels, row_order="id", id_column="id_col"
        )
    with pytest.raises(ValueError, match="only used when"):
        ClusteringVerificationContext(
            labels, prediction_ids=("1", "2", "3"), row_order="input"
        )
    context = ClusteringVerificationContext(
        labels,
        row_order="id",
        id_column="id_col",
        prediction_ids=("1", "2", "3"),
    )
    assert context.expected_count == 3
    assert len(context.fingerprint()) == 64


def test_verifier_silhouette_fallback() -> None:
    context = ClusteringVerificationContext(
        ["a", "a", "b"], test_features=None
    )
    verifier = ClusteringVerifier()
    evidence = verifier.verify(_artifact({"labels": ["a", "a", "b"]}), context)
    values = {o.name: o.value for o in evidence.observations}
    assert values["silhouette"] == 0.0
    assert np.isfinite(values["ari"])
