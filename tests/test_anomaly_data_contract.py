"""R13: anomaly data contract (features/binary labels/score+label artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.anomaly.context import AnomalyVerificationContext
from ves_modeling.anomaly.data_contract import (
    load_host_labels,
    validate_anomaly_data,
    validate_predictions,
)
from ves_modeling.anomaly.problem import build_anomaly_problem
from ves_modeling.anomaly.verifier import AnomalyVerifier


def _make_anomaly_data(
    root: Path,
    *,
    n_train: int = 100,
    n_test_normal: int = 30,
    n_test_anomaly: int = 10,
    string_labels: bool = False,
    seed: int = 7,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    train = rng.normal(0.0, 1.0, size=(n_train, 4))
    test_normal = rng.normal(0.0, 1.0, size=(n_test_normal, 4))
    test_anomaly = rng.normal(5.0, 1.0, size=(n_test_anomaly, 4))
    test = np.vstack([test_normal, test_anomaly])
    if string_labels:
        labels = ["normal"] * n_test_normal + ["anomaly"] * n_test_anomaly
    else:
        labels = [0] * n_test_normal + [1] * n_test_anomaly
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(train, columns=[f"f{i}" for i in range(4)]).to_csv(
        public / "train.csv", index=False
    )
    pd.DataFrame(test, columns=[f"f{i}" for i in range(4)]).to_csv(
        public / "test_features.csv", index=False
    )
    pd.DataFrame({"label": labels}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    return public, host


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="predictions.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contract(tmp_path: Path) -> None:
    public, host = _make_anomaly_data(tmp_path / "data")
    contract = validate_anomaly_data(public)
    assert contract.test_rows == 40
    assert contract.train_rows == 100
    assert contract.input_columns == ("f0", "f1", "f2", "f3")
    labels = load_host_labels(host, contract)
    assert labels.shape == (40,)
    assert set(labels.tolist()) == {0, 1}


def test_string_and_numeric_host_labels(tmp_path: Path) -> None:
    public, host = _make_anomaly_data(
        tmp_path / "data", string_labels=True
    )
    contract = validate_anomaly_data(public)
    labels = load_host_labels(host, contract)
    assert set(labels.tolist()) == {0, 1}
    assert labels[30] == 1  # anomaly is positive


def test_host_label_attacks_rejected(tmp_path: Path) -> None:
    public, host = _make_anomaly_data(tmp_path / "data")
    contract = validate_anomaly_data(public)
    # degenerate split (single class)
    pd.DataFrame({"label": [0] * 40}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    with pytest.raises(ValueError, match="at least one of each class"):
        load_host_labels(host, contract)
    # numeric 1 written into a text column round-trips as the string "1",
    # which is not a valid binary class in CSV; JSON artifacts separately
    # exercise the must-not-mix rule.
    pd.DataFrame(
        {"label": ["normal"] * 20 + [1] * 20}
    ).to_csv(host / "hidden_test_labels.csv", index=False)
    with pytest.raises(ValueError, match="binary"):
        load_host_labels(host, contract)
    # invalid class value
    pd.DataFrame(
        {"label": ["normal"] * 20 + ["weird"] * 20}
    ).to_csv(host / "hidden_test_labels.csv", index=False)
    with pytest.raises(ValueError, match="binary"):
        load_host_labels(host, contract)
    # count mismatch
    pd.DataFrame({"label": [0] * 20 + [1] * 19}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    with pytest.raises(ValueError, match="hidden labels count"):
        load_host_labels(host, contract)


def test_feature_attacks_rejected(tmp_path: Path) -> None:
    public, _host = _make_anomaly_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train["bad"] = True
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="columns must match"):
        validate_anomaly_data(public)
    public2, _host2 = _make_anomaly_data(tmp_path / "data2")
    train2 = pd.read_csv(public2 / "train.csv")
    train2["f0"] = "x"
    train2.to_csv(public2 / "train.csv", index=False)
    with pytest.raises(ValueError, match="numeric"):
        validate_anomaly_data(public2)


def test_validate_predictions_score_mode(tmp_path: Path) -> None:
    public, _host = _make_anomaly_data(tmp_path / "data")
    contract = validate_anomaly_data(public)
    scores = [float(value) for value in range(contract.test_rows)]
    result = validate_predictions(
        {"scores": scores},
        expected_count=contract.test_rows,
        mode="score",
    )
    assert result.shape == (40,)
    with pytest.raises(ValueError, match="missing required field"):
        validate_predictions({}, expected_count=40, mode="score")
    with pytest.raises(ValueError, match="score count"):
        validate_predictions(
            {"scores": scores[:-1]}, expected_count=40, mode="score"
        )
    bad = [0.0] * 40
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_predictions({"scores": bad}, expected_count=40, mode="score")
    with pytest.raises(ValueError, match="numbers"):
        validate_predictions(
            {"scores": ["a"] * 40}, expected_count=40, mode="score"
        )


def test_validate_predictions_label_mode(tmp_path: Path) -> None:
    public, _host = _make_anomaly_data(tmp_path / "data")
    contract = validate_anomaly_data(public)
    labels = ["normal"] * 20 + ["anomaly"] * 20
    result = validate_predictions(
        {"labels": labels},
        expected_count=contract.test_rows,
        mode="label",
    )
    assert set(result.tolist()) == {0, 1}
    with pytest.raises(ValueError, match="missing required field"):
        validate_predictions({}, expected_count=40, mode="label")
    with pytest.raises(ValueError, match="at least one of each class"):
        validate_predictions(
            {"labels": ["normal"] * 40}, expected_count=40, mode="label"
        )
    with pytest.raises(ValueError, match="must not mix"):
        validate_predictions(
            {"labels": ["normal"] * 20 + [1] * 20},
            expected_count=40,
            mode="label",
        )


def test_score_mode_metrics_and_claims_ignored(tmp_path: Path) -> None:
    public, host = _make_anomaly_data(tmp_path / "data")
    problem = build_anomaly_problem(public, host, output_mode="score")
    # Higher scores for the last 10 rows (the anomalies) -> perfect AUROC.
    scores = [float(10 + index) for index in range(30)] + [
        float(100 + index) for index in range(10)
    ]
    payload = {
        "scores": scores,
        "claimed_auroc": 0.0,
        "claimed_average_precision": 0.0,
        "score": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["auroc"] == pytest.approx(1.0)
    assert values["average_precision"] == pytest.approx(1.0)


def test_label_mode_metrics_and_claims_ignored(tmp_path: Path) -> None:
    public, host = _make_anomaly_data(tmp_path / "data")
    problem = build_anomaly_problem(public, host, output_mode="label")
    payload = {
        "labels": ["normal"] * 30 + ["anomaly"] * 10,
        "claimed_f1": 0.0,
        "claimed_balanced_accuracy": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["f1"] == pytest.approx(1.0)
    assert values["balanced_accuracy"] == pytest.approx(1.0)


def test_verifier_finite_metrics(tmp_path: Path) -> None:
    public, host = _make_anomaly_data(tmp_path / "data")
    contract = validate_anomaly_data(public)
    hidden = load_host_labels(host, contract)
    context = AnomalyVerificationContext(
        hidden,
        expected_count=int(hidden.size),
        output_mode="label",
    )
    verifier = AnomalyVerifier()
    evidence = verifier.verify(
        _artifact({"labels": ["normal"] * 30 + ["anomaly"] * 10}),
        context,
    )
    values = {o.name: o.value for o in evidence.observations}
    assert np.isfinite(values["f1"])
    assert np.isfinite(values["balanced_accuracy"])


def test_context_invariant() -> None:
    labels = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="non-empty"):
        AnomalyVerificationContext(np.array([]))
    with pytest.raises(ValueError, match="at least one of each class"):
        AnomalyVerificationContext(np.array([0, 0, 0]))
    with pytest.raises(ValueError, match="expected_count"):
        AnomalyVerificationContext(labels, expected_count=2)
    with pytest.raises(ValueError, match="output_mode"):
        AnomalyVerificationContext(labels, output_mode="both")
    context = AnomalyVerificationContext(labels)
    assert context.expected_count == 3
    assert context.output_mode == "score"
    assert len(context.fingerprint()) == 64
