"""R25: change-point data contract (series/indices/metrics)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.changepoint.data_contract import (
    load_hidden_changepoints,
    validate_changepoint_data,
    validate_changepoints,
)
from ves_modeling.changepoint.problem import build_changepoint_problem
from ves_modeling.changepoint.verifier import compute_changepoint_metrics


def _write_data(
    root: Path,
    *,
    train_rows: int = 40,
    test_rows: int = 60,
    cp_index: int = 30,
) -> tuple[Path, Path]:
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(
        {"t": [float(i) for i in range(train_rows)], "y": [0.0] * train_rows}
    ).to_csv(public / "train.csv", index=False)
    y = [0.0] * (test_rows // 2) + [10.0] * (test_rows - test_rows // 2)
    pd.DataFrame(
        {"t": [float(i) for i in range(test_rows)], "y": y}
    ).to_csv(public / "test_features.csv", index=False)
    pd.DataFrame({"changepoint": [cp_index]}).to_csv(
        host / "hidden_test_changepoints.csv", index=False
    )
    return public, host


def test_valid_contract(tmp_path: Path) -> None:
    public, host = _write_data(tmp_path / "d")
    contract = validate_changepoint_data(public)
    assert contract.train_rows == 40
    assert contract.test_rows == 60
    assert contract.tolerance_window == 3
    hidden = load_hidden_changepoints(host, contract)
    assert hidden.tolist() == [30]
    json.dumps(contract.to_dict())


def test_series_attacks(tmp_path: Path) -> None:
    public, _ = _write_data(tmp_path / "d")
    short = pd.read_csv(public / "train.csv").head(39)
    short.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="at least 40 rows"):
        validate_changepoint_data(public)

    public, _ = _write_data(tmp_path / "d2")
    frame = pd.read_csv(public / "test_features.csv")
    frame.loc[5, "t"] = frame.loc[4, "t"]
    frame.to_csv(public / "test_features.csv", index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_changepoint_data(public)

    public, _ = _write_data(tmp_path / "d3")
    frame = pd.read_csv(public / "train.csv")
    frame.loc[3, "y"] = float("nan")
    frame.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        validate_changepoint_data(public)

    public, _ = _write_data(tmp_path / "d4")
    frame = pd.read_csv(public / "train.csv")
    frame["extra"] = 0.0
    frame.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="exactly columns"):
        validate_changepoint_data(public)

    public, _ = _write_data(tmp_path / "d5", test_rows=2)
    with pytest.raises(ValueError, match="at least 3 rows"):
        validate_changepoint_data(public)


def test_hidden_validation(tmp_path: Path) -> None:
    public, host = _write_data(tmp_path / "d")
    contract = validate_changepoint_data(public)

    pd.DataFrame({"changepoint": []}).to_csv(
        host / "hidden_test_changepoints.csv", index=False
    )
    with pytest.raises(ValueError, match="non-empty"):
        load_hidden_changepoints(host, contract)

    pd.DataFrame({"changepoint": [0]}).to_csv(
        host / "hidden_test_changepoints.csv", index=False
    )
    with pytest.raises(ValueError, match=r"\[1, n-2\]"):
        load_hidden_changepoints(host, contract)

    pd.DataFrame({"changepoint": [30, 30]}).to_csv(
        host / "hidden_test_changepoints.csv", index=False
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        load_hidden_changepoints(host, contract)

    pd.DataFrame({"cp": [30]}).to_csv(
        host / "hidden_test_changepoints.csv", index=False
    )
    with pytest.raises(ValueError, match="exactly column"):
        load_hidden_changepoints(host, contract)


def test_artifact_validation(tmp_path: Path) -> None:
    public, _ = _write_data(tmp_path / "d")
    contract = validate_changepoint_data(public)
    n = contract.test_rows
    with pytest.raises(ValueError, match="missing required field"):
        validate_changepoints({}, n=n)
    with pytest.raises(ValueError, match="at least one index"):
        validate_changepoints({"changepoints": []}, n=n)
    with pytest.raises(ValueError, match="must be an integer"):
        validate_changepoints({"changepoints": [True]}, n=n)
    with pytest.raises(ValueError, match="must be an integer"):
        validate_changepoints({"changepoints": [30.0]}, n=n)
    with pytest.raises(ValueError, match=r"\[1, n-2\]"):
        validate_changepoints({"changepoints": [59]}, n=n)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_changepoints({"changepoints": [30, 30]}, n=n)
    result = validate_changepoints({"changepoints": [10, 30]}, n=n)
    assert result.tolist() == [10, 30]


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="changepoints.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_verifier_and_claims_ignored(tmp_path: Path) -> None:
    public, host = _write_data(tmp_path / "d")
    problem = build_changepoint_problem(public, host)
    payload = {
        "changepoints": [30],
        "claimed_f1": 0.0,
        "claimed_precision": 0.0,
        "claimed_recall": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["f1"] == 1.0
    assert values["precision"] == 1.0
    assert values["recall"] == 1.0
    assert values["mean_distance"] == 0.0


def test_metrics_greedy_matching() -> None:
    metrics = compute_changepoint_metrics(
        np.asarray([30]), np.asarray([30]), 3
    )
    assert metrics == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "mean_distance": 0.0,
    }
    metrics = compute_changepoint_metrics(
        np.asarray([28]), np.asarray([30]), 0
    )
    assert metrics["f1"] == 0.0
    assert metrics["mean_distance"] == 0.0
    metrics = compute_changepoint_metrics(
        np.asarray([10, 30, 31]), np.asarray([30]), 3
    )
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["mean_distance"] == 0.0
    metrics = compute_changepoint_metrics(
        np.asarray([28, 32]), np.asarray([30]), 3
    )
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["mean_distance"] == pytest.approx(2.0)
