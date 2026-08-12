"""R21: survival data contract (outcomes/c-index/MAE)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.survival.context import SurvivalVerificationContext
from ves_modeling.survival.data_contract import (
    load_hidden_outcomes,
    validate_predictions,
    validate_survival_data,
)
from ves_modeling.survival.problem import build_survival_problem


def _make_data(
    root: Path,
    *,
    output_kind: str = "risk_score",
    id_col: str | None = None,
    seed: int = 7,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    n = 60
    x = rng.normal(size=(n, 2))
    times = 15.0 + 8.0 * np.abs(x[:, 0]) + rng.exponential(
        scale=5.0, size=n
    )
    events = (rng.random(n) < 0.6).astype(int)
    test = pd.DataFrame(x[40:], columns=["f0", "f1"])
    train = pd.DataFrame(x[:40], columns=["f0", "f1"])
    train["time"] = times[:40]
    train["event"] = events[:40]
    hidden = pd.DataFrame(
        {"time": times[40:], "event": events[40:]}
    )
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    if id_col:
        train[id_col] = np.arange(1, len(train) + 1)
        test[id_col] = np.arange(1, len(test) + 1)
        hidden[id_col] = np.arange(1, len(test) + 1)
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test_features.csv", index=False)
    hidden.to_csv(host / "hidden_test_outcomes.csv", index=False)
    return public, host


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="predictions.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contract(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    contract = validate_survival_data(public)
    assert contract.test_rows == 20
    assert contract.feature_columns == ("f0", "f1")
    times, events = load_hidden_outcomes(host, contract)
    assert times.shape == (20,)
    assert set(events.tolist()) <= {0, 1}
    assert int(events.sum()) >= 1


def test_id_mode_alignment(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data", id_col="id_col")
    contract = validate_survival_data(public, id_column="id_col", row_order="id")
    times, _events = load_hidden_outcomes(host, contract)
    assert times.shape == (20,)
    assert contract.test_ids is not None


def test_schema_attacks(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train.loc[0, "time"] = 0.0
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="positive"):
        validate_survival_data(public)
    train = pd.read_csv(public / "train.csv")
    train.loc[0, "time"] = 5.0
    train.loc[0, "event"] = 2
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="0/1"):
        validate_survival_data(public)
    train = pd.read_csv(public / "train.csv")
    train.loc[0, "event"] = 1
    train["bad"] = True
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="match train features exactly"):
        validate_survival_data(public)
    with pytest.raises(ValueError, match="output_kind"):
        validate_survival_data(public, output_kind="surv")


def test_hidden_outcome_attacks(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    hidden = pd.read_csv(host / "hidden_test_outcomes.csv")
    hidden.loc[0, "event"] = 0
    hidden = hidden[hidden["event"] == 0].reset_index(drop=True)
    hidden.to_csv(host / "hidden_test_outcomes.csv", index=False)
    contract = validate_survival_data(public)
    with pytest.raises(ValueError, match="at least one event"):
        load_hidden_outcomes(host, contract)


def test_validate_predictions(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    contract = validate_survival_data(public)
    values = list(np.arange(contract.test_rows, dtype=float))
    validate_predictions(
        {"predictions": values}, expected_count=contract.test_rows
    )
    with pytest.raises(ValueError, match="prediction count"):
        validate_predictions(
            {"predictions": values[:-1]},
            expected_count=contract.test_rows,
        )
    bad = [0.0] * contract.test_rows
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(
            {"predictions": bad}, expected_count=contract.test_rows
        )


def test_c_index_and_claims_ignored(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    problem = build_survival_problem(public, host)
    times, _events = load_hidden_outcomes(
        host, validate_survival_data(public)
    )
    # Perfect ordering: risk score = -time (higher risk = shorter time).
    scores = [float(-value) for value in times]
    payload = {"predictions": scores, "c_index": 0.0, "score": 0.0}
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["c_index"] == pytest.approx(1.0)
    assert "mae" not in values


def test_time_mode_mae(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data", output_kind="time")
    problem = build_survival_problem(public, host, output_kind="time")
    times, events = load_hidden_outcomes(
        host, validate_survival_data(public, output_kind="time")
    )
    payload = {"predictions": [float(value) + 1.0 for value in times]}
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert "c_index" in values
    uncensored = events == 1
    assert values["mae"] == pytest.approx(1.0)
    assert uncensored.any()


def test_c_index_requires_two_distinct_scores(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    problem = build_survival_problem(public, host)
    payload = {"predictions": [0.5] * 20}
    verification = VerificationPipeline(problem).verify(_artifact(payload))
    assert verification.status.value == "verification_failed"


def test_context_invariant(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    times, events = load_hidden_outcomes(
        host, validate_survival_data(public)
    )
    with pytest.raises(ValueError, match="at least one event"):
        SurvivalVerificationContext(times, np.zeros_like(events))
    with pytest.raises(ValueError, match="finite and positive"):
        SurvivalVerificationContext(
            np.array([-1.0, 2.0]), np.array([1, 0])
        )
    context = SurvivalVerificationContext(times, events)
    assert len(context.fingerprint()) == 64
