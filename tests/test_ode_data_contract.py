"""R11: ODE data contract (trajectories/t/keys/artifact)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.ode.context import OdeVerificationContext
from ves_modeling.ode.data_contract import (
    _trajectory_key,
    load_host_values,
    validate_ode_data,
    validate_predictions,
)
from ves_modeling.ode.problem import build_ode_problem
from ves_modeling.ode.verifier import OdeVerifier


def _key(value) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    return str(value)


def _make_ode_data(
    root: Path,
    *,
    n_trajectories: int = 1,
    train_len: int = 16,
    horizon: int = 3,
    numeric_ids: bool = False,
    interleave: bool = False,
    host_reversed: bool = False,
    noise: float = 0.05,
    seed: int = 7,
) -> tuple[Path, Path, dict[tuple[str, float], float]]:
    rng = np.random.default_rng(seed)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    if n_trajectories == 1:
        ids: list[str | int] | None = None
    elif numeric_ids:
        ids = list(range(n_trajectories))
    else:
        ids = [f"s{i}" for i in range(n_trajectories)]
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    host_rows: list[dict] = []
    truth: dict[tuple[str, float], float] = {}
    trajectories = ids if ids is not None else [None]
    for trajectory in trajectories:
        slope = float(rng.normal(2.0, 0.4))
        intercept = float(rng.normal(1.0, 0.5))
        for step in range(train_len):
            row = {"t": float(step), "y": slope * step + intercept + rng.normal(0.0, noise)}
            if trajectory is not None:
                row["trajectory_id"] = trajectory
            train_rows.append(row)
        for index in range(horizon):
            step = train_len + index
            value = slope * step + intercept
            row = {"t": float(step)}
            if trajectory is not None:
                row["trajectory_id"] = trajectory
            test_rows.append(dict(row))
            host_rows.append({**row, "y": value})
            truth[(_key(trajectory) if trajectory is not None else "", float(step))] = value
    if interleave:
        test_rows.sort(key=lambda row: (row["t"], str(row.get("trajectory_id", ""))))
    if host_reversed:
        host_rows = host_rows[::-1]
    pd.DataFrame(train_rows).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(public / "test_features.csv", index=False)
    pd.DataFrame(host_rows).to_csv(host / "hidden_test_values.csv", index=False)
    return public, host, truth


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="predictions.json",
        content=json.dumps(payload),
        producer="test",
    )


def _keyed_predictions(truth: dict[tuple[str, float], float], *, shift: float = 0.0):
    return [
        {
            "trajectory_id": trajectory_key or 0,
            "t": time_value,
            "prediction": value + shift,
        }
        for (trajectory_key, time_value), value in sorted(truth.items())
    ]


def test_valid_single_trajectory_contract(tmp_path: Path) -> None:
    public, host, truth = _make_ode_data(tmp_path / "data")
    contract = validate_ode_data(public)
    assert contract.n_trajectories == 1
    assert contract.test_rows == 3
    assert contract.train_rows == 16
    assert contract.row_order == "input"
    assert contract.trajectory_id_column is None
    values = load_host_values(host, contract)
    assert values.shape == (3,)
    assert np.allclose(values, [truth[("", t)] for t in (16.0, 17.0, 18.0)])


def test_valid_multi_trajectory_key_contract(tmp_path: Path) -> None:
    public, host, truth = _make_ode_data(
        tmp_path / "data", n_trajectories=2, train_len=10
    )
    contract = validate_ode_data(
        public,
        trajectory_id_column="trajectory_id",
        row_order="key",
    )
    assert contract.n_trajectories == 2
    assert contract.test_rows == 6
    assert contract.row_order == "key"
    values = load_host_values(host, contract)
    aligned = {
        key: float(value)
        for key, value in zip(contract.test_keys, values)
    }
    assert aligned == pytest.approx(truth)


def test_interleaved_and_reversed_host_alignment(tmp_path: Path) -> None:
    public, host, truth = _make_ode_data(
        tmp_path / "data",
        n_trajectories=2,
        train_len=10,
        interleave=True,
        host_reversed=True,
    )
    contract = validate_ode_data(
        public,
        trajectory_id_column="trajectory_id",
        row_order="key",
    )
    first_two = [key[0] for key in contract.test_keys[:2]]
    assert first_two[0] != first_two[1]
    values = load_host_values(host, contract)
    aligned = {
        key: float(value)
        for key, value in zip(contract.test_keys, values)
    }
    assert aligned == pytest.approx(truth)


def test_numeric_trajectory_key_canonicalization(tmp_path: Path) -> None:
    public, host, truth = _make_ode_data(
        tmp_path / "data", n_trajectories=2, train_len=10, numeric_ids=True
    )
    contract = validate_ode_data(
        public,
        trajectory_id_column="trajectory_id",
        row_order="key",
    )
    assert {key[0] for key in contract.test_keys} == {"0", "1"}
    values = load_host_values(host, contract)
    aligned = {
        key: float(value)
        for key, value in zip(contract.test_keys, values)
    }
    assert aligned == pytest.approx(truth)
    assert _trajectory_key(1) == _trajectory_key(1.0) == _trajectory_key("1") == "1"


def test_trajectory_key_bad_values_rejected() -> None:
    for bad in (True, False, float("nan"), float("inf"), "", ["a"]):
        with pytest.raises(ValueError, match="trajectory ids"):
            _trajectory_key(bad)


def test_t_strictly_increasing_required(tmp_path: Path) -> None:
    public, _host, _truth = _make_ode_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train.loc[5, "t"] = train.loc[4, "t"]
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_ode_data(public)
    train.loc[5, "t"] = train.loc[4, "t"] - 1.0
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_ode_data(public)


def test_duplicate_trajectory_time_rejected(tmp_path: Path) -> None:
    public, _host, _truth = _make_ode_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="strictly increasing|duplicate"):
        validate_ode_data(public)


def test_min_rows_requirements(tmp_path: Path) -> None:
    public, _host, _truth = _make_ode_data(
        tmp_path / "single", train_len=15
    )
    with pytest.raises(ValueError, match="at least 16 rows"):
        validate_ode_data(public)
    public2, _host2, _truth2 = _make_ode_data(
        tmp_path / "multi", n_trajectories=2, train_len=7
    )
    with pytest.raises(ValueError, match="at least 8 rows"):
        validate_ode_data(
            public2,
            trajectory_id_column="trajectory_id",
            row_order="key",
        )


def test_test_trajectory_must_have_history(tmp_path: Path) -> None:
    public, _host, _truth = _make_ode_data(
        tmp_path / "data", n_trajectories=2, train_len=10
    )
    test = pd.read_csv(public / "test_features.csv")
    test.loc[0, "trajectory_id"] = "alien"
    test.to_csv(public / "test_features.csv", index=False)
    with pytest.raises(ValueError, match="missing history"):
        validate_ode_data(
            public,
            trajectory_id_column="trajectory_id",
            row_order="key",
        )


def test_non_finite_t_and_y_rejected(tmp_path: Path) -> None:
    public, _host, _truth = _make_ode_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train.loc[0, "t"] = float("nan")
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        validate_ode_data(public)
    train.loc[0, "t"] = 0.0
    train.loc[0, "y"] = float("inf")
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        validate_ode_data(public)


def test_host_count_mismatch(tmp_path: Path) -> None:
    public, host, _truth = _make_ode_data(tmp_path / "data")
    frame = pd.read_csv(host / "hidden_test_values.csv").head(2)
    frame.to_csv(host / "hidden_test_values.csv", index=False)
    contract = validate_ode_data(public)
    with pytest.raises(ValueError, match="hidden values count"):
        load_host_values(host, contract)


def test_validate_predictions_input_mode(tmp_path: Path) -> None:
    public, _host, truth = _make_ode_data(tmp_path / "data")
    contract = validate_ode_data(public)
    values = [truth[("", t)] for t in (16.0, 17.0, 18.0)]
    validate_predictions(
        {"predictions": values}, expected_count=contract.test_rows
    )
    with pytest.raises(ValueError, match="prediction count"):
        validate_predictions(
            {"predictions": values[:-1]},
            expected_count=contract.test_rows,
        )
    with pytest.raises(ValueError, match="numbers"):
        validate_predictions(
            {"predictions": ["a", "b", "c"]},
            expected_count=contract.test_rows,
        )
    with pytest.raises(ValueError, match="finite"):
        bad = [0.0, 0.0, 0.0]
        bad[0] = float("nan")
        validate_predictions(
            {"predictions": bad}, expected_count=contract.test_rows
        )


def test_validate_predictions_key_mode_attacks(tmp_path: Path) -> None:
    public, _host, truth = _make_ode_data(
        tmp_path / "data", n_trajectories=2, train_len=10
    )
    contract = validate_ode_data(
        public,
        trajectory_id_column="trajectory_id",
        row_order="key",
    )
    test_keys = contract.test_keys
    good = _keyed_predictions(truth)
    validate_predictions(
        {"predictions": good},
        expected_count=len(test_keys),
        test_keys=test_keys,
        key_columns=("trajectory_id", "t"),
    )
    with pytest.raises(ValueError, match="missing="):
        validate_predictions(
            {"predictions": good[:-1]},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    with pytest.raises(ValueError, match="duplicate key"):
        validate_predictions(
            {"predictions": [good[0], *good]},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    with pytest.raises(ValueError, match="extra="):
        extra = [*good, {"trajectory_id": "s9", "t": 99.0, "prediction": 1.0}]
        validate_predictions(
            {"predictions": extra},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    with_extra = [dict(item, probe=1) for item in good]
    with pytest.raises(ValueError, match="exactly"):
        validate_predictions(
            {"predictions": with_extra},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    bad_bool = [dict(item) for item in good]
    bad_bool[0]["prediction"] = True
    with pytest.raises(ValueError, match="number"):
        validate_predictions(
            {"predictions": bad_bool},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    bad_t = [dict(item) for item in good]
    bad_t[0]["t"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(
            {"predictions": bad_t},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )


def test_claims_ignored_and_metrics_finite(tmp_path: Path) -> None:
    public, host, truth = _make_ode_data(tmp_path / "data")
    problem = build_ode_problem(public, host)
    payload = {
        "predictions": [truth[("", t)] for t in (16.0, 17.0, 18.0)],
        "claimed_rmse": 0.000001,
        "score": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["rmse"] == pytest.approx(0.0)
    assert values["mae"] == pytest.approx(0.0)


def test_verifier_recomputes_rmse_mae(tmp_path: Path) -> None:
    public, host, _truth = _make_ode_data(tmp_path / "data")
    contract = validate_ode_data(public)
    hidden = load_host_values(host, contract)
    context = OdeVerificationContext(
        hidden, expected_count=int(hidden.size), row_order="input"
    )
    verifier = OdeVerifier()
    evidence = verifier.verify(
        _artifact({"predictions": [float(value) + 1.0 for value in hidden]}),
        context,
    )
    values = {o.name: o.value for o in evidence.observations}
    assert values["rmse"] == pytest.approx(1.0)
    assert values["mae"] == pytest.approx(1.0)


def test_context_invariant() -> None:
    values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="expected_count"):
        OdeVerificationContext(values, expected_count=2, row_order="input")
    with pytest.raises(ValueError, match="trajectory_keys and time_keys"):
        OdeVerificationContext(values, row_order="key")
    with pytest.raises(ValueError, match="only used when"):
        OdeVerificationContext(
            values,
            trajectory_keys=("a",),
            time_keys=(1.0,),
            row_order="input",
        )
    with pytest.raises(ValueError, match="non-empty"):
        OdeVerificationContext(np.array([]))
    with pytest.raises(ValueError, match="finite"):
        OdeVerificationContext(np.array([float("nan")]))
    context = OdeVerificationContext(
        values,
        trajectory_keys=("a", "a", "b"),
        time_keys=(1.0, 2.0, 3.0),
        row_order="key",
    )
    assert context.expected_count == 3
    assert len(context.fingerprint()) == 64
