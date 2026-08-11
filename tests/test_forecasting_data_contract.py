"""R8: forecasting data contract (series/time/frequency/horizon/exog)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.forecasting.context import ForecastingVerificationContext
from ves_modeling.forecasting.data_contract import (
    _canonical_time,
    _series_key,
    load_host_labels,
    validate_forecasting_data,
    validate_predictions,
)
from ves_modeling.forecasting.problem import build_forecasting_problem
from ves_modeling.forecasting.verifier import ForecastingVerifier


def _key(value) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    return str(value)


def _make_forecast_data(
    root: Path,
    *,
    n_series: int = 2,
    train_len: int = 20,
    horizon: int = 5,
    freq: str = "D",
    numeric_ids: bool = False,
    interleave: bool = False,
    exog: bool = False,
    host_reversed: bool = False,
    noise: float = 0.1,
    seed: int = 7,
) -> tuple[Path, Path, dict[tuple[str, str], float]]:
    rng = np.random.default_rng(seed)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    series_ids = (
        list(range(n_series)) if numeric_ids else [f"s{i}" for i in range(n_series)]
    )
    offset = to_offset(freq)
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    host_rows: list[dict] = []
    truth: dict[tuple[str, str], float] = {}
    for series_id in series_ids:
        slope = float(rng.normal(1.0, 0.3))
        intercept = float(rng.normal(0.0, 1.0))
        start = pd.Timestamp("2024-01-01")
        train_times = pd.date_range(start, periods=train_len, freq=offset)
        for step, timestamp in enumerate(train_times):
            value = intercept + slope * step + rng.normal(0.0, noise)
            row = {
                "series_id": series_id,
                "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                "target": value,
            }
            if exog:
                row["x"] = float(step)
            train_rows.append(row)
        test_times = pd.date_range(
            train_times[-1] + offset, periods=horizon, freq=offset
        )
        for index, timestamp in enumerate(test_times):
            step = train_len + index
            value = intercept + slope * step + rng.normal(0.0, noise)
            row = {
                "series_id": series_id,
                "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if exog:
                row["x"] = float(step)
            test_rows.append(row)
            host_rows.append({**row, "target": value})
            truth[(_key(series_id), timestamp.isoformat())] = value
    if interleave:
        test_rows.sort(key=lambda row: (row["timestamp"], str(row["series_id"])))
    if host_reversed:
        host_rows = host_rows[::-1]
    pd.DataFrame(train_rows).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(public / "test_features.csv", index=False)
    pd.DataFrame(host_rows)[
        ["series_id", "timestamp", "target"]
    ].to_csv(host / "hidden_test_labels.csv", index=False)
    return public, host, truth


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="predictions.json",
        content=json.dumps(payload),
        producer="test",
    )


def _keyed_predictions(truth: dict[tuple[str, str], float], *, shift: float = 0.0):
    return [
        {
            "series_id": series_key,
            "timestamp": time_key,
            "prediction": value + shift,
        }
        for (series_key, time_key), value in sorted(truth.items())
    ]


def test_valid_contract_repeated_series_ids(tmp_path: Path) -> None:
    public, host, _truth = _make_forecast_data(tmp_path / "data")
    contract = validate_forecasting_data(public)
    assert contract.series_ids == ("s0", "s1")
    assert contract.horizon == 5
    assert contract.n_series == 2
    assert contract.test_rows == 10
    assert contract.train_rows == 40
    assert contract.frequency == "D"
    assert contract.row_order == "key"
    assert contract.feature_columns == ()
    assert len(contract.test_keys) == 10
    labels = load_host_labels(host, contract)
    assert labels.shape == (10,)


def test_repeated_series_ids_are_allowed(tmp_path: Path) -> None:
    public, host, truth = _make_forecast_data(
        tmp_path / "data", n_series=3, train_len=25, horizon=4
    )
    contract = validate_forecasting_data(public)
    assert contract.n_series == 3
    assert contract.horizon == 4
    assert contract.test_rows == 12
    labels = load_host_labels(host, contract)
    aligned = {
        key: float(value)
        for key, value in zip(contract.test_keys, labels)
    }
    assert aligned == pytest.approx(truth)


def test_duplicate_series_time_pair_rejected(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        validate_forecasting_data(public)


def test_numeric_series_key_canonicalization(tmp_path: Path) -> None:
    public, host, truth = _make_forecast_data(
        tmp_path / "data", numeric_ids=True, n_series=3
    )
    contract = validate_forecasting_data(public)
    assert contract.series_ids == ("0", "1", "2")
    labels = load_host_labels(host, contract)
    aligned = {
        key: float(value)
        for key, value in zip(contract.test_keys, labels)
    }
    assert aligned == pytest.approx(truth)
    # 1 and 1.0 and '1' all canonicalize to the same series key.
    assert _series_key(1) == _series_key(1.0) == _series_key("1") == "1"
    assert _series_key(1.5) == "1.5"


def test_series_id_bad_values_rejected(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    frame = train.copy()
    frame.loc[0, "series_id"] = ""
    frame.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="series id"):
        validate_forecasting_data(public)
    # bool/NaN/Inf/empty/non-scalar are rejected at the key level.
    for bad in (True, False, float("nan"), float("inf"), "", ["a"]):
        with pytest.raises(ValueError, match="series id"):
            _series_key(bad)


def test_interleaved_row_order_alignment(tmp_path: Path) -> None:
    public, host, truth = _make_forecast_data(
        tmp_path / "data", interleave=True, numeric_ids=True
    )
    contract = validate_forecasting_data(public)
    # Row order is interleaved, so adjacent keys belong to different series.
    first_series = [key[0] for key in contract.test_keys[:2]]
    assert first_series[0] != first_series[1]
    labels = load_host_labels(host, contract)
    aligned = {
        key: float(value)
        for key, value in zip(contract.test_keys, labels)
    }
    assert aligned == pytest.approx(truth)


def test_no_exog_and_with_exog(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data")
    assert validate_forecasting_data(public).feature_columns == ()
    public2, _host2, _truth2 = _make_forecast_data(
        tmp_path / "data2", exog=True
    )
    contract = validate_forecasting_data(public2)
    assert contract.feature_columns == ("x",)


def test_frequency_real_offset_validation(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data")
    for bad in ("banana", "weekly", "every day"):
        with pytest.raises(ValueError, match="real pandas offset"):
            validate_forecasting_data(public, frequency=bad)
    for freq in ("h", "2h", "MS"):
        data_dir = tmp_path / f"freq-{freq}"
        public_freq, _host_freq, _truth_freq = _make_forecast_data(
            data_dir, freq=freq
        )
        contract = validate_forecasting_data(public_freq, frequency=freq)
        assert contract.frequency == freq


def test_frequency_mismatch_rejected(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "daily")
    with pytest.raises(ValueError, match="do not follow frequency"):
        validate_forecasting_data(public, frequency="2h")


def test_irregular_series_rejected(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train = train[train["timestamp"] != "2024-01-03T00:00:00"]
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="do not follow frequency"):
        validate_forecasting_data(public)


def test_anchored_frequency_mismatch_rejected(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data", freq="MS")
    with pytest.raises(ValueError, match="do not follow frequency"):
        validate_forecasting_data(public, frequency="W-MON")


def test_numeric_timestamp_rejected(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train["timestamp"] = 20240101 + np.arange(len(train))
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="numeric timestamps are rejected"):
        validate_forecasting_data(public)


def test_non_iso_timestamp_rejected(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train.loc[0, "timestamp"] = "not-a-date"
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="ISO 8601"):
        validate_forecasting_data(public)
    train.loc[0, "timestamp"] = "2024/01/01"
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="strict ISO 8601"):
        validate_forecasting_data(public)


def test_unsorted_or_duplicate_timestamps_rejected(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data")
    train = pd.read_csv(public / "train.csv")
    train = train.iloc[::-1].reset_index(drop=True)
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_forecasting_data(public)


def test_test_series_must_appear_in_train(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data")
    test = pd.read_csv(public / "test_features.csv")
    test.loc[0, "series_id"] = "alien"
    test.to_csv(public / "test_features.csv", index=False)
    with pytest.raises(ValueError, match="missing history"):
        validate_forecasting_data(public)


def test_horizon_uniform_required(tmp_path: Path) -> None:
    public, _host, _truth = _make_forecast_data(tmp_path / "data")
    test = pd.read_csv(public / "test_features.csv")
    missing_one = test[test["series_id"] == "s1"].head(1)
    test = test[~test.index.isin(missing_one.index)].copy()
    test.to_csv(public / "test_features.csv", index=False)
    with pytest.raises(ValueError, match="same number of rows"):
        validate_forecasting_data(public)


def test_host_labels_key_alignment_reversed(tmp_path: Path) -> None:
    public, host, truth = _make_forecast_data(
        tmp_path / "data", host_reversed=True, interleave=True
    )
    contract = validate_forecasting_data(public)
    labels = load_host_labels(host, contract)
    aligned = {
        key: float(value)
        for key, value in zip(contract.test_keys, labels)
    }
    assert aligned == pytest.approx(truth)


def test_host_labels_count_mismatch(tmp_path: Path) -> None:
    public, host, _truth = _make_forecast_data(tmp_path / "data")
    frame = pd.read_csv(host / "hidden_test_labels.csv").head(9)
    frame.to_csv(host / "hidden_test_labels.csv", index=False)
    contract = validate_forecasting_data(public)
    with pytest.raises(ValueError, match="hidden labels count"):
        load_host_labels(host, contract)


def test_validate_predictions_input_mode(tmp_path: Path) -> None:
    public, host, truth = _make_forecast_data(tmp_path / "data")
    contract = validate_forecasting_data(public, row_order="input")
    labels = load_host_labels(host, contract)
    values = list(labels)
    validate_predictions(
        {"predictions": values}, expected_count=contract.test_rows
    )
    with pytest.raises(ValueError, match="prediction count"):
        validate_predictions(
            {"predictions": values[:-1]},
            expected_count=contract.test_rows,
        )
    with pytest.raises(ValueError, match="booleans"):
        validate_predictions(
            {"predictions": [True] * contract.test_rows},
            expected_count=contract.test_rows,
        )
    with pytest.raises(ValueError, match="finite"):
        bad = [0.0] * contract.test_rows
        bad[0] = float("nan")
        validate_predictions(
            {"predictions": bad}, expected_count=contract.test_rows
        )
    assert truth  # silence unused for input-mode path


def test_validate_predictions_key_mode_attacks(tmp_path: Path) -> None:
    public, _host, truth = _make_forecast_data(tmp_path / "data")
    contract = validate_forecasting_data(public)
    test_keys = contract.test_keys
    good = _keyed_predictions(truth)
    validate_predictions(
        {"predictions": good},
        expected_count=len(test_keys),
        test_keys=test_keys,
        key_columns=("series_id", "timestamp"),
    )
    # missing one key
    with pytest.raises(ValueError, match="missing="):
        validate_predictions(
            {"predictions": good[:-1]},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    # duplicate key
    duplicate = [dict(good[0]), *good]
    with pytest.raises(ValueError, match="duplicate key"):
        validate_predictions(
            {"predictions": duplicate},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    # extra key
    extra = [*good, {"series_id": "s9", "timestamp": "2030-01-01T00:00:00", "prediction": 1.0}]
    with pytest.raises(ValueError, match="extra="):
        validate_predictions(
            {"predictions": extra},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    # extra artifact field inside a record
    with_extra = [dict(item, probe=1) for item in good]
    with pytest.raises(ValueError, match="exactly"):
        validate_predictions(
            {"predictions": with_extra},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    # bool / non-finite prediction
    bad_bool = [dict(item) for item in good]
    bad_bool[0]["prediction"] = True
    with pytest.raises(ValueError, match="number"):
        validate_predictions(
            {"predictions": bad_bool},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )
    # numeric timestamp rejected in artifact
    bad_time = [dict(item) for item in good]
    bad_time[0]["timestamp"] = 20240101
    with pytest.raises(ValueError, match="ISO 8601"):
        validate_predictions(
            {"predictions": bad_time},
            expected_count=len(test_keys),
            test_keys=test_keys,
        )


def test_validate_predictions_key_mode_canonical_ids(tmp_path: Path) -> None:
    public, _host, truth = _make_forecast_data(
        tmp_path / "data", numeric_ids=True, n_series=2
    )
    contract = validate_forecasting_data(public)
    test_keys = contract.test_keys
    good = _keyed_predictions(truth)
    # First sorted key is series "0"; the integer 0 canonicalizes to "0".
    rewritten = [dict(item) for item in good]
    rewritten[0]["series_id"] = 0
    validate_predictions(
        {"predictions": rewritten},
        expected_count=len(test_keys),
        test_keys=test_keys,
    )


def test_claims_ignored(tmp_path: Path) -> None:
    public, host, truth = _make_forecast_data(tmp_path / "data")
    problem = build_forecasting_problem(public, host)
    payload = {
        "predictions": _keyed_predictions(truth, shift=0.5),
        "claimed_rmse": 0.000001,
        "claimed_smape": 0.000001,
        "score": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["rmse"] > 0.0  # host metric, not the tiny claim
    assert np.isfinite(values["rmse"])
    assert np.isfinite(values["mae"])
    assert np.isfinite(values["smape"])


def test_verifier_metrics_finite_and_smape_zero(tmp_path: Path) -> None:
    public, host, _truth = _make_forecast_data(tmp_path / "data")
    contract = validate_forecasting_data(public)
    labels = load_host_labels(host, contract)
    context = ForecastingVerificationContext(
        labels,
        expected_count=int(labels.size),
        series_keys=tuple(key[0] for key in contract.test_keys),
        time_keys=tuple(key[1] for key in contract.test_keys),
        row_order="key",
    )
    verifier = ForecastingVerifier()
    exact = [
        {
            "series_id": series_key,
            "timestamp": time_key,
            "prediction": float(value),
        }
        for (series_key, time_key), value in zip(contract.test_keys, labels)
    ]
    evidence = verifier.verify(_artifact({"predictions": exact}), context)
    values = {o.name: o.value for o in evidence.observations}
    for metric in ("rmse", "mae", "smape"):
        assert np.isfinite(values[metric])
    assert values["smape"] == 0.0


def test_context_invariant() -> None:
    labels = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="expected_count"):
        ForecastingVerificationContext(
            labels, expected_count=2, row_order="input"
        )
    with pytest.raises(ValueError, match="series_keys and time_keys"):
        ForecastingVerificationContext(labels, row_order="key")
    with pytest.raises(ValueError, match="only used when"):
        ForecastingVerificationContext(
            labels, series_keys=("a",), time_keys=("b",), row_order="input"
        )
    with pytest.raises(ValueError, match="non-empty"):
        ForecastingVerificationContext(np.array([]))
    with pytest.raises(ValueError, match="finite"):
        ForecastingVerificationContext(np.array([float("nan")]))
    context = ForecastingVerificationContext(
        labels,
        series_keys=("a", "a", "b"),
        time_keys=("t1", "t2", "t3"),
        row_order="key",
    )
    assert context.expected_count == 3
    assert len(context.fingerprint()) == 64


def test_canonical_time() -> None:
    assert _canonical_time("2024-01-01") == "2024-01-01T00:00:00"
    assert _canonical_time("2024-01-01T06:00:00") == "2024-01-01T06:00:00"
    assert _canonical_time("2024-01-01 06:00:00") == "2024-01-01T06:00:00"
    for bad in (20240101, 1.5, True, "2024/01/01", "not-a-date"):
        with pytest.raises(ValueError, match="ISO 8601"):
            _canonical_time(bad)
