"""R8: forecasting API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

from ves_modeling.forecasting import (
    ApplyForecastingResult,
    ForecastingSearchResult,
    apply_forecasting_solution,
    capabilities,
    run_forecasting_search,
)
from ves_modeling.regression.runner import LocalRegressionRunner

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "forecasting"
LINEAR = (FIXTURES / "linear_forecast.py").read_text(encoding="utf-8")


def _make_forecast_data(
    root: Path,
    *,
    n_series: int = 2,
    train_len: int = 20,
    horizon: int = 5,
    numeric_ids: bool = False,
    interleave: bool = False,
    exog: bool = False,
    seed: int = 11,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    series_ids = (
        list(range(n_series)) if numeric_ids else [f"s{i}" for i in range(n_series)]
    )
    offset = to_offset("D")
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    host_rows: list[dict] = []
    for series_id in series_ids:
        slope = float(rng.normal(1.0, 0.3))
        intercept = float(rng.normal(0.0, 1.0))
        train_times = pd.date_range("2024-01-01", periods=train_len, freq=offset)
        for step, timestamp in enumerate(train_times):
            value = intercept + slope * step + rng.normal(0.0, 0.1)
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
            value = intercept + slope * step + rng.normal(0.0, 0.1)
            row = {
                "series_id": series_id,
                "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if exog:
                row["x"] = float(step)
            test_rows.append(row)
            host_rows.append({**row, "target": value})
    if interleave:
        test_rows.sort(key=lambda row: (row["timestamp"], str(row["series_id"])))
    pd.DataFrame(train_rows).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(public / "test_features.csv", index=False)
    pd.DataFrame(host_rows)[
        ["series_id", "timestamp", "target"]
    ].to_csv(host / "hidden_test_labels.csv", index=False)
    return public, host


def test_run_forecasting_search_mock_verified(tmp_path: Path) -> None:
    public, host = _make_forecast_data(tmp_path / "data")
    result = run_forecasting_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, ForecastingSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_evidence is not None
    assert result.best_rmse is not None and result.best_rmse >= 0.0
    assert result.best_mae is not None
    assert result.best_smape is not None
    assert np.isfinite(result.best_rmse)
    assert np.isfinite(result.best_mae)
    assert np.isfinite(result.best_smape)
    assert result.rejected == 0
    assert result.run_dir.is_dir()
    assert (result.run_dir / "best_solution.py").is_file()
    assert (result.run_dir / "config.json").is_file()
    assert (result.run_dir / "provenance.json").is_file()
    candidates = list((result.run_dir / "candidates").iterdir())
    assert len(candidates) == 3
    assert all((candidate / "run.json").is_file() for candidate in candidates)


def test_search_key_e2e_interleaved_numeric_exog(tmp_path: Path) -> None:
    public, host = _make_forecast_data(
        tmp_path / "data",
        numeric_ids=True,
        interleave=True,
        exog=True,
        n_series=3,
    )
    result = run_forecasting_search(
        public,
        host,
        drafts=2,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
        row_order="key",
    )
    assert result.status == "verified"
    assert result.best_rmse is not None
    assert result.data_contract is not None
    assert result.data_contract["n_series"] == 3
    assert result.data_contract["feature_columns"] == ["x"]


def test_summary_parity(tmp_path: Path) -> None:
    public, host = _make_forecast_data(tmp_path / "data")
    result = run_forecasting_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == result.to_summary()
    assert persisted["task"] == "forecasting"
    assert persisted["status"] == "verified"
    assert persisted["best_rmse"] == result.best_rmse


def test_apply_key_mode_produced_unverified(tmp_path: Path) -> None:
    public, _host = _make_forecast_data(tmp_path / "data")
    result = apply_forecasting_solution(
        LINEAR,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
        row_order="key",
    )
    assert isinstance(result, ApplyForecastingResult)
    assert result.status == "produced_unverified"
    assert result.runner == "local"
    assert result.predictions_path is not None
    assert result.predictions_path.is_file()
    payload = json.loads(result.predictions_path.read_text(encoding="utf-8"))
    assert len(payload["predictions"]) == 10
    for item in payload["predictions"]:
        assert set(item.keys()) == {"series_id", "timestamp", "prediction"}
    assert not hasattr(result, "best_rmse")
    summary = result.to_summary()
    json.dumps(summary)  # must round-trip without a custom encoder
    assert "rmse" not in summary
    assert "mae" not in summary
    assert "smape" not in summary
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(tmp_path: Path) -> None:
    public, _host = _make_forecast_data(tmp_path / "data")
    bad_code = (
        LINEAR.replace(
            'json.dump({"predictions": rows}, fh)',
            'json.dump({"predictions": rows[:-1]}, fh)',
        )
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_forecasting_solution(
            bad_code,
            public,
            workspace=tmp_path / "runs",
            trusted_code=True,
            row_order="key",
            run_id=run_id,
        )
    run_dir = tmp_path / "runs" / run_id
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "artifact_invalid"
    assert summary["error"] is not None
    run_json = json.loads(
        (run_dir / "candidate" / "run.json").read_text(encoding="utf-8")
    )
    assert run_json["status"] == "artifact_invalid"


def test_apply_execution_failed_summary_status(tmp_path: Path) -> None:
    public, _host = _make_forecast_data(tmp_path / "data")
    failing_code = "import sys\nsys.exit(1)\n"
    run_id = "applyfailed0"
    with pytest.raises(RuntimeError, match="execution_failed"):
        apply_forecasting_solution(
            failing_code,
            public,
            workspace=tmp_path / "runs",
            trusted_code=True,
            row_order="key",
            run_id=run_id,
        )
    run_dir = tmp_path / "runs" / run_id
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "execution_failed"
    assert "returncode" in summary["error"]
    run_json = json.loads(
        (run_dir / "candidate" / "run.json").read_text(encoding="utf-8")
    )
    assert run_json["status"] == "execution_failed"


def test_apply_default_untrusted_uses_docker_error_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ves_modeling.regression.runner import DockerRegressionRunner

    public, _host = _make_forecast_data(tmp_path / "data")
    monkeypatch.setattr(
        DockerRegressionRunner, "is_available", lambda self: False
    )
    with pytest.raises(RuntimeError, match="Docker"):
        apply_forecasting_solution(
            LINEAR,
            public,
            workspace=tmp_path / "runs",
            row_order="key",
        )


def test_run_forecasting_search_rejects_unknown_generator(tmp_path: Path) -> None:
    public, host = _make_forecast_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_forecasting_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_forecasting() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_forecasting_search",
        "apply_forecasting_solution",
    ]
    assert declaration["verified_metrics"] == ["rmse", "mae", "smape"]
    assert declaration["apply_statuses"] == ["produced_unverified"]
    assert declaration["data_contract"]["row_order"] == ["input", "key"]


def test_local_runner_sets_ves_env_aliases(tmp_path: Path) -> None:
    public, _host = _make_forecast_data(tmp_path / "data")
    runner = LocalRegressionRunner(
        workspace=tmp_path / "runs", data_dir=public
    )
    code = (
        "import os\n"
        "print(os.environ['VES_DATA_DIR'])\n"
        "print(os.environ['VES_OUTPUT_DIR'])\n"
    )
    result = runner.run(code, "env0")
    assert result.succeeded, result.stderr
    assert str(public) in result.stdout
    assert "env0" in result.stdout
