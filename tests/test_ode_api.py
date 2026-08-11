"""R11: ODE API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.ode import (
    ApplyOdeResult,
    OdeSearchResult,
    apply_ode_solution,
    capabilities,
    run_ode_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ode"
NUMPY_CODE = (FIXTURES / "numpy_fallback.py").read_text(encoding="utf-8")


def _make_ode_data(
    root: Path,
    *,
    n_trajectories: int = 1,
    train_len: int = 16,
    horizon: int = 3,
    seed: int = 11,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    trajectories = (
        [None] if n_trajectories == 1 else [f"s{i}" for i in range(n_trajectories)]
    )
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    host_rows: list[dict] = []
    for trajectory in trajectories:
        slope = float(rng.normal(2.0, 0.3))
        intercept = float(rng.normal(1.0, 0.5))
        for step in range(train_len):
            row = {
                "t": float(step),
                "y": slope * step + intercept + rng.normal(0.0, 0.05),
            }
            if trajectory is not None:
                row["trajectory_id"] = trajectory
            train_rows.append(row)
        for index in range(horizon):
            step = train_len + index
            row = {"t": float(step)}
            if trajectory is not None:
                row["trajectory_id"] = trajectory
            test_rows.append(dict(row))
            host_rows.append({**row, "y": slope * step + intercept})
    pd.DataFrame(train_rows).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(public / "test_features.csv", index=False)
    pd.DataFrame(host_rows).to_csv(
        host / "hidden_test_values.csv", index=False
    )
    return public, host


def test_run_ode_search_mock_single_trajectory_verified(tmp_path: Path) -> None:
    public, host = _make_ode_data(tmp_path / "data")
    result = run_ode_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, OdeSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_rmse is not None and np.isfinite(result.best_rmse)
    assert result.best_mae is not None and np.isfinite(result.best_mae)
    assert result.best_rmse < 1.0
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()
    candidates = list((result.run_dir / "candidates").iterdir())
    assert len(candidates) == 3
    assert all((candidate / "run.json").is_file() for candidate in candidates)


def test_run_ode_search_mock_multi_trajectory_key_verified(
    tmp_path: Path,
) -> None:
    public, host = _make_ode_data(
        tmp_path / "data", n_trajectories=2, train_len=10
    )
    result = run_ode_search(
        public,
        host,
        drafts=2,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
        trajectory_id_column="trajectory_id",
        row_order="key",
    )
    assert result.status == "verified"
    assert result.best_rmse is not None and result.best_rmse < 1.0
    assert result.data_contract["row_order"] == "key"
    assert result.data_contract["n_trajectories"] == 2


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _make_ode_data(tmp_path / "data")
    result = run_ode_search(
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
    assert persisted["task"] == "ode"
    assert persisted["status"] == "verified"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["host"]["hidden_test_values.csv"],
    )


def test_apply_produced_unverified_no_metrics(tmp_path: Path) -> None:
    public, _host = _make_ode_data(tmp_path / "data")
    result = apply_ode_solution(
        NUMPY_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyOdeResult)
    assert result.status == "produced_unverified"
    assert result.runner == "local"
    assert result.predictions_path is not None
    payload = json.loads(result.predictions_path.read_text(encoding="utf-8"))
    assert len(payload["predictions"]) == 3
    assert not hasattr(result, "best_rmse")
    summary = result.to_summary()
    json.dumps(summary)
    assert "rmse" not in summary
    assert "mae" not in summary
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public, _host = _make_ode_data(tmp_path / "data")
    bad_code = NUMPY_CODE.replace(
        "    rows = [float(value) for value in predictions]",
        "    rows = [float(value) for value in predictions][:-1]",
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_ode_solution(
            bad_code,
            public,
            workspace=tmp_path / "runs",
            trusted_code=True,
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


def test_run_ode_search_rejects_unknown_generator(tmp_path: Path) -> None:
    public, host = _make_ode_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_ode_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_ode() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_ode_search",
        "apply_ode_solution",
    ]
    assert declaration["verified_metrics"] == ["rmse", "mae"]
    assert declaration["apply_statuses"] == ["produced_unverified"]
    assert declaration["data_contract"]["row_order"] == ["input", "key"]
