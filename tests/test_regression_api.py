"""C4: stable regression API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ves_modeling.regression import RegressionSearchResult, run_regression_search

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "candidates"


def _make_data(root: Path, n: int = 30) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 2))
    y = 3.0 * x[:, 0] - 1.5 * x[:, 1] + rng.normal(scale=0.1, size=n)
    split = int(n * 0.7)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train = pd.DataFrame(
        {**{f"x{i}": x[:split, i] for i in range(2)}, "target": y[:split]}
    )
    test = pd.DataFrame({f"x{i}": x[split:, i] for i in range(2)})
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test_features.csv", index=False)
    pd.DataFrame({"target": y[split:]}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    return public, host


def test_run_regression_search_mock(tmp_path: Path):
    public, host = _make_data(tmp_path / "data")
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, RegressionSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_evidence is not None
    assert result.best_rmse is not None and result.best_rmse >= 0.0
    assert result.best_mae is not None
    assert result.rejected == 0
    assert result.run_dir.is_dir()
    assert (result.run_dir / "best_solution.py").is_file()
    summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["task"] == "regression"
    assert summary["status"] == "verified"
    assert summary["best_rmse"] == result.best_rmse


def test_run_regression_search_rejects_unknown_generator(tmp_path: Path):
    public, host = _make_data(tmp_path / "data")
    try:
        run_regression_search(public, host, generator="unknown")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown generator")
