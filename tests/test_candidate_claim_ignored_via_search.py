"""R3: adversarial demo — a cheating candidate cannot become best."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_regression
from ves.search import GreedyTop1Policy
from ves.search_engine import SearchEngine

from ves_modeling.regression.problem import build_regression_problem
from ves_modeling.regression.runner import LocalRegressionRunner


class CheatFirstGenerator:
    """Draft 0 = cheating candidate (claimed perfect), draft 1 = linear, improve = gb."""

    def __init__(self, fixture_dir: Path) -> None:
        self._fixture_dir = fixture_dir

    def draft(self, problem, index: int) -> str:
        name = "cheating_candidate.py" if index == 0 else "linear_regression.py"
        return (self._fixture_dir / name).read_text(encoding="utf-8")

    def improve(self, problem, anchor) -> str:
        return (self._fixture_dir / "gradient_boosting.py").read_text(encoding="utf-8")


@pytest.fixture
def dataset(tmp_path: Path):
    public = tmp_path / "public"
    host = tmp_path / "host"
    public.mkdir()
    host.mkdir()
    X, y = make_regression(n_samples=150, n_features=5, noise=10.0, random_state=11)
    feature_names = [f"feature_{i}" for i in range(5)]
    split = 30
    train = pd.DataFrame(X[split:], columns=feature_names)
    train["target"] = y[split:]
    test_features = pd.DataFrame(X[:split], columns=feature_names)
    hidden = pd.DataFrame({"target": y[:split]})
    train.to_csv(public / "train.csv", index=False)
    test_features.to_csv(public / "test_features.csv", index=False)
    hidden.to_csv(host / "hidden_test_labels.csv", index=False)
    return public, host


def test_cheater_never_wins(dataset, tmp_path: Path):
    public, host = dataset
    problem = build_regression_problem(public, host)
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "candidates"
    runner = LocalRegressionRunner(workspace=tmp_path / "runs", data_dir=public)
    engine = SearchEngine(
        problem=problem,
        generator=CheatFirstGenerator(fixtures),
        runner=runner,
        anchor_policy=GreedyTop1Policy(),
        drafts=2,
        improves=1,
    )
    result = engine.search()
    assert result.rejected == 0
    assert result.best_code is not None
    # The best program must not be the cheating candidate.
    assert "claimed_rmse" not in result.best_code
    best_rmse = {o.name: o.value for o in result.best_evidence}["rmse"]
    # The cheating candidate's host-verified rmse is large (zeros vs real target).
    assert best_rmse < 20.0
