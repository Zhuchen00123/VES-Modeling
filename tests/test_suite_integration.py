"""T-010: four vertical slices coexist with a uniform, stable API shape.

This suite is intentionally cross-cutting: it asserts the public contract of
the whole package (regression / forecasting / classification / optimization /
ODE) without re-running every per-slice behavior test.  It also guards the
architecture rule that we do not introduce universal task/solver
abstractions before repeated real-domain requirements exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import ves_modeling
from ves_modeling.classification.problem import build_classification_problem
from ves_modeling.classification.schema import (
    capabilities as classification_capabilities,
)
from ves_modeling.forecasting.problem import build_forecasting_problem
from ves_modeling.forecasting.schema import (
    capabilities as forecasting_capabilities,
)
from ves_modeling.ode.problem import build_ode_problem
from ves_modeling.ode.schema import capabilities as ode_capabilities
from ves_modeling.optimization.problem import build_optimization_problem
from ves_modeling.optimization.schema import (
    capabilities as optimization_capabilities,
)
from ves_modeling.regression.problem import build_regression_problem
from ves_modeling.regression.schema import (
    capabilities as regression_capabilities,
)

ALL_CAPABILITIES = (
    regression_capabilities,
    forecasting_capabilities,
    classification_capabilities,
    optimization_capabilities,
    ode_capabilities,
)


def test_top_level_exports_all_builders() -> None:
    assert set(ves_modeling.__all__) == {
        "build_classification_problem",
        "build_forecasting_problem",
        "build_ode_problem",
        "build_optimization_problem",
        "build_regression_problem",
    }
    for name in ves_modeling.__all__:
        assert callable(getattr(ves_modeling, name))


def test_capabilities_share_schema_version_and_apply_contract() -> None:
    declared = [fn() for fn in ALL_CAPABILITIES]
    assert len(declared) == 5
    for entry in declared:
        assert entry["api_schema_version"] == "1.0"
        assert entry["apply_statuses"] == ["produced_unverified"]
        assert entry["trust_boundaries"]["docker"].startswith(
            "default for untrusted/LLM code"
        )


def test_operations_follow_search_apply_naming() -> None:
    for fn in ALL_CAPABILITIES:
        ops = fn()["operations"]
        assert len(ops) == 2, ops
        assert ops[0].startswith("run_") and ops[0].endswith("_search"), ops
        assert ops[1].startswith("apply_") and ops[1].endswith("_solution"), ops


def test_optimization_capabilities_never_claim_optimality() -> None:
    entry = optimization_capabilities()
    assert "never claimed" in entry["data_contract"]["optimality"]


def test_no_universal_abstraction_in_src() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "ves_modeling"
    forbidden = ("UniversalTask", "TaskRegistry", "SolverRegistry")
    hits: list[str] = []
    for path in sorted(src.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append(f"{path.relative_to(src)}: {token}")
    assert not hits, "forbidden universal abstraction(s) found:\n" + "\n".join(hits)


def _make_regression_data(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 2))
    y = 2.0 * x[:, 0] - x[:, 1] + rng.normal(scale=0.1, size=20)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(
        {"x0": x[:14, 0], "x1": x[:14, 1], "target": y[:14]}
    ).to_csv(public / "train.csv", index=False)
    pd.DataFrame({"x0": x[14:, 0], "x1": x[14:, 1]}).to_csv(
        public / "test_features.csv", index=False
    )
    pd.DataFrame({"target": y[14:]}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    return public, host


def _make_forecast_data(root: Path) -> tuple[Path, Path]:
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    host_rows: list[dict] = []
    for series_id in ("a", "b"):
        times = pd.date_range("2024-01-01", periods=10, freq="D")
        for step, timestamp in enumerate(times):
            train_rows.append(
                {
                    "series_id": series_id,
                    "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "target": float(step),
                }
            )
        for index, timestamp in enumerate(
            pd.date_range(times[-1] + pd.Timedelta(days=1), periods=3, freq="D")
        ):
            value = float(10 + index)
            test_rows.append(
                {
                    "series_id": series_id,
                    "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            host_rows.append(
                {
                    "series_id": series_id,
                    "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "target": value,
                }
            )
    pd.DataFrame(train_rows).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(public / "test_features.csv", index=False)
    pd.DataFrame(host_rows).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    return public, host


def _make_classification_data(root: Path) -> tuple[Path, Path]:
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train = pd.DataFrame(
        {
            "f0": [0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5],
            "f1": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "target": ["no", "no", "no", "no", "yes", "yes", "yes", "yes"],
        }
    )
    test = pd.DataFrame(
        {
            "f0": [0.2, 1.2, 2.2, 3.2],
            "f1": [0.2, 0.2, 0.2, 0.2],
        }
    )
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test_features.csv", index=False)
    pd.DataFrame({"target": ["no", "no", "yes", "yes"]}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    return public, host


def _write_optimization_problem(root: Path) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(
            {
                "version": 1,
                "sense": "minimize",
                "variables": {
                    "x0": {"type": "continuous", "lower": 0.0, "upper": 10.0},
                    "x1": {"type": "continuous", "lower": 0.0, "upper": 10.0},
                },
                "objective": {"coefficients": {"x0": 1.0, "x1": 2.0}},
                "constraints": [
                    {
                        "coefficients": {"x0": 1.0, "x1": 1.0},
                        "sense": "<=",
                        "rhs": 5.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return public


def _make_ode_data(root: Path) -> tuple[Path, Path]:
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(
        {"t": [float(i) for i in range(16)], "y": [float(i) for i in range(16)]}
    ).to_csv(public / "train.csv", index=False)
    pd.DataFrame({"t": [16.0, 17.0, 18.0]}).to_csv(
        public / "test_features.csv", index=False
    )
    pd.DataFrame({"t": [16.0, 17.0, 18.0], "y": [16.0, 17.0, 18.0]}).to_csv(
        host / "hidden_test_values.csv", index=False
    )
    return public, host


def test_all_build_problems_constructible(tmp_path: Path) -> None:
    reg_public, reg_host = _make_regression_data(tmp_path / "reg")
    assert build_regression_problem(reg_public, reg_host) is not None

    fc_public, fc_host = _make_forecast_data(tmp_path / "fc")
    assert build_forecasting_problem(fc_public, fc_host) is not None

    clf_public, clf_host = _make_classification_data(tmp_path / "clf")
    assert (
        build_classification_problem(
            clf_public, clf_host, classes=["no", "yes"]
        )
        is not None
    )

    opt_public = _write_optimization_problem(tmp_path / "opt")
    assert build_optimization_problem(opt_public) is not None

    ode_public, ode_host = _make_ode_data(tmp_path / "ode")
    assert build_ode_problem(ode_public, ode_host) is not None
