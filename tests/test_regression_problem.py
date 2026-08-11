"""R1: problem assembly, contract, judge spec and replay references."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.regression.problem import (
    build_regression_problem,
    load_hidden_labels,
)


def _make_public(root: Path, rows: int = 3) -> Path:
    public = root / "public"
    public.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"x0": [1.0, 2.0, 3.0][:rows], "target": [1.0, 2.0, 3.0][:rows]}
    ).to_csv(public / "train.csv", index=False)
    pd.DataFrame({"x0": [0.5, 0.6, 0.7][:rows]}).to_csv(
        public / "test_features.csv", index=False
    )
    return public


def test_problem_assembly(tmp_path: Path):
    labels = np.array([1.0, 2.0, 3.0])
    problem = build_regression_problem(
        _make_public(tmp_path), tmp_path / "host", labels=labels
    )
    assert problem.contract.filename == "predictions.json"
    assert problem.contract.required_fields == ("predictions",)
    assert problem.name == "regression:regression"
    assert problem.verifier.version == "0.1.0"
    assert problem.verifier_module == "ves_modeling.regression.problem"
    assert problem.verifier_attr == "verifier"
    assert problem.context_factory_ref == "ves_modeling.regression.problem:context_factory"

    objectives = {o.observation: o for o in problem.judge_spec.objectives}
    assert set(objectives) == {"rmse"}
    assert objectives["rmse"].direction.value == "minimize"
    assert any(g.observation == "rmse" for g in problem.judge_spec.gates)

    context = problem.make_context()
    assert context.expected_count == 3
    assert context.id == "regression:regression"
    assert len(context.fingerprint()) == 64


def test_context_fingerprint_is_one_way_and_sensitive(tmp_path: Path):
    labels = np.array([1.0, 2.0, 3.0])
    problem = build_regression_problem(
        _make_public(tmp_path), tmp_path / "host", labels=labels
    )
    fingerprint = problem.make_context().fingerprint()
    assert fingerprint != "".join(
        format(int(v), "02x") for v in labels.astype(np.uint64)
    )
    assert problem.make_context().fingerprint() == fingerprint


def test_build_problem_rejects_nan_labels(tmp_path: Path):
    labels = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="hidden labels"):
        build_regression_problem(
            _make_public(tmp_path), tmp_path / "host", labels=labels
        )


def test_build_problem_rejects_inf_labels(tmp_path: Path):
    labels = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="hidden labels"):
        build_regression_problem(
            _make_public(tmp_path), tmp_path / "host", labels=labels
        )


def test_load_hidden_labels_rejects_nan(tmp_path: Path):
    host = tmp_path / "host"
    host.mkdir()
    pd.DataFrame({"target": [1.0, np.nan]}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    with pytest.raises(ValueError, match="hidden labels must be non-empty and finite"):
        load_hidden_labels(host)


def test_load_hidden_labels_rejects_inf(tmp_path: Path):
    host = tmp_path / "host"
    host.mkdir()
    pd.DataFrame({"target": [1.0, np.inf]}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    with pytest.raises(ValueError, match="hidden labels must be non-empty and finite"):
        load_hidden_labels(host)
