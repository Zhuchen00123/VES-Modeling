"""R1: problem assembly, contract, judge spec and replay references."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ves_modeling.regression.problem import build_regression_problem


def test_problem_assembly(tmp_path: Path):
    labels = np.array([1.0, 2.0, 3.0])
    problem = build_regression_problem(
        tmp_path / "public", tmp_path / "host", labels=labels
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
        tmp_path / "public", tmp_path / "host", labels=labels
    )
    fingerprint = problem.make_context().fingerprint()
    assert fingerprint != "".join(
        format(int(v), "02x") for v in labels.astype(np.uint64)
    )
    assert problem.make_context().fingerprint() == fingerprint
