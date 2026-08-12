"""R26: LQR data contract (instance/control/cost)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.lqr.data_contract import (
    validate_control,
    validate_lqr_data,
)
from ves_modeling.lqr.problem import build_lqr_problem
from ves_modeling.lqr.verifier import (
    reference_optimal_cost,
    simulate_total_cost,
)


def _problem() -> dict:
    return {
        "version": 1,
        "A": [[1.0]],
        "B": [[1.0]],
        "Q": [[1.0]],
        "R": [[1.0]],
        "x0": [1.0],
        "horizon": 2,
    }


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def test_valid_contract(tmp_path: Path) -> None:
    contract = validate_lqr_data(_write_problem(tmp_path / "d", _problem()))
    assert contract.n == 1
    assert contract.m == 1
    assert contract.horizon == 2
    assert contract.terminal_weight == "default Q"
    json.dumps(contract.to_dict())


def test_schema_attacks(tmp_path: Path) -> None:
    base = _problem()
    bad = dict(base, A=[[1.0, 0.0]])
    with pytest.raises(ValueError, match="square"):
        validate_lqr_data(_write_problem(tmp_path / "a", bad))
    bad = dict(base, Q=[[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        validate_lqr_data(_write_problem(tmp_path / "q", bad))
    bad = dict(base, Q=[[-1.0]])
    with pytest.raises(ValueError, match="semidefinite"):
        validate_lqr_data(_write_problem(tmp_path / "q2", bad))
    bad = dict(base, R=[[0.0]])
    with pytest.raises(ValueError, match="positive definite"):
        validate_lqr_data(_write_problem(tmp_path / "r", bad))
    bad = dict(base, B=[[1.0], [0.0]])
    with pytest.raises(ValueError, match="n x m"):
        validate_lqr_data(_write_problem(tmp_path / "b", bad))
    bad = dict(base, x0=[1.0, 0.0])
    with pytest.raises(ValueError, match="exactly n entries"):
        validate_lqr_data(_write_problem(tmp_path / "x", bad))
    bad = dict(base, horizon=1)
    with pytest.raises(ValueError, match=">= 2"):
        validate_lqr_data(_write_problem(tmp_path / "h", bad))
    bad = dict(base, A=[[True]])
    with pytest.raises(ValueError, match="finite number"):
        validate_lqr_data(_write_problem(tmp_path / "bool", bad))
    bad = dict(base, A=[[float("nan")]])
    with pytest.raises(ValueError, match="finite"):
        validate_lqr_data(_write_problem(tmp_path / "nan", bad))
    bad = dict(base, extra=1)
    with pytest.raises(ValueError, match="unknown top-level"):
        validate_lqr_data(_write_problem(tmp_path / "extra", bad))

    public = tmp_path / "dup"
    public.mkdir()
    text = json.dumps(_problem()).replace(
        '"horizon": 2',
        '"horizon": 2,\n    "horizon": 2',
    )
    (public / "problem.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        validate_lqr_data(public)


def test_control_validation(tmp_path: Path) -> None:
    contract = validate_lqr_data(_write_problem(tmp_path / "d", _problem()))
    control = validate_control({"control": [[-0.6], [-0.2]]}, contract)
    assert control.shape == (2, 1)
    with pytest.raises(ValueError, match="missing required field"):
        validate_control({}, contract)
    with pytest.raises(ValueError, match="exactly 2 entries"):
        validate_control({"control": [[0.0]]}, contract)
    with pytest.raises(ValueError, match="exactly 1 entries"):
        validate_control({"control": [[0.0, 1.0], [-0.2]]}, contract)
    with pytest.raises(ValueError, match="finite number"):
        validate_control({"control": [[True], [-0.2]]}, contract)
    with pytest.raises(ValueError, match="finite"):
        validate_control({"control": [[float("inf")], [-0.2]]}, contract)


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_verifier_simulates_cost_and_audit(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", _problem())
    problem = build_lqr_problem(public)
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(
        _artifact({"control": [[-0.6], [-0.2]]})
    )
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["total_cost"] == pytest.approx(1.6)
    assert values["reference_optimal_cost"] == pytest.approx(1.6)


def test_verifier_and_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", _problem())
    problem = build_lqr_problem(public)
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(
        _artifact(
            {
                "control": [[-0.6], [-0.2]],
                "claimed_total_cost": 0.0,
                "claimed_optimal": True,
            }
        )
    )
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["total_cost"] == pytest.approx(1.6)


def test_reference_matches_optimal_simulation(tmp_path: Path) -> None:
    contract = validate_lqr_data(_write_problem(tmp_path / "d", _problem()))
    reference = reference_optimal_cost(contract)
    assert reference == pytest.approx(1.6)
    control = validate_control({"control": [[-0.6], [-0.2]]}, contract)
    assert simulate_total_cost(contract, control) == pytest.approx(reference)
