"""R31: game data contract (instance/control/cost)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.game.data_contract import (
    validate_control,
    validate_game_data,
)
from ves_modeling.game.problem import build_game_problem
from ves_modeling.game.verifier import (
    reference_game_optimal_cost,
    simulate_game_total_cost,
)


def _problem() -> dict:
    return {
        "version": 1,
        "A": [[1.0]],
        "B": [[1.0]],
        "C": [[1.0]],
        "Q": [[1.0]],
        "R": [[1.0]],
        "S": [[1.0]],
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
    contract = validate_game_data(_write_problem(tmp_path / "d", _problem()))
    assert contract.n == 1
    assert contract.m == 1
    assert contract.p == 1
    assert contract.horizon == 2
    assert contract.terminal_weight == "default Q"
    json.dumps(contract.to_dict())


def test_schema_attacks(tmp_path: Path) -> None:
    base = _problem()
    bad = dict(base, A=[[1.0, 0.0]])
    with pytest.raises(ValueError, match="square"):
        validate_game_data(_write_problem(tmp_path / "a", bad))
    bad = dict(base, B=[[1.0], [0.0]])
    with pytest.raises(ValueError, match="n x m"):
        validate_game_data(_write_problem(tmp_path / "b", bad))
    bad = dict(base, C=[[1.0], [0.0]])
    with pytest.raises(ValueError, match="n x p"):
        validate_game_data(_write_problem(tmp_path / "c", bad))
    bad = dict(base, Q=[[-1.0]])
    with pytest.raises(ValueError, match="semidefinite"):
        validate_game_data(_write_problem(tmp_path / "q", bad))
    bad = dict(base, R=[[0.0]])
    with pytest.raises(ValueError, match="positive definite"):
        validate_game_data(_write_problem(tmp_path / "r", bad))
    bad = dict(base, S=[[0.0]])
    with pytest.raises(ValueError, match="positive definite"):
        validate_game_data(_write_problem(tmp_path / "s", bad))
    bad = dict(base, x0=[1.0, 0.0])
    with pytest.raises(ValueError, match="exactly n entries"):
        validate_game_data(_write_problem(tmp_path / "x", bad))
    bad = dict(base, horizon=1)
    with pytest.raises(ValueError, match=">= 2"):
        validate_game_data(_write_problem(tmp_path / "h", bad))
    bad = dict(base, A=[[True]])
    with pytest.raises(ValueError, match="finite number"):
        validate_game_data(_write_problem(tmp_path / "bool", bad))
    bad = dict(base, C=[[float("nan")]])
    with pytest.raises(ValueError, match="finite"):
        validate_game_data(_write_problem(tmp_path / "nan", bad))
    bad = dict(base, extra=1)
    with pytest.raises(ValueError, match="unknown top-level"):
        validate_game_data(_write_problem(tmp_path / "extra", bad))

    public = tmp_path / "dup"
    public.mkdir()
    text = json.dumps(_problem()).replace(
        '"horizon": 2',
        '"horizon": 2,\n    "horizon": 2',
    )
    (public / "problem.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        validate_game_data(public)


def test_control_validation(tmp_path: Path) -> None:
    contract = validate_game_data(_write_problem(tmp_path / "d", _problem()))
    control = validate_control({"control": [[-2.0], [-1.0]]}, contract)
    assert control.shape == (2, 1)
    with pytest.raises(ValueError, match="missing required field"):
        validate_control({}, contract)
    with pytest.raises(ValueError, match="exactly 2 entries"):
        validate_control({"control": [[0.0]]}, contract)
    with pytest.raises(ValueError, match="exactly 1 entries"):
        validate_control({"control": [[0.0, 1.0], [-1.0]]}, contract)
    with pytest.raises(ValueError, match="finite number"):
        validate_control({"control": [[True], [-1.0]]}, contract)
    with pytest.raises(ValueError, match="finite"):
        validate_control({"control": [[float("inf")], [-1.0]]}, contract)


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_verifier_known_scalar_game(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", _problem())
    problem = build_game_problem(public)
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(
        _artifact({"control": [[-2.0], [-1.0]]})
    )
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["total_cost"] == pytest.approx(3.0)
    assert values["reference_optimal_cost"] == pytest.approx(3.0)


def test_verifier_and_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", _problem())
    problem = build_game_problem(public)
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(
        _artifact(
            {
                "control": [[-2.0], [-1.0]],
                "claimed_total_cost": 0.0,
                "claimed_optimal": True,
            }
        )
    )
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["total_cost"] == pytest.approx(3.0)


def test_reference_matches_optimal_simulation(tmp_path: Path) -> None:
    contract = validate_game_data(_write_problem(tmp_path / "d", _problem()))
    reference = reference_game_optimal_cost(contract)
    assert reference == pytest.approx(3.0)
    control = validate_control({"control": [[-2.0], [-1.0]]}, contract)
    assert simulate_game_total_cost(contract, control) == pytest.approx(
        reference
    )
