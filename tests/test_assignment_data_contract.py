"""R22: assignment/TSP data contract (problem/permutation/cost)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.assignment.context import AssignVerificationContext
from ves_modeling.assignment.data_contract import (
    validate_assign_data,
    validate_solution,
)
from ves_modeling.assignment.problem import build_assignment_problem
from ves_modeling.assignment.verifier import AssignVerifier


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _assignment_problem() -> dict:
    return {
        "version": 1,
        "problem_type": "assignment",
        "size": 3,
        "costs": [
            [4.0, 1.0, 3.0],
            [2.0, 0.0, 5.0],
            [3.0, 2.0, 2.0],
        ],
    }


def _tsp_problem() -> dict:
    return {
        "version": 1,
        "problem_type": "tsp",
        "size": 4,
        "costs": [
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 35.0, 25.0],
            [15.0, 35.0, 0.0, 30.0],
            [20.0, 25.0, 30.0, 0.0],
        ],
        "start": 0,
    }


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contracts(tmp_path: Path) -> None:
    contract = validate_assign_data(
        _write_problem(tmp_path / "a", _assignment_problem())
    )
    assert contract.problem_type == "assignment"
    assert contract.size == 3
    json.dumps(contract.to_dict())
    tsp = validate_assign_data(
        _write_problem(tmp_path / "t", _tsp_problem())
    )
    assert tsp.problem_type == "tsp"
    assert tsp.start == 0


def test_schema_attacks(tmp_path: Path) -> None:
    base = _assignment_problem()
    bad = dict(base, size=2)
    with pytest.raises(ValueError, match=">= 3"):
        validate_assign_data(_write_problem(tmp_path / "s", bad))
    bad = dict(base, costs=base["costs"][:2])
    with pytest.raises(ValueError, match="size x size"):
        validate_assign_data(_write_problem(tmp_path / "c", bad))
    bad = dict(base, costs=[[4.0, 1.0], [2.0, 0.0], [3.0, 2.0]])
    with pytest.raises(ValueError, match="exactly 3 entries"):
        validate_assign_data(_write_problem(tmp_path / "c2", bad))
    bad = dict(base, problem_type="flow")
    with pytest.raises(ValueError, match="problem_type"):
        validate_assign_data(_write_problem(tmp_path / "p", bad))
    bad = dict(base, start=1)
    with pytest.raises(ValueError, match="must not declare 'start'"):
        validate_assign_data(_write_problem(tmp_path / "st", bad))
    tsp = _tsp_problem()
    bad = dict(tsp, costs=tsp["costs"][:])
    bad["costs"][0][1] = 99.0  # break symmetry
    with pytest.raises(ValueError, match="symmetric"):
        validate_assign_data(_write_problem(tmp_path / "sym", bad))
    bad = dict(tsp, costs=tsp["costs"][:])
    bad["costs"][0][0] = 1.0
    with pytest.raises(ValueError, match="diagonal costs must be zero"):
        validate_assign_data(_write_problem(tmp_path / "diag", bad))


def test_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir(parents=True)
    text = json.dumps(_assignment_problem()).replace(
        '"size": 3',
        '"size": 3,\n    "size": 3',
    )
    (public / "problem.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        validate_assign_data(public)


def test_solution_validation(tmp_path: Path) -> None:
    contract = validate_assign_data(
        _write_problem(tmp_path / "a", _assignment_problem())
    )
    perm = validate_solution({"assignment": [1, 0, 2]}, contract)
    assert perm == [1, 0, 2]
    with pytest.raises(ValueError, match="missing required field"):
        validate_solution({}, contract)
    with pytest.raises(ValueError, match="permutation"):
        validate_solution({"assignment": [0, 0, 1]}, contract)
    with pytest.raises(ValueError, match="out of range"):
        validate_solution({"assignment": [0, 1, 3]}, contract)
    with pytest.raises(ValueError, match="exactly 3 entries"):
        validate_solution({"assignment": [0, 1]}, contract)
    tsp = validate_assign_data(_write_problem(tmp_path / "t", _tsp_problem()))
    tour = validate_solution({"tour": [0, 2, 3, 1]}, tsp)
    assert tour == [0, 2, 3, 1]
    with pytest.raises(ValueError, match="must start at"):
        validate_solution({"tour": [2, 0, 1, 3]}, tsp)


def test_verifier_cost_and_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "a", _assignment_problem())
    problem = build_assignment_problem(public)
    payload = {"assignment": [1, 0, 2], "total_cost": 0.0}
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["total_cost"] == pytest.approx(1.0 + 2.0 + 2.0)


def test_verifier_tsp_cost(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "t", _tsp_problem())
    contract = validate_assign_data(public)
    context = AssignVerificationContext(contract)
    verifier = AssignVerifier()
    evidence = verifier.verify(_artifact({"tour": [0, 2, 3, 1]}), context)
    values = {o.name: o.value for o in evidence.observations}
    # 0->2 (15) + 2->3 (30) + 3->1 (25) + 1->0 (10) = 80
    assert values["total_cost"] == pytest.approx(80.0)


def test_context_invariant(tmp_path: Path) -> None:
    from dataclasses import replace

    contract = validate_assign_data(
        _write_problem(tmp_path / "a", _assignment_problem())
    )
    with pytest.raises(ValueError, match="problem_type"):
        AssignVerificationContext(replace(contract, problem_type="x"))
    with pytest.raises(ValueError, match="size"):
        AssignVerificationContext(replace(contract, size=2))
    context = AssignVerificationContext(contract)
    assert len(context.fingerprint()) == 64
