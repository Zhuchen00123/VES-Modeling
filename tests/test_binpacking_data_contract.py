"""R24: bin packing data contract (problem/assignment/feasibility)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.binpacking.context import BinVerificationContext
from ves_modeling.binpacking.data_contract import (
    validate_bin_data,
    validate_solution,
)
from ves_modeling.binpacking.problem import build_binpacking_problem
from ves_modeling.binpacking.verifier import BinVerifier


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _problem() -> dict:
    return {
        "version": 1,
        "capacity": 10.0,
        "items": [6.0, 5.0, 4.0, 3.0, 2.0],
        "n_items": 5,
    }


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contract(tmp_path: Path) -> None:
    contract = validate_bin_data(_write_problem(tmp_path / "d", _problem()))
    assert contract.capacity == 10.0
    assert contract.n_items == 5
    json.dumps(contract.to_dict())


def test_schema_attacks(tmp_path: Path) -> None:
    base = _problem()
    bad = dict(base, capacity=0.0)
    with pytest.raises(ValueError, match="positive"):
        validate_bin_data(_write_problem(tmp_path / "c", bad))
    bad = dict(base, items=[6.0, 5.0, -1.0, 3.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        validate_bin_data(_write_problem(tmp_path / "i", bad))
    bad = dict(base, items=[6.0, 5.0, 11.0, 3.0, 2.0])
    with pytest.raises(ValueError, match="must not exceed capacity"):
        validate_bin_data(_write_problem(tmp_path / "i2", bad))
    bad = dict(base, items=[6.0, 5.0, 4.0, 3.0])
    with pytest.raises(ValueError, match="must match"):
        validate_bin_data(_write_problem(tmp_path / "n", bad))
    bad = dict(base, items=[])
    with pytest.raises(ValueError, match="non-empty"):
        validate_bin_data(_write_problem(tmp_path / "e", bad))


def test_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir(parents=True)
    text = json.dumps(_problem()).replace(
        '"capacity": 10.0',
        '"capacity": 10.0,\n    "capacity": 10.0',
    )
    (public / "problem.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        validate_bin_data(public)


def test_solution_validation(tmp_path: Path) -> None:
    contract = validate_bin_data(_write_problem(tmp_path / "d", _problem()))
    assignment = validate_solution(
        {"assignment": [0, 1, 0, 1, 2]}, contract
    )
    assert assignment == [0, 1, 0, 1, 2]
    with pytest.raises(ValueError, match="missing required field"):
        validate_solution({}, contract)
    with pytest.raises(ValueError, match="exactly 5 entries"):
        validate_solution({"assignment": [0, 1]}, contract)
    with pytest.raises(ValueError, match="non-negative"):
        validate_solution({"assignment": [0, 1, 0, -1, 2]}, contract)
    with pytest.raises(ValueError, match="contiguous"):
        validate_solution({"assignment": [0, 2, 0, 2, 0]}, contract)


def test_verifier_and_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", _problem())
    problem = build_binpacking_problem(public)
    payload = {"assignment": [0, 1, 0, 1, 2], "bin_count": 0.0}
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["bin_count"] == 3.0
    assert values["capacity_violation"] == 0.0


def test_verifier_capacity_violation(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", _problem())
    contract = validate_bin_data(public)
    context = BinVerificationContext(contract)
    verifier = BinVerifier()
    evidence = verifier.verify(
        _artifact({"assignment": [0, 0, 0, 1, 1]}), context
    )
    values = {o.name: o.value for o in evidence.observations}
    assert values["bin_count"] == 2.0
    assert values["capacity_violation"] == pytest.approx(5.0)


def test_context_invariant(tmp_path: Path) -> None:
    from dataclasses import replace

    contract = validate_bin_data(_write_problem(tmp_path / "d", _problem()))
    with pytest.raises(ValueError, match="at least one item"):
        BinVerificationContext(replace(contract, items=()))
    with pytest.raises(ValueError, match="capacity"):
        BinVerificationContext(replace(contract, capacity=0.0))
    context = BinVerificationContext(contract)
    assert len(context.fingerprint()) == 64
