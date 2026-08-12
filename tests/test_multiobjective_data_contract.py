"""R16: multi-objective data contract (problem/solution/hypervolume)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.multiobjective.context import MooVerificationContext
from ves_modeling.multiobjective.data_contract import (
    validate_moo_data,
    validate_solution,
)
from ves_modeling.multiobjective.problem import build_multiobjective_problem
from ves_modeling.multiobjective.verifier import MooVerifier


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _moo_problem(reference: list | None = None) -> dict:
    problem = {
        "version": 1,
        "variables": {
            "x0": {"type": "continuous", "lower": 0.0, "upper": 1.0},
            "x1": {"type": "continuous", "lower": 0.0, "upper": 1.0},
        },
        "objectives": [
            {"coefficients": {"x0": 1.0}, "constant": 0.0},
            {"coefficients": {"x1": 1.0}, "constant": 0.0},
        ],
        "constraints": [],
    }
    if reference is not None:
        problem["reference_point"] = reference
    return problem


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contract_default_reference(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _moo_problem())
    contract = validate_moo_data(public)
    assert contract.n_variables == 2
    assert len(contract.objectives) == 2
    assert contract.reference_point == (2.0, 2.0)
    json.dumps(contract.to_dict())
    explicit = _write_problem(
        tmp_path / "data2", _moo_problem(reference=[5.0, 5.0])
    )
    assert validate_moo_data(explicit).reference_point == (5.0, 5.0)


def test_schema_attacks(tmp_path: Path) -> None:
    base = _moo_problem()
    bad = dict(base, objectives=[base["objectives"][0]])
    with pytest.raises(ValueError, match="exactly two"):
        validate_moo_data(_write_problem(tmp_path / "o1", bad))
    bad = dict(
        base,
        objectives=[
            {"coefficients": {"ghost": 1.0}},
            base["objectives"][1],
        ],
    )
    with pytest.raises(ValueError, match="undeclared variable"):
        validate_moo_data(_write_problem(tmp_path / "o2", bad))
    bad = dict(base, variables={"x0": {"type": "continuous"}})
    with pytest.raises(ValueError, match="lower and upper"):
        validate_moo_data(_write_problem(tmp_path / "v1", bad))
    bad = dict(base, reference_point=[1.0])
    with pytest.raises(ValueError, match="\\[r1, r2\\]"):
        validate_moo_data(_write_problem(tmp_path / "r1", bad))
    bad = dict(base, constraints=[{"coefficients": {"x0": 1.0}, "sense": "<"}])
    with pytest.raises(ValueError, match="sense"):
        validate_moo_data(_write_problem(tmp_path / "c1", bad))


def test_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir(parents=True)
    text = json.dumps(_moo_problem()).replace(
        '"constraints": []',
        '"constraints": [],\n    "constraints": []',
    )
    (public / "problem.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        validate_moo_data(public)


def test_solution_validation(tmp_path: Path) -> None:
    contract = validate_moo_data(
        _write_problem(tmp_path / "data", _moo_problem())
    )
    solutions = validate_solution(
        {
            "solutions": [
                {"variables": {"x0": 1.0, "x1": 0.0}},
                {"variables": {"x0": 0.0, "x1": 1.0}},
            ]
        },
        contract,
    )
    assert len(solutions) == 2
    single = validate_solution({"variables": {"x0": 1.0, "x1": 1.0}}, contract)
    assert single == [{"x0": 1.0, "x1": 1.0}]
    with pytest.raises(ValueError, match="at least one solution"):
        validate_solution({"solutions": []}, contract)
    with pytest.raises(ValueError, match="missing="):
        validate_solution(
            {"solutions": [{"variables": {"x0": 1.0}}]}, contract
        )
    with pytest.raises(ValueError, match="extra="):
        validate_solution(
            {
                "solutions": [
                    {"variables": {"x0": 1.0, "x1": 1.0, "z": 0.0}}
                ]
            },
            contract,
        )
    with pytest.raises(ValueError, match="finite"):
        validate_solution(
            {"solutions": [{"variables": {"x0": float("nan"), "x1": 1.0}}]},
            contract,
        )
    with pytest.raises(ValueError, match="'solutions' or a single"):
        validate_solution({}, contract)


def test_hypervolume_and_dominance(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _moo_problem())
    contract = validate_moo_data(public)
    context = MooVerificationContext(contract)
    verifier = MooVerifier()
    payload = {
        "solutions": [
            {"variables": {"x0": 1.0, "x1": 0.0}},
            {"variables": {"x0": 0.0, "x1": 1.0}},
            {"variables": {"x0": 1.0, "x1": 1.0}},
        ],
        "hypervolume": 0.0,
        "optimality": "optimal",
    }
    evidence = verifier.verify(_artifact(payload), context)
    values = {o.name: o.value for o in evidence.observations}
    assert values["hypervolume"] == pytest.approx(1.0)
    assert values["non_dominated_count"] == 1.0
    assert values["feasible_count"] == 3.0
    assert values["total_count"] == 3.0


def test_infeasible_and_dominated_dropped(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _moo_problem())
    contract = validate_moo_data(public)
    context = MooVerificationContext(contract)
    verifier = MooVerifier()
    payload = {
        "solutions": [
            {"variables": {"x0": 5.0, "x1": 0.0}},  # bound violation
            {"variables": {"x0": 1.0, "x1": 0.5}},
            {"variables": {"x0": 0.5, "x1": 1.0}},
        ]
    }
    evidence = verifier.verify(_artifact(payload), context)
    values = {o.name: o.value for o in evidence.observations}
    assert values["total_count"] == 3.0
    assert values["feasible_count"] == 2.0
    assert values["non_dominated_count"] == 2.0
    assert values["hypervolume"] > 0.0


def test_infeasible_only_set(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _moo_problem())
    problem = build_multiobjective_problem(public)
    payload = {
        "solutions": [
            {"variables": {"x0": 5.0, "x1": 0.0}},
            {"variables": {"x0": 0.0, "x1": 5.0}},
        ]
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    # Verification succeeds (finite evidence), but the feasibility gate
    # makes the set infeasible for the judge.
    assert verification.status.value == "verified"
    verdict = problem.judge_spec.gates
    values = {o.name: o.value for o in verification.evidence}
    assert values["feasible_count"] == 0.0
    assert values["hypervolume"] == 0.0
    assert any(
        gate.name == "feasible_solutions"
        and not gate.holds(values[gate.observation])
        for gate in verdict
    )


def test_context_invariant(tmp_path: Path) -> None:
    from dataclasses import replace

    contract = validate_moo_data(
        _write_problem(tmp_path / "data", _moo_problem())
    )
    with pytest.raises(ValueError, match="at least one variable"):
        MooVerificationContext(replace(contract, variables=()))
    with pytest.raises(ValueError, match="exactly two"):
        MooVerificationContext(
            replace(contract, objectives=contract.objectives[:1])
        )
    with pytest.raises(ValueError, match="tolerance"):
        MooVerificationContext(replace(contract, tolerance=0.0))
    context = MooVerificationContext(contract)
    assert len(context.fingerprint()) == 64
