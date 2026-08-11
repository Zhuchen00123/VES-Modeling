"""R10: optimization data contract (problem schema + solution artifact)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.optimization.context import OptimizationVerificationContext
from ves_modeling.optimization.data_contract import (
    validate_optimization_data,
    validate_solution,
)
from ves_modeling.optimization.problem import build_optimization_problem
from ves_modeling.optimization.verifier import OptimizationVerifier


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _lp_problem() -> dict:
    return {
        "version": 1,
        "sense": "minimize",
        "variables": {
            "x0": {"type": "continuous", "lower": 0.0, "upper": 10.0},
            "x1": {"type": "continuous", "lower": 0.0, "upper": 10.0},
        },
        "objective": {
            "coefficients": {"x0": 1.0, "x1": 2.0},
            "constant": 3.0,
        },
        "constraints": [
            {
                "coefficients": {"x0": 1.0, "x1": 1.0},
                "sense": "<=",
                "rhs": 5.0,
            },
            {
                "coefficients": {"x0": 2.0, "x1": -1.0},
                "sense": ">=",
                "rhs": 0.0,
            },
            {
                "coefficients": {"x1": 1.0},
                "sense": "==",
                "rhs": 1.0,
            },
        ],
    }


def _milp_problem() -> dict:
    return {
        "version": 1,
        "sense": "maximize",
        "variables": {
            "n": {"type": "integer", "lower": 0.0, "upper": 10.0},
            "b": {"type": "binary", "lower": 0.0, "upper": 1.0},
        },
        "objective": {
            "coefficients": {"n": 2.0, "b": 5.0},
            "constant": 1.0,
        },
        "constraints": [
            {
                "coefficients": {"n": 1.0, "b": 3.0},
                "sense": "<=",
                "rhs": 7.0,
            }
        ],
    }


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_lp_contract(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _lp_problem())
    contract = validate_optimization_data(public)
    assert contract.version == 1
    assert contract.sense == "minimize"
    assert contract.n_variables == 2
    assert contract.n_constraints == 3
    assert contract.variable_names == ("x0", "x1")
    assert contract.tolerance == 1e-6
    json.dumps(contract.to_dict())


def test_valid_milp_contract(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _milp_problem())
    contract = validate_optimization_data(public)
    assert contract.sense == "maximize"
    assert contract.n_constraints == 1


def test_schema_attacks(tmp_path: Path) -> None:
    base = _lp_problem()
    # bad sense
    bad = dict(base, sense="maximise")
    with pytest.raises(ValueError, match="sense"):
        validate_optimization_data(_write_problem(tmp_path / "bad-sense", bad))
    # bad version
    bad = dict(base, version=0)
    with pytest.raises(ValueError, match="version"):
        validate_optimization_data(_write_problem(tmp_path / "bad-ver", bad))
    bad = dict(base, version="1")
    with pytest.raises(ValueError, match="version"):
        validate_optimization_data(_write_problem(tmp_path / "bad-ver2", bad))
    # empty variables
    bad = dict(base, variables={})
    with pytest.raises(ValueError, match="at least one variable"):
        validate_optimization_data(_write_problem(tmp_path / "empty-var", bad))
    # bad type
    variables = dict(base["variables"])
    variables["x0"] = dict(variables["x0"], type="integerish")
    bad = dict(base, variables=variables)
    with pytest.raises(ValueError, match="type"):
        validate_optimization_data(_write_problem(tmp_path / "bad-type", bad))
    # missing bounds
    variables = dict(base["variables"])
    variables["x0"] = {"type": "continuous"}
    bad = dict(base, variables=variables)
    with pytest.raises(ValueError, match="lower and upper"):
        validate_optimization_data(_write_problem(tmp_path / "no-bounds", bad))
    # lower > upper
    variables = dict(base["variables"])
    variables["x0"] = {"type": "continuous", "lower": 5.0, "upper": 1.0}
    bad = dict(base, variables=variables)
    with pytest.raises(ValueError, match="lower must not exceed"):
        validate_optimization_data(_write_problem(tmp_path / "inv-bounds", bad))
    # binary bounds
    variables = {"b": {"type": "binary", "lower": 0.0, "upper": 2.0}}
    bad = dict(base, variables=variables, objective={"coefficients": {"b": 1.0}})
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        validate_optimization_data(_write_problem(tmp_path / "bin-bounds", bad))
    # non-finite bound
    variables = dict(base["variables"])
    variables["x0"] = {"type": "continuous", "lower": float("nan"), "upper": 1.0}
    bad = dict(base, variables=variables)
    with pytest.raises(ValueError, match="finite"):
        validate_optimization_data(_write_problem(tmp_path / "nan-bound", bad))


def test_objective_and_constraint_attacks(tmp_path: Path) -> None:
    base = _lp_problem()
    # unknown variable in objective
    bad = dict(
        base,
        objective={"coefficients": {"ghost": 1.0}},
    )
    with pytest.raises(ValueError, match="undeclared variable"):
        validate_optimization_data(_write_problem(tmp_path / "obj-ghost", bad))
    # non-finite coefficient
    bad = dict(
        base,
        objective={"coefficients": {"x0": float("inf")}},
    )
    with pytest.raises(ValueError, match="finite"):
        validate_optimization_data(_write_problem(tmp_path / "obj-inf", bad))
    # unknown variable in constraint
    constraints = [dict(base["constraints"][0], coefficients={"ghost": 1.0})]
    bad = dict(base, constraints=constraints)
    with pytest.raises(ValueError, match="undeclared variable"):
        validate_optimization_data(_write_problem(tmp_path / "c-ghost", bad))
    # bad constraint sense
    constraints = [dict(base["constraints"][0], sense="<")]
    bad = dict(base, constraints=constraints)
    with pytest.raises(ValueError, match="sense"):
        validate_optimization_data(_write_problem(tmp_path / "c-sense", bad))
    # constraints not a list
    bad = dict(base, constraints={})
    with pytest.raises(ValueError, match="array"):
        validate_optimization_data(_write_problem(tmp_path / "c-list", bad))


def test_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir(parents=True)
    text = json.dumps(_lp_problem()).replace(
        '"x1": {"type": "continuous", "lower": 0.0, "upper": 10.0}',
        '"x0": {"type": "continuous", "lower": 0.0, "upper": 10.0},'
        '"x0": {"type": "continuous", "lower": 0.0, "upper": 10.0}',
    )
    (public / "problem.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        validate_optimization_data(public)


def test_tolerance_validation(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _lp_problem())
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="tolerance"):
            validate_optimization_data(public, tolerance=bad)


def test_validate_solution_exact_variables(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _lp_problem())
    contract = validate_optimization_data(public)
    good = {"variables": {"x0": 1.0, "x1": 1.0}}
    values = validate_solution(good, contract)
    assert values == {"x0": 1.0, "x1": 1.0}
    with pytest.raises(ValueError, match="missing="):
        validate_solution({"variables": {"x0": 1.0}}, contract)
    with pytest.raises(ValueError, match="extra="):
        validate_solution(
            {"variables": {"x0": 1.0, "x1": 1.0, "ghost": 2.0}},
            contract,
        )
    with pytest.raises(ValueError, match="finite"):
        validate_solution(
            {"variables": {"x0": float("nan"), "x1": 1.0}}, contract
        )
    with pytest.raises(ValueError, match="finite"):
        validate_solution(
            {"variables": {"x0": True, "x1": 1.0}}, contract
        )


def test_verifier_feasible_and_metrics(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _lp_problem())
    contract = validate_optimization_data(public)
    context = OptimizationVerificationContext(
        contract, dataset_name="unit"
    )
    verifier = OptimizationVerifier()
    evidence = verifier.verify(
        _artifact({"variables": {"x0": 1.0, "x1": 1.0}}), context
    )
    values = {o.name: o.value for o in evidence.observations}
    assert values["max_bound_violation"] == 0.0
    assert values["max_constraint_violation"] == 0.0
    assert values["integrality_violation"] == 0.0
    assert values["objective"] == pytest.approx(6.0)


def test_verifier_detects_violations(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _lp_problem())
    contract = validate_optimization_data(public)
    context = OptimizationVerificationContext(contract)
    verifier = OptimizationVerifier()
    # Bound violation (x0 < lower 0).
    values = {
        o.name: o.value
        for o in verifier.verify(
            _artifact({"variables": {"x0": -1.0, "x1": 1.0}}), context
        ).observations
    }
    assert values["max_bound_violation"] == pytest.approx(1.0)
    # Constraint violation (x0 + x1 > 5).
    values = {
        o.name: o.value
        for o in verifier.verify(
            _artifact({"variables": {"x0": 5.0, "x1": 5.0}}), context
        ).observations
    }
    assert values["max_constraint_violation"] == pytest.approx(5.0)
    # Equality constraint violation (x1 != 1).
    values = {
        o.name: o.value
        for o in verifier.verify(
            _artifact({"variables": {"x0": 0.0, "x1": 2.0}}), context
        ).observations
    }
    # x1 != 1 violates the equality by 1; 2*x0 - x1 = -2 also violates >= 0.
    assert values["max_constraint_violation"] == pytest.approx(2.0)


def test_integrality_violation(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _milp_problem())
    contract = validate_optimization_data(public)
    context = OptimizationVerificationContext(contract)
    verifier = OptimizationVerifier()
    values = {
        o.name: o.value
        for o in verifier.verify(
            _artifact({"variables": {"n": 2.5, "b": 1.0}}), context
        ).observations
    }
    assert values["integrality_violation"] == pytest.approx(0.5)


def test_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _milp_problem())
    problem = build_optimization_problem(public)
    payload = {
        "variables": {"n": 2.0, "b": 1.0},
        "objective": -9999.0,
        "feasibility": True,
        "optimality": "optimal",
        "gap": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["objective"] == pytest.approx(10.0)  # host, not -9999
    assert values["max_bound_violation"] == 0.0
    assert values["max_constraint_violation"] == 0.0
    assert values["integrality_violation"] == 0.0


def test_context_invariant(tmp_path: Path) -> None:
    from dataclasses import replace

    public = _write_problem(tmp_path / "data", _lp_problem())
    contract = validate_optimization_data(public)
    with pytest.raises(ValueError, match="at least one variable"):
        OptimizationVerificationContext(replace(contract, variables=()))
    with pytest.raises(ValueError, match="sense"):
        OptimizationVerificationContext(replace(contract, sense="min"))
    with pytest.raises(ValueError, match="tolerance"):
        OptimizationVerificationContext(replace(contract, tolerance=0.0))
    context = OptimizationVerificationContext(contract)
    assert len(context.fingerprint()) == 64
