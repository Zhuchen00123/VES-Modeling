"""R15: Monte Carlo data contract (kinds/reference/solution/CI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.montecarlo.context import MonteCarloVerificationContext
from ves_modeling.montecarlo.data_contract import (
    compute_reference,
    validate_montecarlo_data,
    validate_solution,
)
from ves_modeling.montecarlo.problem import build_montecarlo_problem
from ves_modeling.montecarlo.verifier import MonteCarloVerifier


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _expectation_problem(
    target: str = "mean", threshold: float | None = None
) -> dict:
    params: dict = {
        "outcomes": [0.0, 1.0, 2.0],
        "probabilities": [0.2, 0.5, 0.3],
        "target": target,
    }
    if threshold is not None:
        params["threshold"] = threshold
    return {"version": 1, "kind": "expectation", "params": params}


def _integral_problem() -> dict:
    return {
        "version": 1,
        "kind": "integral",
        "params": {
            "integrand": "polynomial",
            "coefficients": [1.0, 2.0, 3.0],
            "interval": [0.0, 1.0],
        },
    }


def _probability_problem(event: str = "ge", threshold: int = 2) -> dict:
    return {
        "version": 1,
        "kind": "probability",
        "params": {
            "distribution": "binomial",
            "n": 10,
            "p": 0.5,
            "event": event,
            "threshold": threshold,
        },
    }


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contracts_and_references(tmp_path: Path) -> None:
    exp = validate_montecarlo_data(
        _write_problem(tmp_path / "exp", _expectation_problem("mean"))
    )
    assert exp.kind == "expectation"
    assert compute_reference(exp) == pytest.approx(1.1)
    exp_var = validate_montecarlo_data(
        _write_problem(tmp_path / "var", _expectation_problem("variance"))
    )
    assert compute_reference(exp_var) == pytest.approx(0.49)
    exp_ge = validate_montecarlo_data(
        _write_problem(tmp_path / "ge", _expectation_problem("prob_ge", 1.5))
    )
    assert compute_reference(exp_ge) == pytest.approx(0.3)
    integ = validate_montecarlo_data(
        _write_problem(tmp_path / "int", _integral_problem())
    )
    # integral of 1 + 2x + 3x^2 over [0,1] = 1 + 1 + 1 = 3
    assert compute_reference(integ) == pytest.approx(3.0)
    prob = validate_montecarlo_data(
        _write_problem(tmp_path / "prob", _probability_problem())
    )
    # P(X >= 2) for Binomial(10, 0.5)
    from scipy.stats import binom

    assert compute_reference(prob) == pytest.approx(binom.sf(1, 10, 0.5))


def test_expectation_schema_attacks(tmp_path: Path) -> None:
    base = _expectation_problem()
    bad = dict(base, params=dict(base["params"], probabilities=[0.1, 0.1, 0.1]))
    with pytest.raises(ValueError, match="sum to 1"):
        validate_montecarlo_data(_write_problem(tmp_path / "sum", bad))
    bad = dict(base, params=dict(base["params"], probabilities=[0.2, 0.5]))
    with pytest.raises(ValueError, match="same length"):
        validate_montecarlo_data(_write_problem(tmp_path / "len", bad))
    bad = dict(base, params=dict(base["params"], probabilities=[0.2, -0.5, 1.3]))
    with pytest.raises(ValueError, match="non-negative"):
        validate_montecarlo_data(_write_problem(tmp_path / "neg", bad))
    bad = dict(base, params=dict(base["params"], target="mode"))
    with pytest.raises(ValueError, match="target"):
        validate_montecarlo_data(_write_problem(tmp_path / "tgt", bad))
    bad = dict(base, params=dict(base["params"], target="prob_ge"))
    with pytest.raises(ValueError, match="requires 'threshold'"):
        validate_montecarlo_data(_write_problem(tmp_path / "thr", bad))
    bad = dict(base, params=dict(base["params"], threshold=1.0))
    with pytest.raises(ValueError, match="must not declare"):
        validate_montecarlo_data(_write_problem(tmp_path / "thr2", bad))
    bad = dict(base, params=dict(base["params"], outcomes=[0.0, 1.0]))
    with pytest.raises(ValueError, match="at least two outcomes|same length"):
        validate_montecarlo_data(_write_problem(tmp_path / "out", bad))


def test_integral_and_probability_schema_attacks(tmp_path: Path) -> None:
    base = _integral_problem()
    bad = dict(base, params=dict(base["params"], integrand="exp"))
    with pytest.raises(ValueError, match="'polynomial'"):
        validate_montecarlo_data(_write_problem(tmp_path / "i1", bad))
    bad = dict(base, params=dict(base["params"], interval=[1.0, 1.0]))
    with pytest.raises(ValueError, match="a < b"):
        validate_montecarlo_data(_write_problem(tmp_path / "i2", bad))
    bad = dict(base, params=dict(base["params"], coefficients=[]))
    with pytest.raises(ValueError, match="non-empty"):
        validate_montecarlo_data(_write_problem(tmp_path / "i3", bad))
    prob = _probability_problem()
    bad = dict(prob, params=dict(prob["params"], n=0))
    with pytest.raises(ValueError, match="'n' must be an integer >= 1"):
        validate_montecarlo_data(_write_problem(tmp_path / "p1", bad))
    bad = dict(prob, params=dict(prob["params"], p=1.0))
    with pytest.raises(ValueError, match="within \\(0, 1\\)"):
        validate_montecarlo_data(_write_problem(tmp_path / "p2", bad))
    bad = dict(prob, params=dict(prob["params"], event="gt"))
    with pytest.raises(ValueError, match="event"):
        validate_montecarlo_data(_write_problem(tmp_path / "p3", bad))
    bad = dict(prob, params=dict(prob["params"], threshold=11))
    with pytest.raises(ValueError, match="threshold"):
        validate_montecarlo_data(_write_problem(tmp_path / "p4", bad))


def test_solution_validation(tmp_path: Path) -> None:
    contract = validate_montecarlo_data(
        _write_problem(tmp_path / "data", _expectation_problem())
    )
    estimate, ci = validate_solution(
        {"estimate": 1.0, "confidence_interval": [0.5, 1.5]}, contract
    )
    assert estimate == 1.0
    assert ci == (0.5, 1.5)
    estimate, ci = validate_solution({"estimate": 1.0}, contract)
    assert ci is None
    with pytest.raises(ValueError, match="missing required field"):
        validate_solution({}, contract)
    with pytest.raises(ValueError, match="finite"):
        validate_solution({"estimate": float("nan")}, contract)
    with pytest.raises(ValueError, match="must be a finite number"):
        validate_solution({"estimate": True}, contract)
    with pytest.raises(ValueError, match="lo, hi"):
        validate_solution(
            {"estimate": 1.0, "confidence_interval": [0.5]}, contract
        )
    with pytest.raises(ValueError, match="lo must not exceed"):
        validate_solution(
            {"estimate": 1.0, "confidence_interval": [1.5, 0.5]}, contract
        )
    with pytest.raises(ValueError, match="must lie within"):
        validate_solution(
            {"estimate": 2.0, "confidence_interval": [0.5, 1.5]}, contract
        )


def test_verifier_metrics_and_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(
        tmp_path / "data", _expectation_problem("mean")
    )
    problem = build_montecarlo_problem(public)
    payload = {
        "estimate": 1.1,
        "confidence_interval": [1.05, 1.15],
        "reference": 0.0,
        "claimed_error": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["absolute_error"] == pytest.approx(0.0)
    assert values["relative_error"] == pytest.approx(0.0)
    assert values["ci_coverage"] == 1.0


def test_relative_error_zero_reference_falls_back(tmp_path: Path) -> None:
    problem = {
        "version": 1,
        "kind": "integral",
        "params": {
            "integrand": "polynomial",
            "coefficients": [0.0],
            "interval": [-1.0, 1.0],
        },
    }
    public = _write_problem(tmp_path / "data", problem)
    contract = validate_montecarlo_data(public)
    assert compute_reference(contract) == 0.0
    context = MonteCarloVerificationContext(
        compute_reference(contract), contract
    )
    verifier = MonteCarloVerifier()
    evidence = verifier.verify(_artifact({"estimate": 0.5}), context)
    values = {o.name: o.value for o in evidence.observations}
    assert values["absolute_error"] == pytest.approx(0.5)
    assert values["relative_error"] == pytest.approx(0.5)
    assert values["ci_coverage"] == 0.0


def test_ci_coverage_audit(tmp_path: Path) -> None:
    contract = validate_montecarlo_data(
        _write_problem(tmp_path / "data", _expectation_problem())
    )
    context = MonteCarloVerificationContext(
        compute_reference(contract), contract
    )
    verifier = MonteCarloVerifier()
    missed = verifier.verify(
        _artifact({"estimate": 1.0, "confidence_interval": [0.5, 1.05]}),
        context,
    )
    values = {o.name: o.value for o in missed.observations}
    assert values["ci_coverage"] == 0.0


def test_context_invariant(tmp_path: Path) -> None:
    contract = validate_montecarlo_data(
        _write_problem(tmp_path / "data", _expectation_problem())
    )
    with pytest.raises(ValueError, match="reference"):
        MonteCarloVerificationContext(float("nan"), contract)
    from dataclasses import replace

    with pytest.raises(ValueError, match="kind"):
        MonteCarloVerificationContext(
            1.0, replace(contract, kind="bad")
        )
    context = MonteCarloVerificationContext(
        compute_reference(contract), contract
    )
    assert context.reference == pytest.approx(1.1)
    assert len(context.fingerprint()) == 64
