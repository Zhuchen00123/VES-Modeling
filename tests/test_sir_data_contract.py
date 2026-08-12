"""R28: SIR data contract (instance/estimate/metrics/reference)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.sir.data_contract import (
    validate_estimate,
    validate_sir_data,
)
from ves_modeling.sir.problem import build_sir_problem
from ves_modeling.sir.verifier import (
    compute_sir_metrics,
    reference_sir_value,
)


def _problem() -> dict:
    return {
        "version": 1,
        "model": "sir",
        "beta": 0.5,
        "gamma": 0.1,
        "N": 1000,
        "i0": 5,
        "r0": 0,
        "t_end": 50.0,
        "quantity": "final_size",
    }


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def test_valid_contract(tmp_path: Path) -> None:
    contract = validate_sir_data(_write_problem(tmp_path / "d", _problem()))
    assert contract.to_dict()["model"] == "sir"
    assert contract.N == 1000
    assert contract.quantity == "final_size"
    assert contract.t is None
    json.dumps(contract.to_dict())


def test_schema_attacks(tmp_path: Path) -> None:
    base = _problem()
    bad = dict(base, model="seir")
    with pytest.raises(ValueError, match="must be 'sir'"):
        validate_sir_data(_write_problem(tmp_path / "m", bad))
    bad = dict(base, beta=0.0)
    with pytest.raises(ValueError, match="positive"):
        validate_sir_data(_write_problem(tmp_path / "b", bad))
    bad = dict(base, gamma=-0.1)
    with pytest.raises(ValueError, match="positive"):
        validate_sir_data(_write_problem(tmp_path / "g", bad))
    bad = dict(base, N=99)
    with pytest.raises(ValueError, match=">= 100"):
        validate_sir_data(_write_problem(tmp_path / "n", bad))
    bad = dict(base, i0=0)
    with pytest.raises(ValueError, match=">= 1"):
        validate_sir_data(_write_problem(tmp_path / "i", bad))
    bad = dict(base, r0=-1)
    with pytest.raises(ValueError, match=">= 0"):
        validate_sir_data(_write_problem(tmp_path / "r", bad))
    bad = dict(base, i0=800, r0=300)
    with pytest.raises(ValueError, match="must not exceed"):
        validate_sir_data(_write_problem(tmp_path / "ir", bad))
    bad = dict(base, t_end=0.0)
    with pytest.raises(ValueError, match="positive"):
        validate_sir_data(_write_problem(tmp_path / "t", bad))
    bad = dict(base, quantity="unknown")
    with pytest.raises(ValueError, match="quantity"):
        validate_sir_data(_write_problem(tmp_path / "q", bad))
    bad = dict(base, quantity="infected_at")
    with pytest.raises(ValueError, match="requires 't'"):
        validate_sir_data(_write_problem(tmp_path / "tq", bad))
    bad = dict(base, quantity="infected_at", t=60.0)
    with pytest.raises(ValueError, match=r"\(0, t_end\]"):
        validate_sir_data(_write_problem(tmp_path / "tout", bad))
    bad = dict(base, t=10.0)
    with pytest.raises(ValueError, match="only valid"):
        validate_sir_data(_write_problem(tmp_path / "tbad", bad))
    bad = dict(base, beta=True)
    with pytest.raises(ValueError, match="finite number"):
        validate_sir_data(_write_problem(tmp_path / "bool", bad))
    bad = dict(base, gamma=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        validate_sir_data(_write_problem(tmp_path / "nan", bad))
    bad = dict(base, extra=1)
    with pytest.raises(ValueError, match="unknown top-level"):
        validate_sir_data(_write_problem(tmp_path / "extra", bad))

    public = tmp_path / "dup"
    public.mkdir()
    text = json.dumps(_problem()).replace(
        '"t_end": 50.0',
        '"t_end": 50.0,\n    "t_end": 50.0',
    )
    (public / "problem.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        validate_sir_data(public)


def test_estimate_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required field"):
        validate_estimate({})
    with pytest.raises(ValueError, match="finite number"):
        validate_estimate({"estimate": True})
    with pytest.raises(ValueError, match="finite"):
        validate_estimate({"estimate": float("inf")})
    with pytest.raises(ValueError, match="JSON array"):
        validate_estimate({"estimate": 0.5, "confidence_interval": "x"})
    with pytest.raises(ValueError, match="exactly 2 entries"):
        validate_estimate({"estimate": 0.5, "confidence_interval": [0.0]})
    with pytest.raises(ValueError, match="must not exceed hi"):
        validate_estimate(
            {"estimate": 0.5, "confidence_interval": [0.6, 0.4]}
        )
    with pytest.raises(ValueError, match="within confidence_interval"):
        validate_estimate(
            {"estimate": 0.8, "confidence_interval": [0.0, 0.5]}
        )
    estimate, ci = validate_estimate(
        {"estimate": 0.5, "confidence_interval": [0.4, 0.6]}
    )
    assert estimate == 0.5
    assert ci == (0.4, 0.6)
    estimate, ci = validate_estimate({"estimate": 0.5})
    assert ci is None


def test_metrics(tmp_path: Path) -> None:
    metrics = compute_sir_metrics(0.45, None, 0.5)
    assert metrics["absolute_error"] == pytest.approx(0.05)
    assert metrics["relative_error"] == pytest.approx(0.1)
    assert metrics["ci_coverage"] == 0.0
    metrics = compute_sir_metrics(0.45, (0.0, 0.5), 0.5)
    assert metrics["ci_coverage"] == 1.0
    metrics = compute_sir_metrics(0.45, (0.0, 0.4), 0.5)
    assert metrics["ci_coverage"] == 0.0
    metrics = compute_sir_metrics(0.1, None, 0.0)
    assert metrics["relative_error"] == metrics["absolute_error"]


def test_reference_computed(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", _problem())
    contract = validate_sir_data(public)
    reference = reference_sir_value(contract)
    assert 0.0 < reference <= 1.0

    peak = dict(_problem(), quantity="peak_infected")
    contract = validate_sir_data(_write_problem(tmp_path / "p", peak))
    reference = reference_sir_value(contract)
    assert reference >= 5 / 1000
    assert reference <= 1.0

    infected = dict(_problem(), quantity="infected_at", t=10.0)
    contract = validate_sir_data(_write_problem(tmp_path / "i", infected))
    reference = reference_sir_value(contract)
    assert 0.0 <= reference <= 1.0


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_verifier_and_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", _problem())
    problem = build_sir_problem(public)
    contract = validate_sir_data(public)
    reference = reference_sir_value(contract)
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(
        _artifact(
            {
                "estimate": reference,
                "claimed_reference": 0.0,
                "claimed_relative_error": 99.0,
            }
        )
    )
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["absolute_error"] == pytest.approx(0.0)
    assert values["relative_error"] == pytest.approx(0.0)
