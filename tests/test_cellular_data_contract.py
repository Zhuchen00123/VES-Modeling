"""R29: cellular-automaton data contract (instance/estimate/metrics)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.cellular.data_contract import (
    validate_cellular_data,
    validate_estimate,
)
from ves_modeling.cellular.problem import build_cellular_problem
from ves_modeling.cellular.verifier import (
    compute_cellular_metrics,
    reference_ca_value,
)


def _problem() -> dict:
    return {
        "version": 1,
        "rule": 110,
        "width": 30,
        "steps": 10,
        "initial": [0] * 14 + [1] + [0] * 15,
        "quantity": "final_density",
    }


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def test_valid_contract(tmp_path: Path) -> None:
    contract = validate_cellular_data(_write_problem(tmp_path / "d", _problem()))
    assert contract.rule == 110
    assert contract.width == 30
    assert contract.steps == 10
    assert contract.quantity == "final_density"
    json.dumps(contract.to_dict())


def test_schema_attacks(tmp_path: Path) -> None:
    base = _problem()
    bad = dict(base, rule=256)
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        validate_cellular_data(_write_problem(tmp_path / "r", bad))
    bad = dict(base, rule=-1)
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        validate_cellular_data(_write_problem(tmp_path / "r2", bad))
    bad = dict(base, width=19)
    with pytest.raises(ValueError, match=r"\[20, 200\]"):
        validate_cellular_data(_write_problem(tmp_path / "w", bad))
    bad = dict(base, steps=0)
    with pytest.raises(ValueError, match=r"\[1, 200\]"):
        validate_cellular_data(_write_problem(tmp_path / "s", bad))
    bad = dict(base, initial=[0] * 29)
    with pytest.raises(ValueError, match="exactly 30 entries"):
        validate_cellular_data(_write_problem(tmp_path / "i", bad))
    bad = dict(base, initial=[0] * 14 + [2] + [0] * 15)
    with pytest.raises(ValueError, match="0 or 1"):
        validate_cellular_data(_write_problem(tmp_path / "i2", bad))
    bad = dict(base, initial=[0] * 30)
    with pytest.raises(ValueError, match="at least one 1"):
        validate_cellular_data(_write_problem(tmp_path / "i3", bad))
    bad = dict(base, initial=[0] * 14 + [True] + [0] * 15)
    with pytest.raises(ValueError, match="must be an integer"):
        validate_cellular_data(_write_problem(tmp_path / "bool", bad))
    bad = dict(base, quantity="cell_state")
    with pytest.raises(ValueError, match="requires 'index'"):
        validate_cellular_data(_write_problem(tmp_path / "q", bad))
    bad = dict(base, quantity="cell_state", index=30)
    with pytest.raises(ValueError, match=r"\[0, width\)"):
        validate_cellular_data(_write_problem(tmp_path / "idx", bad))
    bad = dict(base, index=3)
    with pytest.raises(ValueError, match="only valid"):
        validate_cellular_data(_write_problem(tmp_path / "idx2", bad))
    bad = dict(base, extra=1)
    with pytest.raises(ValueError, match="unknown top-level"):
        validate_cellular_data(_write_problem(tmp_path / "extra", bad))

    public = tmp_path / "dup"
    public.mkdir()
    text = json.dumps(_problem()).replace(
        '"steps": 10',
        '"steps": 10,\n    "steps": 10',
    )
    (public / "problem.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        validate_cellular_data(public)


def test_estimate_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required field"):
        validate_estimate({})
    with pytest.raises(ValueError, match="finite number"):
        validate_estimate({"estimate": True})
    with pytest.raises(ValueError, match="finite"):
        validate_estimate({"estimate": float("nan")})
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


def test_metrics(tmp_path: Path) -> None:
    metrics = compute_cellular_metrics(0.45, None, 0.5)
    assert metrics["absolute_error"] == pytest.approx(0.05)
    assert metrics["relative_error"] == pytest.approx(0.1)
    assert metrics["ci_coverage"] == 0.0
    metrics = compute_cellular_metrics(0.45, (0.0, 0.5), 0.5)
    assert metrics["ci_coverage"] == 1.0
    metrics = compute_cellular_metrics(0.1, None, 0.0)
    assert metrics["relative_error"] == metrics["absolute_error"]


def test_reference_known_rules(tmp_path: Path) -> None:
    zero = dict(_problem(), rule=0)
    contract = validate_cellular_data(_write_problem(tmp_path / "z", zero))
    assert reference_ca_value(contract) == 0.0

    all_ones = dict(_problem(), rule=255)
    contract = validate_cellular_data(_write_problem(tmp_path / "o", all_ones))
    assert reference_ca_value(contract) == 1.0

    persistent = dict(_problem(), rule=255, quantity="persistent_ones")
    contract = validate_cellular_data(
        _write_problem(tmp_path / "p", persistent)
    )
    assert reference_ca_value(contract) == 1.0

    cell = dict(_problem(), rule=255, quantity="cell_state", index=7)
    contract = validate_cellular_data(_write_problem(tmp_path / "c", cell))
    assert reference_ca_value(contract) == 1.0


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_verifier_and_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", dict(_problem(), rule=255))
    problem = build_cellular_problem(public)
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(
        _artifact(
            {
                "estimate": 1.0,
                "claimed_reference": 0.0,
                "claimed_relative_error": 99.0,
            }
        )
    )
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["absolute_error"] == pytest.approx(0.0)
    assert values["relative_error"] == pytest.approx(0.0)
