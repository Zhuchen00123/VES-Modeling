"""R30: network-SIR data contract (instance/estimate/metrics/reference)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.networksir.data_contract import (
    validate_estimate,
    validate_networksir_data,
)
from ves_modeling.networksir.problem import build_networksir_problem
from ves_modeling.networksir.verifier import (
    compute_networksir_metrics,
    reference_networksir_value,
)


def _problem() -> dict:
    return {
        "version": 1,
        "model": "network_sir",
        "beta": 0.3,
        "gamma": 0.1,
        "n_nodes": 10,
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]],
        "i0": 1,
        "t_end": 20,
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
    contract = validate_networksir_data(
        _write_problem(tmp_path / "d", _problem())
    )
    assert contract.to_dict()["model"] == "network_sir"
    assert contract.n_nodes == 10
    assert contract.to_dict()["edge_count"] == 10
    assert contract.quantity == "final_size"
    json.dumps(contract.to_dict())


def test_schema_attacks(tmp_path: Path) -> None:
    base = _problem()
    bad = dict(base, model="sir")
    with pytest.raises(ValueError, match="must be 'network_sir'"):
        validate_networksir_data(_write_problem(tmp_path / "m", bad))
    bad = dict(base, beta=0.0)
    with pytest.raises(ValueError, match="positive"):
        validate_networksir_data(_write_problem(tmp_path / "b", bad))
    bad = dict(base, n_nodes=9)
    with pytest.raises(ValueError, match=r"\[10, 100\]"):
        validate_networksir_data(_write_problem(tmp_path / "n", bad))
    bad = dict(base, edges=[[0, 0]])
    with pytest.raises(ValueError, match="self-loop"):
        validate_networksir_data(_write_problem(tmp_path / "e", bad))
    bad = dict(base, edges=[[0, 10]])
    with pytest.raises(ValueError, match="endpoints"):
        validate_networksir_data(_write_problem(tmp_path / "e2", bad))
    bad = dict(base, edges=[[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="duplicates"):
        validate_networksir_data(_write_problem(tmp_path / "e3", bad))
    bad = dict(base, edges=[0])
    with pytest.raises(ValueError, match="must be a pair"):
        validate_networksir_data(_write_problem(tmp_path / "e4", bad))
    bad = dict(base, edges=[[0, 1, 2]])
    with pytest.raises(ValueError, match="exactly 2 entries"):
        validate_networksir_data(_write_problem(tmp_path / "e5", bad))
    bad = dict(base, i0=0)
    with pytest.raises(ValueError, match=">= 1"):
        validate_networksir_data(_write_problem(tmp_path / "i", bad))
    bad = dict(base, i0=11)
    with pytest.raises(ValueError, match="must not exceed"):
        validate_networksir_data(_write_problem(tmp_path / "i2", bad))
    bad = dict(base, t_end=0.0)
    with pytest.raises(ValueError, match="positive"):
        validate_networksir_data(_write_problem(tmp_path / "t", bad))
    bad = dict(base, quantity="unknown")
    with pytest.raises(ValueError, match="quantity"):
        validate_networksir_data(_write_problem(tmp_path / "q", bad))
    bad = dict(base, quantity="infected_at")
    with pytest.raises(ValueError, match="requires 't'"):
        validate_networksir_data(_write_problem(tmp_path / "q2", bad))
    bad = dict(base, quantity="infected_at", t=25.0)
    with pytest.raises(ValueError, match=r"\(0, t_end\]"):
        validate_networksir_data(_write_problem(tmp_path / "q3", bad))
    bad = dict(base, t=5.0)
    with pytest.raises(ValueError, match="only valid"):
        validate_networksir_data(_write_problem(tmp_path / "q4", bad))
    bad = dict(base, beta=True)
    with pytest.raises(ValueError, match="finite number"):
        validate_networksir_data(_write_problem(tmp_path / "bool", bad))
    bad = dict(base, extra=1)
    with pytest.raises(ValueError, match="unknown top-level"):
        validate_networksir_data(_write_problem(tmp_path / "extra", bad))

    public = tmp_path / "dup"
    public.mkdir()
    text = json.dumps(_problem()).replace(
        '"i0": 1',
        '"i0": 1,\n    "i0": 1',
    )
    (public / "problem.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        validate_networksir_data(public)


def test_estimate_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required field"):
        validate_estimate({})
    with pytest.raises(ValueError, match="finite number"):
        validate_estimate({"estimate": True})
    with pytest.raises(ValueError, match="finite"):
        validate_estimate({"estimate": float("inf")})
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
    metrics = compute_networksir_metrics(0.45, None, 0.5)
    assert metrics["absolute_error"] == pytest.approx(0.05)
    assert metrics["relative_error"] == pytest.approx(0.1)
    assert metrics["ci_coverage"] == 0.0
    metrics = compute_networksir_metrics(0.45, (0.0, 0.5), 0.5)
    assert metrics["ci_coverage"] == 1.0
    metrics = compute_networksir_metrics(0.1, None, 0.0)
    assert metrics["relative_error"] == metrics["absolute_error"]


def test_reference_empty_graph(tmp_path: Path) -> None:
    empty = dict(_problem(), edges=[])
    contract = validate_networksir_data(
        _write_problem(tmp_path / "e", empty)
    )
    assert reference_networksir_value(contract) == pytest.approx(1 / 10)
    peak = dict(empty, quantity="peak_infected")
    contract = validate_networksir_data(
        _write_problem(tmp_path / "p", peak)
    )
    assert reference_networksir_value(contract) == pytest.approx(1 / 10)


def test_reference_deterministic_and_range(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", _problem())
    contract = validate_networksir_data(public)
    first = reference_networksir_value(contract)
    second = reference_networksir_value(contract)
    assert first == second
    assert 0.0 < first <= 1.0


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_verifier_and_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "d", dict(_problem(), edges=[]))
    problem = build_networksir_problem(public)
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(
        _artifact(
            {
                "estimate": 0.1,
                "claimed_reference": 0.0,
                "claimed_relative_error": 99.0,
            }
        )
    )
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["absolute_error"] == pytest.approx(0.0)
    assert values["relative_error"] == pytest.approx(0.0)
