"""R19: queueing data contract (kinds/quantities/reference/CI)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.queueing.context import QueueingVerificationContext
from ves_modeling.queueing.data_contract import (
    compute_reference,
    validate_queueing_data,
    validate_solution,
)
from ves_modeling.queueing.problem import build_queueing_problem


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _mm1_problem(quantity: str = "mean_wait", threshold: float | None = None) -> dict:
    problem = {
        "version": 1,
        "kind": "mm1",
        "lambda": 2.0,
        "mu": 4.0,
        "quantity": quantity,
    }
    if threshold is not None:
        problem["threshold"] = threshold
    return problem


def _mmc_problem(quantity: str = "mean_wait") -> dict:
    return {
        "version": 1,
        "kind": "mmc",
        "lambda": 4.0,
        "mu": 3.0,
        "c": 2,
        "quantity": quantity,
    }


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contracts_and_references(tmp_path: Path) -> None:
    mm1 = validate_queueing_data(
        _write_problem(tmp_path / "mm1", _mm1_problem("mean_wait"))
    )
    rho = 2.0 / 4.0
    assert compute_reference(mm1) == pytest.approx(rho / (4.0 * (1 - rho)))
    mm1q = validate_queueing_data(
        _write_problem(tmp_path / "mm1q", _mm1_problem("mean_queue"))
    )
    assert compute_reference(mm1q) == pytest.approx(rho**2 / (1 - rho))
    mm1u = validate_queueing_data(
        _write_problem(tmp_path / "mm1u", _mm1_problem("mean_utilization"))
    )
    assert compute_reference(mm1u) == pytest.approx(rho)
    mm1p = validate_queueing_data(
        _write_problem(
            tmp_path / "mm1p", _mm1_problem("prob_wait_gt", 1.0)
        )
    )
    assert compute_reference(mm1p) == pytest.approx(
        rho * math.exp(-4.0 * (1 - rho) * 1.0)
    )
    mmc = validate_queueing_data(
        _write_problem(tmp_path / "mmc", _mmc_problem("mean_wait"))
    )
    c = 2
    rho_c = 4.0 / (3.0 * c)
    term = (c * rho_c) ** c / (math.factorial(c) * (1 - rho_c))
    series = sum((c * rho_c) ** k / math.factorial(k) for k in range(c))
    p_wait = term / (series + term)
    assert compute_reference(mmc) == pytest.approx(
        p_wait / (c * 3.0 * (1 - rho_c))
    )


def test_schema_attacks(tmp_path: Path) -> None:
    base = _mm1_problem()
    bad = dict(base, kind="mg1")
    with pytest.raises(ValueError, match="kind"):
        validate_queueing_data(_write_problem(tmp_path / "k", bad))
    bad = dict(base, **{"lambda": 0.0})
    with pytest.raises(ValueError, match="positive"):
        validate_queueing_data(_write_problem(tmp_path / "l", bad))
    bad = dict(base, **{"lambda": 10.0})  # rho = 2.5 >= 1
    with pytest.raises(ValueError, match="stable"):
        validate_queueing_data(_write_problem(tmp_path / "r", bad))
    bad = dict(base, quantity="mean_wait", c=2)
    with pytest.raises(ValueError, match="must not declare 'c'"):
        validate_queueing_data(_write_problem(tmp_path / "c", bad))
    bad = dict(base, quantity="mean_wait", threshold=1.0)
    with pytest.raises(ValueError, match="must not declare 'threshold'"):
        validate_queueing_data(_write_problem(tmp_path / "t", bad))
    bad = dict(base, quantity="prob_wait_gt")
    with pytest.raises(ValueError, match="requires 'threshold'"):
        validate_queueing_data(_write_problem(tmp_path / "p", bad))
    bad = dict(base, quantity="prob_wait_gt", threshold=-1.0)
    with pytest.raises(ValueError, match=">= 0"):
        validate_queueing_data(_write_problem(tmp_path / "n", bad))
    mmc = _mmc_problem()
    bad = dict(mmc, c=0)
    with pytest.raises(ValueError, match="'c' must be an integer >= 1"):
        validate_queueing_data(_write_problem(tmp_path / "c0", bad))


def test_solution_validation(tmp_path: Path) -> None:
    contract = validate_queueing_data(
        _write_problem(tmp_path / "data", _mm1_problem())
    )
    estimate, ci = validate_solution(
        {"estimate": 0.5, "confidence_interval": [0.4, 0.6]}, contract
    )
    assert estimate == 0.5
    assert ci == (0.4, 0.6)
    with pytest.raises(ValueError, match="missing required field"):
        validate_solution({}, contract)
    with pytest.raises(ValueError, match="finite"):
        validate_solution({"estimate": float("nan")}, contract)
    with pytest.raises(ValueError, match="must lie within"):
        validate_solution(
            {"estimate": 1.0, "confidence_interval": [0.4, 0.6]},
            contract,
        )


def test_metrics_and_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _mm1_problem("mean_wait"))
    problem = build_queueing_problem(public)
    reference = compute_reference(validate_queueing_data(public))
    payload = {
        "estimate": reference,
        "confidence_interval": [reference - 0.01, reference + 0.01],
        "reference": 0.0,
        "lambda": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["absolute_error"] == pytest.approx(0.0)
    assert values["relative_error"] == pytest.approx(0.0)
    assert values["ci_coverage"] == 1.0


def test_context_invariant(tmp_path: Path) -> None:
    from dataclasses import replace

    contract = validate_queueing_data(
        _write_problem(tmp_path / "data", _mm1_problem())
    )
    with pytest.raises(ValueError, match="reference"):
        QueueingVerificationContext(float("nan"), contract)
    with pytest.raises(ValueError, match="kind"):
        QueueingVerificationContext(1.0, replace(contract, kind="x"))
    context = QueueingVerificationContext(1.0, contract)
    assert len(context.fingerprint()) == 64
