"""R18: probabilistic data contract (problem/samples/parameters/reference)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.probabilistic.context import (
    ProbabilisticVerificationContext,
)
from ves_modeling.probabilistic.data_contract import (
    compute_reference,
    load_hidden_parameters,
    validate_probabilistic_data,
    validate_solution,
)
from ves_modeling.probabilistic.problem import build_probabilistic_problem
from ves_modeling.probabilistic.verifier import ProbabilisticVerifier


def _make_data(
    root: Path,
    *,
    family: str = "normal",
    quantity: str = "mean",
    params: dict | None = None,
    q: float | None = None,
    threshold: float | None = None,
    n: int = 200,
    seed: int = 7,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    if family == "normal":
        params = params or {"mean": 3.0, "std": 2.0}
        samples = rng.normal(params["mean"], params["std"], size=n)
    elif family == "gamma":
        params = params or {"shape": 2.0, "scale": 3.0}
        samples = rng.gamma(params["shape"], scale=params["scale"], size=n)
    else:
        params = params or {"alpha": 2.0, "beta": 5.0}
        samples = rng.beta(params["alpha"], params["beta"], size=n)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    problem: dict = {
        "version": 1,
        "family": family,
        "quantity": quantity,
    }
    if q is not None:
        problem["q"] = q
    if threshold is not None:
        problem["threshold"] = threshold
    (public / "problem.json").write_text(json.dumps(problem), encoding="utf-8")
    pd.DataFrame({"value": samples}).to_csv(
        public / "train.csv", index=False
    )
    (host / "hidden_parameters.json").write_text(
        json.dumps(params), encoding="utf-8"
    )
    return public, host


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contracts_and_references(tmp_path: Path) -> None:
    public, host = _make_data(
        tmp_path / "normal-mean", quantity="mean"
    )
    contract = validate_probabilistic_data(public)
    assert contract.family == "normal"
    assert contract.n_samples == 200
    parameters = load_hidden_parameters(host, contract)
    assert compute_reference(contract, parameters) == pytest.approx(3.0)
    public2, host2 = _make_data(
        tmp_path / "normal-var", quantity="variance"
    )
    contract2 = validate_probabilistic_data(public2)
    assert compute_reference(contract2, load_hidden_parameters(host2, contract2)) == pytest.approx(4.0)
    public3, host3 = _make_data(
        tmp_path / "normal-quantile", quantity="quantile", q=0.5
    )
    contract3 = validate_probabilistic_data(public3)
    assert compute_reference(
        contract3, load_hidden_parameters(host3, contract3)
    ) == pytest.approx(3.0)
    public4, host4 = _make_data(
        tmp_path / "normal-prob",
        quantity="probability_ge",
        threshold=4.0,
    )
    contract4 = validate_probabilistic_data(public4)
    expected = stats.norm.sf(4.0, 3.0, 2.0)
    assert compute_reference(
        contract4, load_hidden_parameters(host4, contract4)
    ) == pytest.approx(expected)
    public5, host5 = _make_data(
        tmp_path / "gamma-mean", family="gamma", quantity="mean"
    )
    contract5 = validate_probabilistic_data(public5)
    assert compute_reference(
        contract5, load_hidden_parameters(host5, contract5)
    ) == pytest.approx(6.0)
    public6, host6 = _make_data(
        tmp_path / "beta-var", family="beta", quantity="variance"
    )
    contract6 = validate_probabilistic_data(public6)
    reference = compute_reference(
        contract6, load_hidden_parameters(host6, contract6)
    )
    expected_var = 2.0 * 5.0 / ((7.0) ** 2 * 8.0)
    assert reference == pytest.approx(expected_var)


def test_schema_attacks(tmp_path: Path) -> None:
    base = _make_data(tmp_path / "base", quantity="mean")
    public = base[0]
    problem = json.loads((public / "problem.json").read_text())
    bad = dict(problem, family="poisson")
    (public / "problem.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="family"):
        validate_probabilistic_data(public)
    bad = dict(problem, quantity="mode")
    (public / "problem.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="quantity"):
        validate_probabilistic_data(public)
    bad = dict(problem, quantity="quantile")
    (public / "problem.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="requires 'q'"):
        validate_probabilistic_data(public)
    bad = dict(problem, quantity="quantile", q=1.5)
    (public / "problem.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="within \\(0, 1\\)"):
        validate_probabilistic_data(public)
    bad = dict(problem, quantity="mean", q=0.5)
    (public / "problem.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="must not declare"):
        validate_probabilistic_data(public)
    bad = dict(problem, quantity="probability_ge")
    (public / "problem.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="requires 'threshold'"):
        validate_probabilistic_data(public)


def test_sample_and_parameter_attacks(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data", quantity="mean")
    samples = pd.read_csv(public / "train.csv")
    samples = samples.head(10)
    samples.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="at least 20"):
        validate_probabilistic_data(public)
    public2, _host2 = _make_data(tmp_path / "data2", quantity="mean")
    samples2 = pd.read_csv(public2 / "train.csv")
    samples2.loc[0, "value"] = float("nan")
    samples2.to_csv(public2 / "train.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        validate_probabilistic_data(public2)
    public3, host3 = _make_data(tmp_path / "data3", quantity="mean")
    params = json.loads((host3 / "hidden_parameters.json").read_text())
    bad = dict(params, std=-1.0)
    (host3 / "hidden_parameters.json").write_text(json.dumps(bad))
    contract = validate_probabilistic_data(public3)
    with pytest.raises(ValueError, match="positive"):
        load_hidden_parameters(host3, contract)
    bad = dict(params, extra=1.0)
    (host3 / "hidden_parameters.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="exactly"):
        load_hidden_parameters(host3, contract)


def test_solution_validation(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data", quantity="mean")
    contract = validate_probabilistic_data(public)
    estimate, ci = validate_solution(
        {"estimate": 3.0, "confidence_interval": [2.0, 4.0]}, contract
    )
    assert estimate == 3.0
    assert ci == (2.0, 4.0)
    with pytest.raises(ValueError, match="missing required field"):
        validate_solution({}, contract)
    with pytest.raises(ValueError, match="finite"):
        validate_solution({"estimate": float("nan")}, contract)
    with pytest.raises(ValueError, match="must lie within"):
        validate_solution(
            {"estimate": 5.0, "confidence_interval": [2.0, 4.0]},
            contract,
        )


def test_metrics_and_claims_ignored(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data", quantity="mean")
    problem = build_probabilistic_problem(public, host)
    payload = {
        "estimate": 3.0,
        "confidence_interval": [2.9, 3.1],
        "reference": 0.0,
        "mean": 0.0,
        "std": 0.0,
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["absolute_error"] == pytest.approx(0.0)
    assert values["relative_error"] == pytest.approx(0.0)
    assert values["ci_coverage"] == 1.0


def test_relative_error_zero_reference(tmp_path: Path) -> None:
    public, host = _make_data(
        tmp_path / "data", quantity="mean", params={"mean": 0.0, "std": 1.0}
    )
    contract = validate_probabilistic_data(public)
    parameters = load_hidden_parameters(host, contract)
    context = ProbabilisticVerificationContext(
        compute_reference(contract, parameters), contract
    )
    verifier = ProbabilisticVerifier()
    evidence = verifier.verify(_artifact({"estimate": 0.5}), context)
    values = {o.name: o.value for o in evidence.observations}
    assert values["absolute_error"] == pytest.approx(0.5)
    assert values["relative_error"] == pytest.approx(0.5)


def test_context_invariant(tmp_path: Path) -> None:
    from dataclasses import replace

    public, _host = _make_data(tmp_path / "data", quantity="mean")
    contract = validate_probabilistic_data(public)
    with pytest.raises(ValueError, match="reference"):
        ProbabilisticVerificationContext(float("nan"), contract)
    with pytest.raises(ValueError, match="family"):
        ProbabilisticVerificationContext(1.0, replace(contract, family="x"))
    context = ProbabilisticVerificationContext(1.0, contract)
    assert len(context.fingerprint()) == 64
