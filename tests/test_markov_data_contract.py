"""R23: markov data contract (states/matrix/reference)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.markov.context import MarkovVerificationContext
from ves_modeling.markov.data_contract import (
    compute_reference,
    load_hidden_parameters,
    validate_markov_data,
    validate_solution,
)
from ves_modeling.markov.problem import build_markov_problem


def _make_data(
    root: Path,
    *,
    quantity: str = "transition_probability",
    from_state: str = "a",
    to_state: str = "b",
    state: str = "a",
    seed: int = 7,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    p = np.asarray(
        [[0.7, 0.3], [0.4, 0.6]], dtype=float
    )
    current = 0
    rows = []
    for _ in range(200):
        rows.append({"sequence_id": 1, "state": ["a", "b"][current]})
        current = int(rng.choice(2, p=p[current]))
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    problem: dict = {
        "version": 1,
        "quantity": quantity,
        "states": ["a", "b"],
    }
    if quantity == "transition_probability":
        problem["from_state"] = from_state
        problem["to_state"] = to_state
    else:
        problem["state"] = state
    (public / "problem.json").write_text(json.dumps(problem), encoding="utf-8")
    pd.DataFrame(rows).to_csv(public / "train.csv", index=False)
    (host / "hidden_parameters.json").write_text(
        json.dumps(
            {"transition_matrix": p.tolist()}, encoding="utf-8"
        )
        if False
        else json.dumps({"transition_matrix": p.tolist()}),
        encoding="utf-8",
    )
    return public, host


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contracts_and_references(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data", quantity="transition_probability")
    contract = validate_markov_data(public)
    assert contract.n_states == 2
    p = load_hidden_parameters(host, contract)
    assert compute_reference(contract, p) == pytest.approx(0.3)
    public2, host2 = _make_data(tmp_path / "data2", quantity="steady_state", state="a")
    contract2 = validate_markov_data(public2)
    p2 = load_hidden_parameters(host2, contract2)
    # Stationary distribution of [[0.7,0.3],[0.4,0.6]]: pi = (4/7, 3/7)
    assert compute_reference(contract2, p2) == pytest.approx(4.0 / 7.0)
    public3, host3 = _make_data(
        tmp_path / "data3", quantity="expected_recurrence_time", state="a"
    )
    contract3 = validate_markov_data(public3)
    assert compute_reference(
        contract3, load_hidden_parameters(host3, contract3)
    ) == pytest.approx(7.0 / 4.0)


def test_schema_attacks(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    problem = json.loads((public / "problem.json").read_text())
    bad = dict(problem, quantity="hitting_time")
    (public / "problem.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="quantity"):
        validate_markov_data(public)
    bad = dict(problem, quantity="transition_probability")
    bad.pop("to_state")
    (public / "problem.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="requires from_state"):
        validate_markov_data(public)
    bad = dict(problem, quantity="steady_state")
    bad.pop("from_state", None)
    bad.pop("to_state", None)
    (public / "problem.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="requires 'state'"):
        validate_markov_data(public)
    bad = dict(problem, quantity="transition_probability", state="a")
    (public / "problem.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="must not declare 'state'"):
        validate_markov_data(public)
    (public / "problem.json").write_text(json.dumps(problem))
    train = pd.read_csv(public / "train.csv")
    train = train.head(10)
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="at least 50"):
        validate_markov_data(public)


def test_hidden_matrix_attacks(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    contract = validate_markov_data(public)
    bad = {"transition_matrix": [[0.7, 0.3], [0.4, 0.5]]}
    (host / "hidden_parameters.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="sum to 1"):
        load_hidden_parameters(host, contract)
    bad = {"transition_matrix": [[0.7, 0.3], [0.0, 1.0]]}  # not irreducible
    (host / "hidden_parameters.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="irreducible"):
        load_hidden_parameters(host, contract)


def test_solution_validation(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    contract = validate_markov_data(public)
    estimate, ci = validate_solution(
        {"estimate": 0.3, "confidence_interval": [0.2, 0.4]}, contract
    )
    assert estimate == 0.3
    assert ci == (0.2, 0.4)
    with pytest.raises(ValueError, match="missing required field"):
        validate_solution({}, contract)
    with pytest.raises(ValueError, match="finite"):
        validate_solution({"estimate": float("nan")}, contract)


def test_metrics_and_claims_ignored(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    problem = build_markov_problem(public, host)
    payload = {
        "estimate": 0.3,
        "confidence_interval": [0.29, 0.31],
        "reference": 0.0,
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

    public, _host = _make_data(tmp_path / "data")
    contract = validate_markov_data(public)
    with pytest.raises(ValueError, match="reference"):
        MarkovVerificationContext(float("nan"), contract)
    with pytest.raises(ValueError, match="quantity"):
        MarkovVerificationContext(1.0, replace(contract, quantity="x"))
    context = MarkovVerificationContext(1.0, contract)
    assert len(context.fingerprint()) == 64
