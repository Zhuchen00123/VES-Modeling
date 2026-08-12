"""R27: sequential-pattern data contract (sequences/patterns/metrics)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.seqpattern.data_contract import (
    evaluate_patterns,
    load_hidden_sequences,
    validate_patterns,
    validate_seqpattern_data,
)
from ves_modeling.seqpattern.problem import build_seqpattern_problem

TRAIN_SEQUENCES = [
    ("a", "b", "c"),
    ("a", "b", "d"),
    ("a", "b", "e"),
    ("a", "c", "b"),
    ("a", "b", "c"),
    ("a", "b", "d"),
    ("a", "b", "e"),
    ("a", "c", "b"),
    ("a", "b", "c"),
    ("a", "b", "d"),
]


def _write_data(
    root: Path,
    *,
    train: list[tuple[str, ...]] | None = None,
    hidden: list[tuple[str, ...]] | None = None,
) -> tuple[Path, Path]:
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train = train if train is not None else TRAIN_SEQUENCES
    hidden = hidden if hidden is not None else [("a", "b", "x")] * 5
    train_rows = [
        (sid, step, event)
        for sid, sequence in enumerate(train)
        for step, event in enumerate(sequence)
    ]
    hidden_rows = [
        (sid, step, event)
        for sid, sequence in enumerate(hidden)
        for step, event in enumerate(sequence)
    ]
    pd.DataFrame(train_rows, columns=["sequence_id", "step", "event"]).to_csv(
        public / "train.csv", index=False
    )
    pd.DataFrame(
        hidden_rows, columns=["sequence_id", "step", "event"]
    ).to_csv(host / "hidden_test_sequences.csv", index=False)
    return public, host


def test_valid_contract(tmp_path: Path) -> None:
    public, host = _write_data(tmp_path / "d")
    contract = validate_seqpattern_data(public)
    assert len(contract.train_sequences) == 10
    assert {"a", "b", "c", "d", "e"} <= set(contract.event_set)
    hidden = load_hidden_sequences(host)
    assert len(hidden) == 5
    json.dumps(contract.to_dict())


def test_train_attacks(tmp_path: Path) -> None:
    public, _ = _write_data(tmp_path / "d", train=TRAIN_SEQUENCES[:9])
    with pytest.raises(ValueError, match="at least 10 sequences"):
        validate_seqpattern_data(public)

    short_train = [("a", "b")] * 10
    public, _ = _write_data(tmp_path / "d2", train=short_train)
    with pytest.raises(ValueError, match="at least 3 steps"):
        validate_seqpattern_data(public)

    public, _ = _write_data(tmp_path / "d3")
    frame = pd.read_csv(public / "train.csv")
    frame.loc[3, "step"] = frame.loc[2, "step"]
    frame.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_seqpattern_data(public)

    public, _ = _write_data(
        tmp_path / "d4", train=[("True", "False", "True")] * 10
    )
    with pytest.raises(ValueError, match="must not be boolean"):
        validate_seqpattern_data(public)

    public, _ = _write_data(
        tmp_path / "d5", train=[(1.5, 2.5, 3.5)] * 10
    )
    with pytest.raises(ValueError, match="integer or string"):
        validate_seqpattern_data(public)


def test_hidden_validation(tmp_path: Path) -> None:
    _, host = _write_data(tmp_path / "d")
    frame = pd.read_csv(host / "hidden_test_sequences.csv")
    frame.loc[3, "step"] = frame.loc[2, "step"]
    frame.to_csv(host / "hidden_test_sequences.csv", index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_hidden_sequences(host)

    pd.DataFrame(columns=["sequence_id", "step", "event"]).to_csv(
        host / "hidden_test_sequences.csv", index=False
    )
    with pytest.raises(ValueError, match="must not be empty"):
        load_hidden_sequences(host)

    pd.DataFrame(columns=["sequence_id", "step"]).to_csv(
        host / "hidden_test_sequences.csv", index=False
    )
    with pytest.raises(ValueError, match="exactly columns"):
        load_hidden_sequences(host)


def test_pattern_validation(tmp_path: Path) -> None:
    public, _ = _write_data(tmp_path / "d")
    contract = validate_seqpattern_data(public)
    event_set = contract.event_set
    with pytest.raises(ValueError, match="missing required field"):
        validate_patterns({}, event_set=event_set)
    with pytest.raises(ValueError, match="at least one pattern"):
        validate_patterns({"patterns": []}, event_set=event_set)
    with pytest.raises(ValueError, match="must be an object"):
        validate_patterns({"patterns": [["a"]]}, event_set=event_set)
    with pytest.raises(ValueError, match="unknown fields"):
        validate_patterns(
            {"patterns": [{"prefix": ["a"], "suffix": ["b"], "x": 1}]},
            event_set=event_set,
        )
    with pytest.raises(ValueError, match="'prefix' and 'suffix'"):
        validate_patterns(
            {"patterns": [{"prefix": ["a"]}]}, event_set=event_set
        )
    with pytest.raises(ValueError, match="non-empty list"):
        validate_patterns(
            {"patterns": [{"prefix": [], "suffix": ["b"]}]},
            event_set=event_set,
        )
    with pytest.raises(ValueError, match="disjoint"):
        validate_patterns(
            {"patterns": [{"prefix": ["a"], "suffix": ["a"]}]},
            event_set=event_set,
        )
    with pytest.raises(ValueError, match="not in the train event set"):
        validate_patterns(
            {"patterns": [{"prefix": ["a"], "suffix": ["zz"]}]},
            event_set=event_set,
        )
    with pytest.raises(ValueError, match="duplicates"):
        validate_patterns(
            {
                "patterns": [
                    {"prefix": ["a"], "suffix": ["b"]},
                    {"prefix": ["a"], "suffix": ["b"]},
                ]
            },
            event_set=event_set,
        )
    patterns = validate_patterns(
        {"patterns": [{"prefix": ["a"], "suffix": ["b"]}]},
        event_set=event_set,
    )
    assert patterns == [(("a",), ("b",))]


def test_metrics_on_hidden(tmp_path: Path) -> None:
    metrics = evaluate_patterns(
        [(("a",), ("b",)), (("b",), ("c",)), (("b",), ("d",))],
        [("a", "b", "x")] * 5,
    )
    assert metrics["evaluable_pattern_count"] == 3
    assert metrics["pattern_count"] == 3
    assert metrics["mean_lift"] == pytest.approx(1 / 3)
    assert metrics["mean_confidence"] == pytest.approx(1 / 3)

    metrics = evaluate_patterns(
        [(("c",), ("d",))],
        [("a", "b", "x")] * 5,
    )
    assert metrics["evaluable_pattern_count"] == 0
    assert metrics["mean_lift"] == 0.0

    metrics = evaluate_patterns(
        [(("b",), ("x",))],
        [("a", "b", "x")] * 5,
    )
    assert metrics["mean_lift"] == 1.0


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="patterns.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_verifier_and_claims_ignored(tmp_path: Path) -> None:
    public, host = _write_data(tmp_path / "d")
    problem = build_seqpattern_problem(public, host)
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(
        _artifact(
            {
                "patterns": [
                    {"prefix": ["a"], "suffix": ["b"]},
                    {"prefix": ["b"], "suffix": ["c"]},
                    {"prefix": ["b"], "suffix": ["d"]},
                ],
                "claimed_mean_lift": 0.0,
                "claimed_mean_confidence": 0.0,
            }
        )
    )
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["mean_lift"] == pytest.approx(1 / 3)
    assert values["evaluable_pattern_count"] == 3.0


def test_no_evaluable_pattern_fails_gate(tmp_path: Path) -> None:
    public, host = _write_data(tmp_path / "d")
    problem = build_seqpattern_problem(public, host)
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(
        _artifact({"patterns": [{"prefix": ["c"], "suffix": ["d"]}]})
    )
    assert verification.status.value == "verified"
    assert verification.evidence is not None
    violated = [
        gate
        for gate in problem.judge_spec.gates
        if gate.violated_by(verification.evidence)
    ]
    assert any(gate.name == "at_least_one_evaluable" for gate in violated)
