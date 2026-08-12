"""R27: sequential-pattern API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from ves_modeling.seqpattern import (
    ApplySeqPatternResult,
    SeqPatternSearchResult,
    apply_seqpattern_solution,
    capabilities,
    run_seqpattern_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "seqpattern"
PREFIXSPAN_CODE = (FIXTURES / "seq_prefixspan.py").read_text(encoding="utf-8")

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


def _write_data(root: Path) -> tuple[Path, Path]:
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train_rows = [
        (sid, step, event)
        for sid, sequence in enumerate(TRAIN_SEQUENCES)
        for step, event in enumerate(sequence)
    ]
    hidden_rows = [
        (sid, step, event)
        for sid, sequence in enumerate([("a", "b", "x")] * 5)
        for step, event in enumerate(sequence)
    ]
    pd.DataFrame(train_rows, columns=["sequence_id", "step", "event"]).to_csv(
        public / "train.csv", index=False
    )
    pd.DataFrame(
        hidden_rows, columns=["sequence_id", "step", "event"]
    ).to_csv(host / "hidden_test_sequences.csv", index=False)
    return public, host


def test_run_seqpattern_search_verified(tmp_path: Path) -> None:
    public, host = _write_data(tmp_path / "data")
    result = run_seqpattern_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, SeqPatternSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_mean_lift is not None and result.best_mean_lift > 0.0
    assert (
        result.best_mean_confidence is not None
        and result.best_mean_confidence > 0.0
    )
    assert result.best_evaluable_pattern_count >= 1.0
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _write_data(tmp_path / "data")
    result = run_seqpattern_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == result.to_summary()
    assert persisted["task"] == "seqpattern"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["public"]["train.csv"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public, _ = _write_data(tmp_path / "data")
    result = apply_seqpattern_solution(
        PREFIXSPAN_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplySeqPatternResult)
    assert result.status == "produced_unverified"
    assert result.solutions_path is not None
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert len(payload["patterns"]) >= 1
    assert not hasattr(result, "best_mean_lift")
    summary = result.to_summary()
    json.dumps(summary)
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public, _ = _write_data(tmp_path / "data")
    bad_code = PREFIXSPAN_CODE.replace(
        'json.dump({"patterns": patterns}, fh)',
        'json.dump({"patterns": []}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_seqpattern_solution(
            bad_code,
            public,
            workspace=tmp_path / "runs",
            trusted_code=True,
            run_id=run_id,
        )
    run_dir = tmp_path / "runs" / run_id
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "artifact_invalid"
    assert summary["error"] is not None


def test_run_seqpattern_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public, host = _write_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_seqpattern_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_seqpattern() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_seqpattern_search",
        "apply_seqpattern_solution",
    ]
    assert "mean_lift" in declaration["verified_metrics"]
    assert "evaluable_pattern_count" in declaration[
        "evaluation_observations"
    ]
