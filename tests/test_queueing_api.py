"""R19: queueing API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ves_modeling.queueing import (
    ApplyQueueingResult,
    QueueingSearchResult,
    apply_queueing_solution,
    capabilities,
    run_queueing_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "queueing"
MM1_CODE = (FIXTURES / "des_mm1.py").read_text(encoding="utf-8")


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _mm1_problem(quantity: str = "mean_wait") -> dict:
    return {
        "version": 1,
        "kind": "mm1",
        "lambda": 2.0,
        "mu": 4.0,
        "quantity": quantity,
    }


def _mmc_problem(quantity: str = "mean_utilization") -> dict:
    return {
        "version": 1,
        "kind": "mmc",
        "lambda": 4.0,
        "mu": 3.0,
        "c": 2,
        "quantity": quantity,
    }


def test_run_queueing_search_mm1_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _mm1_problem())
    result = run_queueing_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, QueueingSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.kind == "mm1"
    assert result.quantity == "mean_wait"
    assert result.best_relative_error is not None
    assert result.best_relative_error < 0.2
    assert result.best_ci_coverage == 1.0
    assert result.rejected == 0


def test_run_queueing_search_mmc_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _mmc_problem())
    result = run_queueing_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert result.status == "verified"
    assert result.kind == "mmc"
    assert result.quantity == "mean_utilization"
    assert result.best_relative_error is not None
    assert result.best_relative_error < 0.2


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _mm1_problem())
    result = run_queueing_search(
        public,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == result.to_summary()
    assert persisted["task"] == "queueing"
    assert "reference" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["public"]["problem.json"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _mm1_problem())
    result = apply_queueing_solution(
        MM1_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyQueueingResult)
    assert result.status == "produced_unverified"
    assert result.solutions_path is not None
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert "estimate" in payload
    assert not hasattr(result, "best_absolute_error")
    summary = result.to_summary()
    json.dumps(summary)
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data", _mm1_problem())
    bad_code = MM1_CODE.replace(
        '"estimate": estimate,',
        '"estimate": estimate + float("nan"),',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_queueing_solution(
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


def test_run_queueing_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data", _mm1_problem())
    with pytest.raises(ValueError, match="unknown generator"):
        run_queueing_search(
            public,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_queueing() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_queueing_search",
        "apply_queueing_solution",
    ]
    assert declaration["data_contract"]["kinds"] == ["mm1", "mmc"]
    assert "mean_wait" in declaration["data_contract"]["quantities"]
