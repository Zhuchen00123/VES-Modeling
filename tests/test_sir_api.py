"""R28: SIR API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ves_modeling.sir import (
    ApplySirResult,
    SirSearchResult,
    apply_sir_solution,
    capabilities,
    run_sir_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sir"
RK4_CODE = (FIXTURES / "sir_rk4.py").read_text(encoding="utf-8")


def _write_problem(root: Path) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    return public


def test_run_sir_search_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data")
    result = run_sir_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, SirSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_relative_error is not None
    assert result.best_relative_error < 0.05
    assert result.best_absolute_error is not None
    assert result.best_ci_coverage is not None
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data")
    result = run_sir_search(
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
    assert persisted["task"] == "sir"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["public"]["problem.json"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data")
    result = apply_sir_solution(
        RK4_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplySirResult)
    assert result.status == "produced_unverified"
    assert result.solutions_path is not None
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert "estimate" in payload
    assert not hasattr(result, "best_relative_error")
    summary = result.to_summary()
    json.dumps(summary)
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data")
    bad_code = RK4_CODE.replace(
        'json.dump({"estimate": estimate}, fh)',
        'json.dump({"confidence_interval": [0.0, 1.0]}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_sir_solution(
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


def test_run_sir_search_rejects_unknown_generator(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_sir_search(
            public,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_sir() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_sir_search",
        "apply_sir_solution",
    ]
    assert "relative_error" in declaration["verified_metrics"]
    assert declaration["audit_observations"] == ["ci_coverage"]
