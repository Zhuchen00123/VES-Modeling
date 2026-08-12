"""R26: LQR API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ves_modeling.lqr import (
    ApplyLqrResult,
    LqrSearchResult,
    apply_lqr_solution,
    capabilities,
    run_lqr_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lqr"
RICCATI_CODE = (FIXTURES / "lqr_riccati.py").read_text(encoding="utf-8")


def _write_problem(root: Path) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(
            {
                "version": 1,
                "A": [[1.0]],
                "B": [[1.0]],
                "Q": [[1.0]],
                "R": [[1.0]],
                "x0": [1.0],
                "horizon": 2,
            }
        ),
        encoding="utf-8",
    )
    return public


def test_run_lqr_search_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data")
    result = run_lqr_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, LqrSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_total_cost == pytest.approx(1.6)
    assert result.best_reference_optimal_cost == pytest.approx(1.6)
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data")
    result = run_lqr_search(
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
    assert persisted["task"] == "lqr"
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
    result = apply_lqr_solution(
        RICCATI_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyLqrResult)
    assert result.status == "produced_unverified"
    assert result.solutions_path is not None
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert len(payload["control"]) == 2
    assert not hasattr(result, "best_total_cost")
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
    bad_code = RICCATI_CODE.replace(
        'json.dump({"control": control}, fh)',
        'json.dump({"control": control[:-1]}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_lqr_solution(
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


def test_run_lqr_search_rejects_unknown_generator(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_lqr_search(
            public,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_lqr() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_lqr_search",
        "apply_lqr_solution",
    ]
    assert "total_cost" in declaration["verified_metrics"]
    assert declaration["audit_observations"] == ["reference_optimal_cost"]
