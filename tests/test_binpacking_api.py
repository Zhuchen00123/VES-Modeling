"""R24: bin packing API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ves_modeling.binpacking import (
    ApplyBinResult,
    BinSearchResult,
    apply_binpacking_solution,
    capabilities,
    run_binpacking_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "binpacking"
FFD_CODE = (FIXTURES / "bin_ffd.py").read_text(encoding="utf-8")


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _problem() -> dict:
    return {
        "version": 1,
        "capacity": 10.0,
        "items": [6.0, 5.0, 4.0, 3.0, 2.0, 4.0, 5.0],
        "n_items": 7,
    }


def test_run_binpacking_search_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _problem())
    result = run_binpacking_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, BinSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_bin_count is not None
    assert result.best_bin_count >= 1.0
    assert result.best_capacity_violation == 0.0
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _problem())
    result = run_binpacking_search(
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
    assert persisted["task"] == "binpacking"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["public"]["problem.json"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _problem())
    result = apply_binpacking_solution(
        FFD_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyBinResult)
    assert result.status == "produced_unverified"
    assert result.solutions_path is not None
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert len(payload["assignment"]) == 7
    assert not hasattr(result, "best_bin_count")
    summary = result.to_summary()
    json.dumps(summary)
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data", _problem())
    bad_code = FFD_CODE.replace(
        'json.dump({"assignment": assignment}, fh)',
        'json.dump({"assignment": assignment[:-1]}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_binpacking_solution(
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


def test_run_binpacking_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data", _problem())
    with pytest.raises(ValueError, match="unknown generator"):
        run_binpacking_search(
            public,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_binpacking() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_binpacking_search",
        "apply_binpacking_solution",
    ]
    assert "bin_count" in declaration["verified_metrics"]
