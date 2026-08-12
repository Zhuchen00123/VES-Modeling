"""R31: game API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ves_modeling.game import (
    ApplyGameResult,
    GameSearchResult,
    apply_game_solution,
    capabilities,
    run_game_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "game"
RICCATI_CODE = (FIXTURES / "game_riccati.py").read_text(encoding="utf-8")


def _write_problem(root: Path) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(
            {
                "version": 1,
                "A": [[1.0]],
                "B": [[1.0]],
                "C": [[1.0]],
                "Q": [[1.0]],
                "R": [[1.0]],
                "S": [[1.0]],
                "x0": [1.0],
                "horizon": 2,
            }
        ),
        encoding="utf-8",
    )
    return public


def test_run_game_search_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data")
    result = run_game_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, GameSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_total_cost == pytest.approx(3.0)
    assert result.best_reference_optimal_cost == pytest.approx(3.0)
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data")
    result = run_game_search(
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
    assert persisted["task"] == "game"
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
    result = apply_game_solution(
        RICCATI_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyGameResult)
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
        apply_game_solution(
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


def test_run_game_search_rejects_unknown_generator(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_game_search(
            public,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_game() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_game_search",
        "apply_game_solution",
    ]
    assert "total_cost" in declaration["verified_metrics"]
    assert declaration["audit_observations"] == ["reference_optimal_cost"]
