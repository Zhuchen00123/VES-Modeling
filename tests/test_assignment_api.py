"""R22: assignment/TSP API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ves_modeling.assignment import (
    ApplyAssignResult,
    AssignSearchResult,
    apply_assignment_solution,
    capabilities,
    run_assignment_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "assignment"
ASSIGN_CODE = (FIXTURES / "assignment_hungarian.py").read_text(
    encoding="utf-8"
)


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _assignment_problem() -> dict:
    return {
        "version": 1,
        "problem_type": "assignment",
        "size": 4,
        "costs": [
            [4.0, 1.0, 3.0, 2.0],
            [2.0, 0.0, 5.0, 1.0],
            [3.0, 2.0, 2.0, 4.0],
            [1.0, 3.0, 4.0, 0.0],
        ],
    }


def _tsp_problem() -> dict:
    return {
        "version": 1,
        "problem_type": "tsp",
        "size": 4,
        "costs": [
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 35.0, 25.0],
            [15.0, 35.0, 0.0, 30.0],
            [20.0, 25.0, 30.0, 0.0],
        ],
        "start": 0,
    }


def test_run_assignment_search_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _assignment_problem())
    result = run_assignment_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, AssignSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.problem_type == "assignment"
    assert result.best_total_cost is not None
    assert result.best_total_cost >= 0.0
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_run_tsp_search_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _tsp_problem())
    result = run_assignment_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert result.status == "verified"
    assert result.problem_type == "tsp"
    assert result.best_total_cost is not None
    assert result.best_total_cost > 0.0


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _assignment_problem())
    result = run_assignment_search(
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
    assert persisted["task"] == "assignment"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["public"]["problem.json"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _assignment_problem())
    result = apply_assignment_solution(
        ASSIGN_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyAssignResult)
    assert result.status == "produced_unverified"
    assert result.solutions_path is not None
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert len(payload["assignment"]) == 4
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
    public = _write_problem(tmp_path / "data", _assignment_problem())
    bad_code = ASSIGN_CODE.replace(
        'json.dump({"assignment": assignment}, fh)',
        'json.dump({"assignment": assignment[:-1]}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_assignment_solution(
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


def test_run_assignment_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data", _assignment_problem())
    with pytest.raises(ValueError, match="unknown generator"):
        run_assignment_search(
            public,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_assignment() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_assignment_search",
        "apply_assignment_solution",
    ]
    assert declaration["data_contract"]["problem_types"] == [
        "assignment",
        "tsp",
    ]
