"""R16: multi-objective API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ves_modeling.multiobjective import (
    ApplyMooResult,
    MooSearchResult,
    apply_multiobjective_solution,
    capabilities,
    run_multiobjective_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "multiobjective"
SCALARIZATION_CODE = (FIXTURES / "scalarization_linprog.py").read_text(
    encoding="utf-8"
)


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _moo_problem() -> dict:
    return {
        "version": 1,
        "variables": {
            "x0": {"type": "continuous", "lower": 0.0, "upper": 2.0},
            "x1": {"type": "continuous", "lower": 0.0, "upper": 2.0},
        },
        "objectives": [
            {"coefficients": {"x0": 1.0}, "constant": 0.0},
            {"coefficients": {"x1": 1.0}, "constant": 0.0},
        ],
        "constraints": [
            {"coefficients": {"x0": 1.0, "x1": 1.0}, "sense": "<=", "rhs": 2.0}
        ],
    }


def test_run_multiobjective_search_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _moo_problem())
    result = run_multiobjective_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, MooSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_hypervolume is not None and result.best_hypervolume > 0.0
    assert result.best_feasible_count is not None
    assert result.best_feasible_count >= 1.0
    assert result.best_non_dominated_count is not None
    assert result.best_total_count is not None
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _moo_problem())
    result = run_multiobjective_search(
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
    assert persisted["task"] == "multiobjective"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["public"]["problem.json"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _moo_problem())
    result = apply_multiobjective_solution(
        SCALARIZATION_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyMooResult)
    assert result.status == "produced_unverified"
    assert result.solutions_path is not None
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert len(payload["solutions"]) >= 1
    assert not hasattr(result, "best_hypervolume")
    summary = result.to_summary()
    json.dumps(summary)
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data", _moo_problem())
    bad_code = SCALARIZATION_CODE.replace(
        '    json.dump(payload, fh)',
        '    payload["solutions"] = []\n'
        '    json.dump(payload, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_multiobjective_solution(
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


def test_run_multiobjective_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data", _moo_problem())
    with pytest.raises(ValueError, match="unknown generator"):
        run_multiobjective_search(
            public,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_multiobjective() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_multiobjective_search",
        "apply_multiobjective_solution",
    ]
    assert "hypervolume" in declaration["verified_metrics"]
    assert declaration["data_contract"]["objectives"] == (
        "exactly two linear objectives"
    )
