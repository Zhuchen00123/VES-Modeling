"""R10: optimization API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ves_modeling.optimization import (
    ApplyOptimizationResult,
    OptimizationSearchResult,
    apply_optimization_solution,
    capabilities,
    run_optimization_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "optimization"
MILP_CODE = (FIXTURES / "milp_solution.py").read_text(encoding="utf-8")


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _lp_problem() -> dict:
    return {
        "version": 1,
        "sense": "minimize",
        "variables": {
            "x0": {"type": "continuous", "lower": 0.0, "upper": 10.0},
            "x1": {"type": "continuous", "lower": 0.0, "upper": 10.0},
        },
        "objective": {
            "coefficients": {"x0": 1.0, "x1": 2.0},
            "constant": 3.0,
        },
        "constraints": [
            {
                "coefficients": {"x0": 1.0, "x1": 1.0},
                "sense": "<=",
                "rhs": 5.0,
            },
            {
                "coefficients": {"x0": 2.0, "x1": -1.0},
                "sense": ">=",
                "rhs": 0.0,
            },
            {
                "coefficients": {"x1": 1.0},
                "sense": "==",
                "rhs": 1.0,
            },
        ],
    }


def _milp_problem() -> dict:
    return {
        "version": 1,
        "sense": "maximize",
        "variables": {
            "n": {"type": "integer", "lower": 0.0, "upper": 10.0},
            "b": {"type": "binary", "lower": 0.0, "upper": 1.0},
        },
        "objective": {
            "coefficients": {"n": 2.0, "b": 5.0},
            "constant": 1.0,
        },
        "constraints": [
            {
                "coefficients": {"n": 1.0, "b": 3.0},
                "sense": "<=",
                "rhs": 7.0,
            }
        ],
    }


def test_run_optimization_search_lp_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _lp_problem())
    result = run_optimization_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, OptimizationSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_feasible is True
    assert result.best_objective == pytest.approx(5.5)
    assert result.best_max_bound_violation == 0.0
    assert result.best_max_constraint_violation == 0.0
    assert result.best_integrality_violation == 0.0
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()
    candidates = list((result.run_dir / "candidates").iterdir())
    assert len(candidates) == 3
    assert all((candidate / "run.json").is_file() for candidate in candidates)


def test_run_optimization_search_milp_maximize_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _milp_problem())
    result = run_optimization_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert result.status == "verified"
    assert result.best_feasible is True
    assert result.best_objective == pytest.approx(15.0)  # n=7, b=0
    assert result.data_contract["sense"] == "maximize"


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _lp_problem())
    result = run_optimization_search(
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
    assert persisted["task"] == "optimization"
    assert persisted["status"] == "verified"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert set(provenance["inputs"]) == {"public"}
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["public"]["problem.json"],
    )


def test_apply_produced_unverified_with_host_facts(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _milp_problem())
    result = apply_optimization_solution(
        MILP_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyOptimizationResult)
    assert result.status == "produced_unverified"
    assert result.runner == "local"
    assert result.solutions_path is not None
    assert result.solutions_path.is_file()
    assert result.feasible is True
    assert result.objective == pytest.approx(15.0)
    assert result.max_bound_violation == 0.0
    assert result.max_constraint_violation == 0.0
    assert result.integrality_violation == 0.0
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert set(payload["variables"]) == {"n", "b"}
    summary = result.to_summary()
    json.dumps(summary)
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data", _milp_problem())
    bad_code = MILP_CODE.replace(
        '    json.dump(solution, fh)',
        '    solution["variables"]["ghost"] = 1.0\n'
        '    json.dump(solution, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_optimization_solution(
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
    run_json = json.loads(
        (run_dir / "candidate" / "run.json").read_text(encoding="utf-8")
    )
    assert run_json["status"] == "artifact_invalid"


def test_run_optimization_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data", _lp_problem())
    with pytest.raises(ValueError, match="unknown generator"):
        run_optimization_search(
            public,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_optimization() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_optimization_search",
        "apply_optimization_solution",
    ]
    assert declaration["apply_statuses"] == ["produced_unverified"]
    assert "objective" in declaration["verified_metrics"]
    assert declaration["data_contract"]["sense"] == ["minimize", "maximize"]
