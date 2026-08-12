"""R14: graph API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ves_modeling.graph import (
    ApplyGraphResult,
    GraphSearchResult,
    apply_graph_solution,
    capabilities,
    run_graph_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "graph"
DIJKSTRA_CODE = (FIXTURES / "dijkstra_shortest_path.py").read_text(
    encoding="utf-8"
)


def _write_problem(root: Path, problem: dict) -> Path:
    public = root / "public"
    public.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(problem), encoding="utf-8"
    )
    return public


def _shortest_path_problem() -> dict:
    return {
        "version": 1,
        "problem_type": "shortest_path",
        "nodes": ["a", "b", "c", "d"],
        "edges": [
            {"u": "a", "v": "b", "weight": 1.0},
            {"u": "b", "v": "c", "weight": 2.0},
            {"u": "a", "v": "c", "weight": 5.0},
            {"u": "c", "v": "d", "weight": 1.0},
        ],
        "source": "a",
        "target": "d",
    }


def _max_flow_problem() -> dict:
    return {
        "version": 1,
        "problem_type": "max_flow",
        "nodes": ["s", "a", "b", "t"],
        "edges": [
            {"u": "s", "v": "a", "weight": 3.0},
            {"u": "s", "v": "b", "weight": 2.0},
            {"u": "a", "v": "b", "weight": 1.0},
            {"u": "a", "v": "t", "weight": 3.0},
            {"u": "b", "v": "t", "weight": 2.0},
        ],
        "source": "s",
        "target": "t",
    }


def _mst_problem() -> dict:
    return {
        "version": 1,
        "problem_type": "min_spanning_tree",
        "nodes": ["a", "b", "c", "d"],
        "edges": [
            {"u": "a", "v": "b", "weight": 1.0},
            {"u": "b", "v": "c", "weight": 2.0},
            {"u": "c", "v": "d", "weight": 3.0},
            {"u": "a", "v": "d", "weight": 10.0},
            {"u": "b", "v": "d", "weight": 4.0},
        ],
    }


def test_run_graph_search_shortest_path_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _shortest_path_problem())
    result = run_graph_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, GraphSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_feasible is True
    assert result.best_total_weight == pytest.approx(4.0)
    assert result.best_path_violation == 0.0
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_run_graph_search_max_flow_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _max_flow_problem())
    result = run_graph_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert result.status == "verified"
    assert result.best_feasible is True
    assert result.best_total_value == pytest.approx(5.0)
    assert result.best_capacity_violation == 0.0
    assert result.best_conservation_violation == 0.0


def test_run_graph_search_mst_verified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _mst_problem())
    result = run_graph_search(
        public,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert result.status == "verified"
    assert result.best_feasible is True
    assert result.best_total_weight == pytest.approx(6.0)
    assert result.best_tree_violation == 0.0


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _mst_problem())
    result = run_graph_search(
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
    assert persisted["task"] == "graph"
    assert persisted["problem_type"] == "min_spanning_tree"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert set(provenance["inputs"]) == {"public"}
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["public"]["problem.json"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _shortest_path_problem())
    result = apply_graph_solution(
        DIJKSTRA_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyGraphResult)
    assert result.status == "produced_unverified"
    assert result.solutions_path is not None
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert payload["path"][0] == "a"
    assert payload["path"][-1] == "d"
    assert not hasattr(result, "best_total_weight")
    summary = result.to_summary()
    json.dumps(summary)
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public = _write_problem(tmp_path / "data", _shortest_path_problem())
    bad_code = DIJKSTRA_CODE.replace(
        'json.dump({"path": path}, fh)',
        'json.dump({"path": path[:-1]}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_graph_solution(
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


def test_run_graph_search_rejects_unknown_generator(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _mst_problem())
    with pytest.raises(ValueError, match="unknown generator"):
        run_graph_search(
            public,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_graph() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_graph_search",
        "apply_graph_solution",
    ]
    assert declaration["data_contract"]["problem_types"] == [
        "shortest_path",
        "max_flow",
        "min_spanning_tree",
    ]
