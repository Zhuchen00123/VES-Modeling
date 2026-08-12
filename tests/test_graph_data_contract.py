"""R14: graph data contract (problem schema + per-type solutions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.graph.context import GraphVerificationContext
from ves_modeling.graph.data_contract import (
    validate_graph_data,
    validate_solution,
)
from ves_modeling.graph.problem import build_graph_problem
from ves_modeling.graph.verifier import GraphVerifier


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


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="solution.json",
        content=json.dumps(payload),
        producer="test",
    )


def test_valid_contracts(tmp_path: Path) -> None:
    sp = validate_graph_data(_write_problem(tmp_path / "sp", _shortest_path_problem()))
    assert sp.problem_type == "shortest_path"
    assert sp.n_nodes == 4
    assert sp.n_edges == 4
    assert sp.source == "a" and sp.target == "d"
    json.dumps(sp.to_dict())
    mf = validate_graph_data(_write_problem(tmp_path / "mf", _max_flow_problem()))
    assert mf.problem_type == "max_flow"
    assert mf.n_edges == 5
    mst = validate_graph_data(_write_problem(tmp_path / "mst", _mst_problem()))
    assert mst.problem_type == "min_spanning_tree"
    assert mst.source is None


def test_schema_attacks(tmp_path: Path) -> None:
    base = _shortest_path_problem()
    for index, bad_type in enumerate(("hamiltonian", 1)):
        with pytest.raises(ValueError, match="problem_type"):
            validate_graph_data(
                _write_problem(
                    tmp_path / f"t{index}", dict(base, problem_type=bad_type)
                )
            )
    bad = dict(base, version=0)
    with pytest.raises(ValueError, match="version"):
        validate_graph_data(_write_problem(tmp_path / "v", bad))
    bad = dict(base, nodes=["a", "a", "b", "c"])
    with pytest.raises(ValueError, match="unique"):
        validate_graph_data(_write_problem(tmp_path / "n", bad))
    bad = dict(base, nodes=["a", True, "b", "c"])
    with pytest.raises(ValueError, match="node labels"):
        validate_graph_data(_write_problem(tmp_path / "nb", bad))
    bad = dict(base, edges=[dict(base["edges"][0], v="ghost")])
    with pytest.raises(ValueError, match="undeclared node"):
        validate_graph_data(_write_problem(tmp_path / "e", bad))
    bad = dict(base, edges=[dict(base["edges"][0], u="a", v="a")])
    with pytest.raises(ValueError, match="self-loop"):
        validate_graph_data(_write_problem(tmp_path / "sl", bad))
    bad = dict(base, edges=[dict(base["edges"][0], weight=float("nan"))])
    with pytest.raises(ValueError, match="finite"):
        validate_graph_data(_write_problem(tmp_path / "ew", bad))
    bad = dict(base, edges=base["edges"] + [base["edges"][0]])
    with pytest.raises(ValueError, match="duplicates an existing edge"):
        validate_graph_data(_write_problem(tmp_path / "dup", bad))
    bad = dict(base, source="ghost")
    with pytest.raises(ValueError, match="declared nodes"):
        validate_graph_data(_write_problem(tmp_path / "src", bad))
    bad = dict(base, source="a", target="a")
    with pytest.raises(ValueError, match="differ from target"):
        validate_graph_data(_write_problem(tmp_path / "st", bad))
    mst = _mst_problem()
    bad = dict(mst, source="a", target="b")
    with pytest.raises(ValueError, match="must not declare"):
        validate_graph_data(_write_problem(tmp_path / "mst-extra", bad))
    no_source = {k: v for k, v in base.items() if k not in ("source", "target")}
    with pytest.raises(ValueError, match="requires 'source'"):
        validate_graph_data(_write_problem(tmp_path / "no-src", no_source))


def test_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir(parents=True)
    text = json.dumps(_shortest_path_problem()).replace(
        '"target": "d"',
        '"target": "d",\n    "target": "d"',
    )
    (public / "problem.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        validate_graph_data(public)


def test_validate_solution_shortest_path(tmp_path: Path) -> None:
    contract = validate_graph_data(
        _write_problem(tmp_path / "data", _shortest_path_problem())
    )
    path = validate_solution({"path": ["a", "b", "c", "d"]}, contract)
    assert path == ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="simple path"):
        validate_solution({"path": ["a", "b", "a", "d"]}, contract)
    with pytest.raises(ValueError, match="start at source"):
        validate_solution({"path": ["b", "c", "d"]}, contract)
    with pytest.raises(ValueError, match="non-edge"):
        validate_solution({"path": ["a", "d"]}, contract)
    with pytest.raises(ValueError, match="undeclared nodes"):
        validate_solution({"path": ["a", "z", "d"]}, contract)


def test_validate_solution_max_flow(tmp_path: Path) -> None:
    contract = validate_graph_data(
        _write_problem(tmp_path / "data", _max_flow_problem())
    )
    flow = {
        "s->a": 3.0,
        "s->b": 2.0,
        "a->b": 1.0,
        "a->t": 2.0,
        "b->t": 3.0,
    }
    result = validate_solution({"flow": flow}, contract)
    assert result[("s", "a")] == 3.0
    missing = dict(flow)
    missing.pop("s->a")
    with pytest.raises(ValueError, match="missing="):
        validate_solution({"flow": missing}, contract)
    extra = dict(flow, **{"a->s": 1.0})
    with pytest.raises(ValueError, match="undeclared edge"):
        validate_solution({"flow": extra}, contract)
    negative = dict(flow, **{"s->a": -1.0})
    with pytest.raises(ValueError, match="not be negative"):
        validate_solution({"flow": negative}, contract)
    bad_key = dict(flow)
    bad_key["s=>a"] = 1.0
    bad_key.pop("s->a")
    with pytest.raises(ValueError, match="missing=|flow keys"):
        validate_solution({"flow": bad_key}, contract)


def test_validate_solution_mst(tmp_path: Path) -> None:
    contract = validate_graph_data(
        _write_problem(tmp_path / "data", _mst_problem())
    )
    edges = [["a", "b"], ["b", "c"], ["c", "d"]]
    result = validate_solution({"edges": edges}, contract)
    assert result == [("a", "b"), ("b", "c"), ("c", "d")]
    with pytest.raises(ValueError, match="n-1"):
        validate_solution({"edges": edges[:-1]}, contract)
    with pytest.raises(ValueError, match="self-loop"):
        validate_solution({"edges": [["a", "b"], ["b", "c"], ["c", "c"]]}, contract)
    with pytest.raises(ValueError, match="duplicates"):
        validate_solution(
            {"edges": [["a", "b"], ["b", "a"], ["c", "d"]]}, contract
        )
    with pytest.raises(ValueError, match="not a declared edge"):
        validate_solution(
            {"edges": [["a", "b"], ["b", "c"], ["a", "c"]]}, contract
        )


def test_verifier_shortest_path_and_claims_ignored(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _shortest_path_problem())
    problem = build_graph_problem(public)
    payload = {
        "path": ["a", "b", "c", "d"],
        "total_weight": 0.0,
        "optimality": "optimal",
    }
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["total_weight"] == pytest.approx(4.0)  # 1+2+1, not claim 0
    assert values["path_violation"] == 0.0


def test_verifier_max_flow_violations(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _max_flow_problem())
    contract = validate_graph_data(public)
    context = GraphVerificationContext(contract)
    verifier = GraphVerifier()
    feasible = {
        "flow": {
            "s->a": 2.0,
            "s->b": 2.0,
            "a->b": 0.0,
            "a->t": 2.0,
            "b->t": 2.0,
        }
    }
    values = {
        o.name: o.value
        for o in verifier.verify(_artifact(feasible), context).observations
    }
    assert values["total_value"] == pytest.approx(4.0)
    assert values["capacity_violation"] == 0.0
    assert values["conservation_violation"] == 0.0
    violating = {
        "flow": {
            "s->a": 10.0,
            "s->b": 0.0,
            "a->b": 0.0,
            "a->t": 10.0,
            "b->t": 0.0,
        }
    }
    values = {
        o.name: o.value
        for o in verifier.verify(_artifact(violating), context).observations
    }
    assert values["capacity_violation"] == pytest.approx(7.0)
    assert values["conservation_violation"] == pytest.approx(0.0)


def test_verifier_mst_tree_violation(tmp_path: Path) -> None:
    public = _write_problem(tmp_path / "data", _mst_problem())
    contract = validate_graph_data(public)
    context = GraphVerificationContext(contract)
    verifier = GraphVerifier()
    good = {"edges": [["a", "b"], ["b", "c"], ["c", "d"]]}
    values = {
        o.name: o.value
        for o in verifier.verify(_artifact(good), context).observations
    }
    assert values["total_weight"] == pytest.approx(6.0)
    assert values["tree_violation"] == 0.0
    cyclic = {"edges": [["a", "b"], ["b", "d"], ["a", "d"]]}
    values = {
        o.name: o.value
        for o in verifier.verify(_artifact(cyclic), context).observations
    }
    assert values["tree_violation"] == 1.0


def test_context_invariant(tmp_path: Path) -> None:
    from dataclasses import replace

    contract = validate_graph_data(
        _write_problem(tmp_path / "data", _mst_problem())
    )
    with pytest.raises(ValueError, match="problem_type"):
        GraphVerificationContext(replace(contract, problem_type="bad"))
    with pytest.raises(ValueError, match="at least two nodes"):
        GraphVerificationContext(replace(contract, nodes=("a",)))
    with pytest.raises(ValueError, match="tolerance"):
        GraphVerificationContext(replace(contract, tolerance=0.0))
    context = GraphVerificationContext(contract)
    assert context.problem_type == "min_spanning_tree"
    assert len(context.fingerprint()) == 64
