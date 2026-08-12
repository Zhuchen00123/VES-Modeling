"""Graph/network data contract validation (R14).

``problem.json`` is the complete public instance (no hidden truth):
version, problem_type (shortest_path | max_flow | min_spanning_tree),
unique nodes, finite-weight edges, and type-specific source/target.  The
candidate artifact ``solution.json`` is validated per problem type.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

PROBLEM_TYPES = ("shortest_path", "max_flow", "min_spanning_tree")
UNDIRECTED_TYPES = ("shortest_path", "min_spanning_tree")


@dataclass(frozen=True)
class GraphEdge:
    u: str
    v: str
    weight: float


@dataclass(frozen=True)
class GraphDataContract:
    """Canonical public graph problem (never hidden values)."""

    version: int
    problem_type: str
    nodes: tuple[str, ...] = field(repr=False, compare=False)
    edges: tuple[GraphEdge, ...] = field(repr=False, compare=False)
    source: str | None = field(default=None, repr=False, compare=False)
    target: str | None = field(default=None, repr=False, compare=False)
    tolerance: float = 1e-6

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def edge_weights(self) -> dict[tuple[str, str], float]:
        result: dict[tuple[str, str], float] = {}
        for edge in self.edges:
            result[(edge.u, edge.v)] = edge.weight
            if self.problem_type in UNDIRECTED_TYPES:
                result[(edge.v, edge.u)] = edge.weight
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "problem_type": self.problem_type,
            "nodes": list(self.nodes),
            "edges": [
                {"u": edge.u, "v": edge.v, "weight": edge.weight}
                for edge in self.edges
            ],
            "source": self.source,
            "target": self.target,
            "tolerance": self.tolerance,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
        }


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in JSON object: {key!r}")
        result[key] = value
    return result


def load_problem(path: Path) -> dict[str, Any]:
    """Load problem.json with duplicate-key rejection."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from None
    except OSError as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} root must be an object")
    return data


def _node_key(value: Any) -> str:
    """Canonical node key (non-empty str or finite number, bool rejected)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("node labels must not be booleans")
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("node labels must be finite")
        return str(int(number)) if number.is_integer() else str(number)
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("node labels must not be empty")
        return value
    raise ValueError(
        "node labels must be non-empty strings or finite numbers, "
        f"got {type(value).__name__}"
    )


def _finite_number(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float)
    ):
        raise ValueError(f"{what} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite (no NaN/Infinity)")
    return number


def _edge_key(u: str, v: str, problem_type: str) -> tuple[str, ...]:
    if problem_type in UNDIRECTED_TYPES:
        return tuple(sorted((u, v)))
    return (u, v)


def validate_graph_data(
    public_dir: Path,
    *,
    tolerance: float = 1e-6,
) -> GraphDataContract:
    """Validate the public problem.json and return the canonical contract."""
    tolerance_value = _finite_number(tolerance, "tolerance")
    if tolerance_value <= 0.0:
        raise ValueError("tolerance must be positive")
    problem = load_problem(Path(public_dir) / "problem.json")
    unknown_top = set(problem) - {
        "version",
        "problem_type",
        "nodes",
        "edges",
        "source",
        "target",
    }
    if unknown_top:
        raise ValueError(
            f"problem.json has unknown top-level fields: {sorted(unknown_top)}"
        )
    if "version" not in problem:
        raise ValueError("problem.json is missing 'version'")
    version = problem["version"]
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, int):
        raise ValueError("problem.json 'version' must be an integer")
    if version < 1:
        raise ValueError("problem.json 'version' must be >= 1")
    if "problem_type" not in problem:
        raise ValueError("problem.json is missing 'problem_type'")
    problem_type = problem["problem_type"]
    if problem_type not in PROBLEM_TYPES:
        raise ValueError(
            f"problem.json 'problem_type' must be one of {PROBLEM_TYPES}"
        )
    if "nodes" not in problem:
        raise ValueError("problem.json is missing 'nodes'")
    raw_nodes = problem["nodes"]
    if not isinstance(raw_nodes, list):
        raise ValueError("'nodes' must be a JSON array")
    nodes = [_node_key(value) for value in raw_nodes]
    if len(nodes) < 2:
        raise ValueError("at least two nodes are required")
    if len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be unique")
    node_set = set(nodes)
    if "edges" not in problem:
        raise ValueError("problem.json is missing 'edges'")
    raw_edges = problem["edges"]
    if not isinstance(raw_edges, list):
        raise ValueError("'edges' must be a JSON array")
    edges: list[GraphEdge] = []
    edge_keys: set[tuple[str, ...]] = set()
    for index, entry in enumerate(raw_edges):
        if not isinstance(entry, dict):
            raise ValueError(f"edge {index} must be an object")
        unknown = set(entry) - {"u", "v", "weight"}
        if unknown:
            raise ValueError(
                f"edge {index} has unknown fields: {sorted(unknown)}"
            )
        if "u" not in entry or "v" not in entry or "weight" not in entry:
            raise ValueError(
                f"edge {index} must contain 'u', 'v' and 'weight'"
            )
        u = _node_key(entry["u"])
        v = _node_key(entry["v"])
        if u not in node_set or v not in node_set:
            raise ValueError(
                f"edge {index} references undeclared node"
            )
        if u == v:
            raise ValueError(f"edge {index} must not be a self-loop")
        weight = _finite_number(entry["weight"], f"edge {index} weight")
        key = _edge_key(u, v, problem_type)
        if key in edge_keys:
            raise ValueError(f"edge {index} duplicates an existing edge")
        edge_keys.add(key)
        edges.append(GraphEdge(u=u, v=v, weight=weight))
    source: str | None = None
    target: str | None = None
    if problem_type in ("shortest_path", "max_flow"):
        if "source" not in problem or "target" not in problem:
            raise ValueError(
                f"problem_type {problem_type!r} requires 'source' and "
                "'target'"
            )
        source = _node_key(problem["source"])
        target = _node_key(problem["target"])
        if source not in node_set or target not in node_set:
            raise ValueError("source/target must be declared nodes")
        if source == target:
            raise ValueError("source must differ from target")
    else:
        if "source" in problem or "target" in problem:
            raise ValueError(
                "min_spanning_tree must not declare source/target"
            )
    return GraphDataContract(
        version=version,
        problem_type=problem_type,
        nodes=tuple(nodes),
        edges=tuple(edges),
        source=source,
        target=target,
        tolerance=tolerance_value,
    )


def _validate_shortest_path(
    payload: dict[str, Any], contract: GraphDataContract
) -> list[str]:
    if "path" not in payload:
        raise ValueError("missing required field 'path'")
    raw_path = payload["path"]
    if isinstance(raw_path, (str, bytes)) or not isinstance(raw_path, list):
        raise ValueError("'path' must be a JSON array of nodes")
    path = [_node_key(value) for value in raw_path]
    if not path:
        raise ValueError("'path' must not be empty")
    node_set = set(contract.nodes)
    if any(node not in node_set for node in path):
        raise ValueError("'path' references undeclared nodes")
    if len(set(path)) != len(path):
        raise ValueError("'path' must be a simple path (no repeated nodes)")
    if path[0] != contract.source or path[-1] != contract.target:
        raise ValueError(
            "'path' must start at source and end at target"
        )
    edge_pairs = contract.edge_weights()
    for u, v in pairwise(path):
        if (u, v) not in edge_pairs:
            raise ValueError(f"'path' uses non-edge ({u!r} -> {v!r})")
    return path


def _validate_max_flow(
    payload: dict[str, Any], contract: GraphDataContract
) -> dict[tuple[str, str], float]:
    if "flow" not in payload:
        raise ValueError("missing required field 'flow'")
    raw_flow = payload["flow"]
    if isinstance(raw_flow, (str, bytes)) or not isinstance(raw_flow, dict):
        raise ValueError("'flow' must be a JSON object keyed by edge")
    declared_keys = {
        (edge.u, edge.v): edge.weight for edge in contract.edges
    }
    flow: dict[tuple[str, str], float] = {}
    parsed_keys: set[tuple[str, str]] = set()
    for edge_key, value in raw_flow.items():
        if not isinstance(edge_key, str) or "->" not in edge_key:
            raise ValueError(
                "flow keys must be '<u>-><v>' strings matching declared "
                "edges"
            )
        u, v = edge_key.split("->", 1)
        pair = (u, v)
        parsed_keys.add(pair)
        if pair not in declared_keys:
            raise ValueError(f"flow references undeclared edge {edge_key!r}")
        flow_value = _finite_number(value, f"flow {edge_key!r}")
        if flow_value < 0.0:
            raise ValueError(f"flow {edge_key!r} must not be negative")
        flow[pair] = flow_value
    missing = sorted(set(declared_keys) - parsed_keys)
    if missing:
        raise ValueError(
            f"flow must cover declared edges exactly (missing={missing})"
        )
    return flow


def _validate_mst(
    payload: dict[str, Any], contract: GraphDataContract
) -> list[tuple[str, str]]:
    if "edges" not in payload:
        raise ValueError("missing required field 'edges'")
    raw_edges = payload["edges"]
    if isinstance(raw_edges, (str, bytes)) or not isinstance(raw_edges, list):
        raise ValueError("'edges' must be a JSON array of [u, v] pairs")
    if len(raw_edges) != contract.n_nodes - 1:
        raise ValueError(
            f"spanning tree must have exactly n-1 = {contract.n_nodes - 1} "
            f"edges, got {len(raw_edges)}"
        )
    node_set = set(contract.nodes)
    edge_pairs = contract.edge_weights()
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for index, pair in enumerate(raw_edges):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or isinstance(pair[0], bool)
            or isinstance(pair[1], bool)
        ):
            raise ValueError(
                f"tree edge {index} must be a [u, v] pair of nodes"
            )
        u = _node_key(pair[0])
        v = _node_key(pair[1])
        if u not in node_set or v not in node_set:
            raise ValueError(
                f"tree edge {index} references undeclared nodes"
            )
        if u == v:
            raise ValueError(f"tree edge {index} must not be a self-loop")
        key = tuple(sorted((u, v)))
        if key in seen:
            raise ValueError(f"tree edge {index} duplicates an edge")
        seen.add(key)
        if (u, v) not in edge_pairs:
            raise ValueError(
                f"tree edge {index} is not a declared edge ({u!r}, {v!r})"
            )
        selected.append((u, v))
    return selected


def validate_solution(
    payload: dict[str, Any], contract: GraphDataContract
) -> Any:
    """Validate a graph solution artifact for the problem type."""
    if contract.problem_type == "shortest_path":
        return _validate_shortest_path(payload, contract)
    if contract.problem_type == "max_flow":
        return _validate_max_flow(payload, contract)
    return _validate_mst(payload, contract)
