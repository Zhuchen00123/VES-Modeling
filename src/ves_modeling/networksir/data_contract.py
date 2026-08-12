"""Network epidemic (graph SIR) data contract (R30).

``problem.json`` is the complete public instance (no hidden truth):
``version``, ``model: "network_sir"``, ``beta`` (> 0), ``gamma`` (> 0),
``n_nodes`` (int in [10, 100]), ``edges`` (undirected, no self-loops, no
duplicates), ``i0`` (int >= 1 initial infected nodes), ``t_end`` (> 0) and
``quantity``: ``final_size`` | ``peak_infected`` | ``infected_at`` (with
``t`` in (0, t_end]).

Artifact ``solution.json``: ``{"estimate": finite_number,
"confidence_interval": [lo, hi]}`` — CI optional, lo <= estimate <= hi.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

QUANTITIES = ("final_size", "peak_infected", "infected_at")


@dataclass(frozen=True)
class NetworkSirDataContract:
    """Canonical public network-SIR instance."""

    beta: float
    gamma: float
    n_nodes: int
    adjacency: tuple[frozenset[int], ...] = field(repr=False)
    i0: int
    t_end: float
    quantity: str
    t: float | None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": "network_sir",
            "beta": self.beta,
            "gamma": self.gamma,
            "n_nodes": self.n_nodes,
            "edge_count": sum(len(neighbors) for neighbors in self.adjacency)
            // 2,
            "i0": self.i0,
            "t_end": self.t_end,
            "quantity": self.quantity,
        }
        if self.t is not None:
            result["t"] = self.t
        return result


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in JSON object: {key!r}")
        result[key] = value
    return result


def _finite_number(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float)
    ):
        raise ValueError(f"{what} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite (no NaN/Infinity)")
    return number


def _positive_number(value: Any, what: str) -> float:
    number = _finite_number(value, what)
    if number <= 0.0:
        raise ValueError(f"{what} must be positive")
    return number


def _strict_int(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer")
    return int(value)


def validate_networksir_data(public_dir: Path) -> NetworkSirDataContract:
    """Validate the public problem.json and return the canonical contract."""
    try:
        with (Path(public_dir) / "problem.json").open(
            encoding="utf-8"
        ) as handle:
            problem = json.load(handle, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in problem.json: {exc}") from None
    except OSError as exc:
        raise ValueError(f"cannot read problem.json: {exc}") from None
    if not isinstance(problem, dict):
        raise ValueError("problem.json root must be an object")
    unknown_top = set(problem) - {
        "version",
        "model",
        "beta",
        "gamma",
        "n_nodes",
        "edges",
        "i0",
        "t_end",
        "quantity",
        "t",
    }
    if unknown_top:
        raise ValueError(
            f"problem.json has unknown top-level fields: {sorted(unknown_top)}"
        )
    if "version" not in problem:
        raise ValueError("problem.json is missing 'version'")
    version = _strict_int(problem["version"], "problem.json 'version'")
    if version < 1:
        raise ValueError("problem.json 'version' must be >= 1")
    if "model" not in problem:
        raise ValueError("problem.json is missing 'model'")
    if problem["model"] != "network_sir":
        raise ValueError("problem.json 'model' must be 'network_sir'")
    for key in ("beta", "gamma", "n_nodes", "edges", "i0", "t_end", "quantity"):
        if key not in problem:
            raise ValueError(f"problem.json is missing {key!r}")
    beta = _positive_number(problem["beta"], "beta")
    gamma = _positive_number(problem["gamma"], "gamma")
    n_nodes = _strict_int(problem["n_nodes"], "n_nodes")
    if not 10 <= n_nodes <= 100:
        raise ValueError("n_nodes must be an integer in [10, 100]")
    edges_raw = problem["edges"]
    if isinstance(edges_raw, (str, bytes)) or not isinstance(edges_raw, list):
        raise ValueError("'edges' must be a JSON array")
    seen: set[tuple[int, int]] = set()
    adjacency: list[frozenset[int]] = [
        frozenset() for _ in range(n_nodes)
    ]
    for edge_index, edge in enumerate(edges_raw):
        if isinstance(edge, (str, bytes)) or not isinstance(edge, list):
            raise ValueError(f"edges[{edge_index}] must be a pair")
        if len(edge) != 2:
            raise ValueError(f"edges[{edge_index}] must have exactly 2 entries")
        u = _strict_int(edge[0], f"edges[{edge_index}][0]")
        v = _strict_int(edge[1], f"edges[{edge_index}][1]")
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            raise ValueError(
                f"edges[{edge_index}] endpoints must lie in [0, n_nodes)"
            )
        if u == v:
            raise ValueError(f"edges[{edge_index}] must not be a self-loop")
        key = (min(u, v), max(u, v))
        if key in seen:
            raise ValueError(
                f"edges[{edge_index}] duplicates an earlier edge"
            )
        seen.add(key)
        adjacency[u] = adjacency[u] | {v}
        adjacency[v] = adjacency[v] | {u}
    i0 = _strict_int(problem["i0"], "i0")
    if i0 < 1:
        raise ValueError("i0 must be an integer >= 1")
    if i0 > n_nodes:
        raise ValueError("i0 must not exceed n_nodes")
    t_end = _positive_number(problem["t_end"], "t_end")
    quantity = problem["quantity"]
    if not isinstance(quantity, str) or quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}")
    t: float | None = None
    if quantity == "infected_at":
        if "t" not in problem:
            raise ValueError("quantity 'infected_at' requires 't'")
        t = _finite_number(problem["t"], "t")
        if t <= 0.0 or t > t_end:
            raise ValueError("t must lie in (0, t_end]")
    elif "t" in problem:
        raise ValueError("'t' is only valid for quantity 'infected_at'")
    return NetworkSirDataContract(
        beta=beta,
        gamma=gamma,
        n_nodes=n_nodes,
        adjacency=tuple(adjacency),
        i0=i0,
        t_end=t_end,
        quantity=quantity,
        t=t,
    )


def _finite_number_estimate(value: Any, what: str) -> float:
    return _finite_number(value, what)


def validate_estimate(
    payload: dict[str, Any],
) -> tuple[float, tuple[float, float] | None]:
    """Validate a solution artifact; returns (estimate, CI or None)."""
    if "estimate" not in payload:
        raise ValueError("missing required field 'estimate'")
    estimate = _finite_number_estimate(payload["estimate"], "estimate")
    confidence_interval: tuple[float, float] | None = None
    if "confidence_interval" in payload:
        raw = payload["confidence_interval"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
            raise ValueError("'confidence_interval' must be a JSON array")
        if len(raw) != 2:
            raise ValueError(
                "'confidence_interval' must have exactly 2 entries"
            )
        lo = _finite_number(raw[0], "confidence_interval[0]")
        hi = _finite_number(raw[1], "confidence_interval[1]")
        if lo > hi:
            raise ValueError("confidence_interval lo must not exceed hi")
        if estimate < lo or estimate > hi:
            raise ValueError("estimate must lie within confidence_interval")
        confidence_interval = (lo, hi)
    return estimate, confidence_interval
