"""Markov chain estimation data contract (R23).

Public files: ``problem.json`` (states and quantity to estimate) and
``train.csv`` (observed state sequence long format).  Host-only file:
``hidden_parameters.json`` (the true transition matrix, never public).

Quantities: transition_probability (from_state/to_state), steady_state
(state), expected_recurrence_time (state).  The hidden transition matrix
rows sum to 1 and the chain is irreducible (host validation).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ves_modeling.regression.data_contract import (
    _check_no_duplicate_headers,
    _id_key,
    _raw_headers,
)

QUANTITIES = (
    "transition_probability",
    "steady_state",
    "expected_recurrence_time",
)
MIN_ROWS = 50


@dataclass(frozen=True)
class MarkovDataContract:
    """Canonical public Markov problem (hidden matrix kept host-only)."""

    version: int
    quantity: str
    states: tuple[str, ...] = field(repr=False, compare=False)
    from_state: str | None = field(default=None, repr=False, compare=False)
    to_state: str | None = field(default=None, repr=False, compare=False)
    state: str | None = field(default=None, repr=False, compare=False)
    sequence_id_column: str | None = "sequence_id"
    n_rows: int = 0

    @property
    def n_states(self) -> int:
        return len(self.states)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "quantity": self.quantity,
            "states": list(self.states),
            "from_state": self.from_state,
            "to_state": self.to_state,
            "state": self.state,
            "sequence_id_column": self.sequence_id_column,
            "n_rows": self.n_rows,
        }


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


def _state_key(value: Any) -> str:
    """Canonical state key (1, 1.0 and '1' are the same)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, str)
    ):
        raise ValueError(
            "state keys must be a scalar string or finite number, "
            f"got {type(value).__name__}"
        )
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("state keys must not be empty")
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("state keys must be finite")
    return _id_key(value)


def validate_markov_data(
    public_dir: Path,
    *,
    state_column: str = "state",
    sequence_id_column: str | None = "sequence_id",
) -> MarkovDataContract:
    """Validate problem.json + train.csv; returns the public contract."""
    if not state_column.strip():
        raise ValueError("state_column must be non-empty")
    try:
        with (Path(public_dir) / "problem.json").open(
            encoding="utf-8"
        ) as handle:
            problem = json.load(
                handle, object_pairs_hook=_reject_duplicates
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in problem.json: {exc}") from None
    except OSError as exc:
        raise ValueError(f"cannot read problem.json: {exc}") from None
    if not isinstance(problem, dict):
        raise ValueError("problem.json root must be an object")
    unknown_top = set(problem) - {
        "version",
        "quantity",
        "states",
        "from_state",
        "to_state",
        "state",
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
    if "quantity" not in problem:
        raise ValueError("problem.json is missing 'quantity'")
    quantity = problem["quantity"]
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}")
    if "states" not in problem:
        raise ValueError("problem.json is missing 'states'")
    raw_states = problem["states"]
    if not isinstance(raw_states, list) or len(raw_states) < 2:
        raise ValueError("states must be a list with at least two entries")
    states = [_state_key(value) for value in raw_states]
    if len(set(states)) != len(states):
        raise ValueError("states must be unique")
    state_set = set(states)
    from_state: str | None = None
    to_state: str | None = None
    state: str | None = None
    if quantity == "transition_probability":
        if "from_state" not in problem or "to_state" not in problem:
            raise ValueError(
                "quantity 'transition_probability' requires from_state and "
                "to_state"
            )
        from_state = _state_key(problem["from_state"])
        to_state = _state_key(problem["to_state"])
        if from_state not in state_set or to_state not in state_set:
            raise ValueError("from_state/to_state must be declared states")
        if "state" in problem:
            raise ValueError(
                "quantity 'transition_probability' must not declare 'state'"
            )
    elif quantity == "steady_state":
        if "state" not in problem:
            raise ValueError("quantity 'steady_state' requires 'state'")
        state = _state_key(problem["state"])
        if state not in state_set:
            raise ValueError("state must be a declared state")
        if "from_state" in problem or "to_state" in problem:
            raise ValueError(
                "quantity 'steady_state' must not declare from/to_state"
            )
    else:
        if "state" not in problem:
            raise ValueError(
                "quantity 'expected_recurrence_time' requires 'state'"
            )
        state = _state_key(problem["state"])
        if state not in state_set:
            raise ValueError("state must be a declared state")
        if "from_state" in problem or "to_state" in problem:
            raise ValueError(
                "quantity 'expected_recurrence_time' must not declare "
                "from/to_state"
            )
    train_path = Path(public_dir) / "train.csv"
    _check_no_duplicate_headers(train_path, _raw_headers(train_path))
    train = pd.read_csv(train_path)
    if len(train) < MIN_ROWS:
        raise ValueError(f"train.csv needs at least {MIN_ROWS} rows")
    if state_column not in train.columns:
        raise ValueError(
            f"train.csv must contain state column {state_column!r}"
        )
    raw_states_col = train[state_column]
    if raw_states_col.isna().any():
        raise ValueError("train state column must not contain nulls")
    keys = [_state_key(value) for value in raw_states_col]
    if len(set(keys)) < 2:
        raise ValueError("train must contain at least two distinct states")
    if sequence_id_column is not None and sequence_id_column in train.columns:
        ids = train[sequence_id_column]
        if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
            raise ValueError("train sequence ids must not be empty")
    return MarkovDataContract(
        version=version,
        quantity=quantity,
        states=tuple(states),
        from_state=from_state,
        to_state=to_state,
        state=state,
        sequence_id_column=sequence_id_column,
        n_rows=len(train),
    )


def load_hidden_parameters(
    host_dir: Path, contract: MarkovDataContract
) -> np.ndarray:
    """Load the true transition matrix (host-only, never exposed)."""
    try:
        with (Path(host_dir) / "hidden_parameters.json").open(
            encoding="utf-8"
        ) as handle:
            params = json.load(
                handle, object_pairs_hook=_reject_duplicates
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in hidden_parameters.json: {exc}") from None
    except OSError as exc:
        raise ValueError(f"cannot read hidden_parameters.json: {exc}") from None
    if not isinstance(params, dict) or "transition_matrix" not in params:
        raise ValueError(
            "hidden_parameters.json must contain 'transition_matrix'"
        )
    raw = params["transition_matrix"]
    n = contract.n_states
    if not isinstance(raw, list) or len(raw) != n:
        raise ValueError("transition_matrix must be n x n")
    matrix: list[list[float]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError(f"transition_matrix row {i} must have {n} entries")
        parsed = [
            _finite_number(value, f"transition_matrix[{i}][{j}]")
            for j, value in enumerate(row)
        ]
        if any(value < 0.0 for value in parsed):
            raise ValueError("transition probabilities must be non-negative")
        if not math.isclose(sum(parsed), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("transition matrix rows must sum to 1")
        matrix.append(parsed)
    p = np.asarray(matrix, dtype=np.float64)
    if not _is_irreducible(p):
        raise ValueError("transition matrix must be irreducible")
    return p


def _is_irreducible(p: np.ndarray) -> bool:
    n = p.shape[0]
    adj = p > 0.0
    reachable = np.zeros((n, n), dtype=bool)
    for start in range(n):
        visited = np.zeros(n, dtype=bool)
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            for neighbor in range(n):
                if adj[node, neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        reachable[start] = visited
    return bool(reachable.all())


def compute_reference(
    contract: MarkovDataContract, transition_matrix: np.ndarray
) -> float:
    """Exact reference from the hidden matrix (host-only)."""
    p = transition_matrix
    if contract.quantity == "transition_probability":
        from_index = contract.states.index(contract.from_state)
        to_index = contract.states.index(contract.to_state)
        value = p[from_index, to_index]
    else:
        steady = _power_iteration(p)
        state_index = contract.states.index(contract.state)
        if contract.quantity == "steady_state":
            value = steady[state_index]
        else:
            if steady[state_index] <= 0.0:
                raise ValueError("steady-state probability must be positive")
            value = 1.0 / steady[state_index]
    reference = float(value)
    if not math.isfinite(reference):
        raise ValueError("analytic reference must be finite")
    return reference


def _power_iteration(p: np.ndarray, iterations: int = 10_000) -> np.ndarray:
    n = p.shape[0]
    distribution = np.full(n, 1.0 / n)
    for _ in range(iterations):
        new = distribution @ p
        if np.max(np.abs(new - distribution)) < 1e-15:
            distribution = new
            break
        distribution = new
    return distribution / distribution.sum()


def validate_solution(
    payload: dict[str, Any], contract: MarkovDataContract
) -> tuple[float, tuple[float, float] | None]:
    """Validate a solution artifact; returns (estimate, CI or None)."""
    if "estimate" not in payload:
        raise ValueError("missing required field 'estimate'")
    estimate = _finite_number(payload["estimate"], "estimate")
    confidence_interval: tuple[float, float] | None = None
    if "confidence_interval" in payload:
        raw_ci = payload["confidence_interval"]
        if (
            not isinstance(raw_ci, list)
            or len(raw_ci) != 2
            or isinstance(raw_ci[0], bool)
            or isinstance(raw_ci[1], bool)
        ):
            raise ValueError(
                "'confidence_interval' must be a [lo, hi] pair of numbers"
            )
        lo = _finite_number(raw_ci[0], "confidence_interval lo")
        hi = _finite_number(raw_ci[1], "confidence_interval hi")
        if lo > hi:
            raise ValueError("confidence_interval lo must not exceed hi")
        if not (lo <= estimate <= hi):
            raise ValueError(
                "estimate must lie within the confidence_interval"
            )
        confidence_interval = (lo, hi)
    return estimate, confidence_interval
