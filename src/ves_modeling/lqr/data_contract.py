"""Finite-horizon discrete LQR data contract (R26).

``problem.json`` is the complete public instance (no hidden truth): version,
A (n x n finite), B (n x m finite), Q (n x n symmetric PSD), R (m x m
symmetric PD), x0 (n finite), horizon N (int >= 2) and optional
``terminal_weight`` Q_N (n x n symmetric PSD, default Q).

Artifact ``solution.json``: ``{"control": [u_0, ..., u_{N-1}]}`` with exactly
N control vectors of m finite numbers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LqrDataContract:
    """Canonical public LQR instance (never host secrets)."""

    A: np.ndarray = field(repr=False)
    B: np.ndarray = field(repr=False)
    Q: np.ndarray = field(repr=False)
    R: np.ndarray = field(repr=False)
    QN: np.ndarray = field(repr=False)
    x0: np.ndarray = field(repr=False)
    horizon: int
    n: int
    m: int
    terminal_weight: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": {"state": self.n, "control": self.m},
            "horizon": self.horizon,
            "terminal_weight": self.terminal_weight,
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


def _finite_matrix(value: Any, what: str) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{what} must be a non-empty matrix (list of lists)")
    rows: list[list[float]] = []
    width: int | None = None
    for row_index, row in enumerate(value):
        if not isinstance(row, list):
            raise ValueError(f"{what} rows must be lists")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError(f"{what} must be a rectangular matrix")
        numbers: list[float] = []
        for column, entry in enumerate(row):
            numbers.append(
                _finite_number(entry, f"{what}[{row_index}][{column}]")
            )
        rows.append(numbers)
    return np.asarray(rows, dtype=np.float64)


def _finite_vector(value: Any, what: str) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{what} must be a non-empty list")
    numbers: list[float] = []
    for index, entry in enumerate(value):
        numbers.append(_finite_number(entry, f"{what}[{index}]"))
    return np.asarray(numbers, dtype=np.float64)


def _require_square(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")


def _require_symmetric(matrix: np.ndarray, name: str) -> None:
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric")


def _require_psd(matrix: np.ndarray, name: str) -> None:
    eigenvalues = np.linalg.eigvalsh(matrix)
    tol = 1e-8 * max(1.0, float(np.max(np.abs(matrix))))
    if float(eigenvalues.min()) < -tol:
        raise ValueError(f"{name} must be positive semidefinite")


def _require_pd(matrix: np.ndarray, name: str) -> None:
    eigenvalues = np.linalg.eigvalsh(matrix)
    tol = 1e-8 * max(1.0, float(np.max(np.abs(matrix))))
    if float(eigenvalues.min()) <= tol:
        raise ValueError(f"{name} must be positive definite")


def validate_lqr_data(public_dir: Path) -> LqrDataContract:
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
        "A",
        "B",
        "Q",
        "R",
        "x0",
        "horizon",
        "terminal_weight",
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
    for key in ("A", "B", "Q", "R", "x0", "horizon"):
        if key not in problem:
            raise ValueError(f"problem.json is missing {key!r}")

    A = _finite_matrix(problem["A"], "A")
    _require_square(A, "A")
    n = int(A.shape[0])
    B = _finite_matrix(problem["B"], "B")
    if B.ndim != 2 or B.shape[0] != n or B.shape[1] < 1:
        raise ValueError("B must be an n x m matrix with m >= 1")
    m = int(B.shape[1])
    Q = _finite_matrix(problem["Q"], "Q")
    _require_square(Q, "Q")
    _require_symmetric(Q, "Q")
    _require_psd(Q, "Q")
    if Q.shape[0] != n:
        raise ValueError("Q must be an n x n matrix")
    R = _finite_matrix(problem["R"], "R")
    _require_square(R, "R")
    _require_symmetric(R, "R")
    _require_pd(R, "R")
    if R.shape[0] != m:
        raise ValueError("R must be an m x m matrix")
    x0 = _finite_vector(problem["x0"], "x0")
    if x0.size != n:
        raise ValueError("x0 must have exactly n entries")
    horizon = problem["horizon"]
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, int):
        raise ValueError("'horizon' must be an integer")
    if horizon < 2:
        raise ValueError("'horizon' must be >= 2")

    QN = Q
    terminal_weight = "default Q"
    if "terminal_weight" in problem:
        QN = _finite_matrix(problem["terminal_weight"], "terminal_weight")
        _require_square(QN, "terminal_weight")
        _require_symmetric(QN, "terminal_weight")
        _require_psd(QN, "terminal_weight")
        if QN.shape[0] != n:
            raise ValueError("terminal_weight must be an n x n matrix")
        terminal_weight = "provided"

    return LqrDataContract(
        A=A,
        B=B,
        Q=Q,
        R=R,
        QN=QN,
        x0=x0,
        horizon=int(horizon),
        n=n,
        m=m,
        terminal_weight=terminal_weight,
    )


def validate_control(
    payload: dict[str, Any], contract: LqrDataContract
) -> np.ndarray:
    """Validate a control artifact; returns an (N, m) array."""
    if "control" not in payload:
        raise ValueError("missing required field 'control'")
    raw = payload["control"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError("'control' must be a JSON array")
    if len(raw) != contract.horizon:
        raise ValueError(
            f"'control' must have exactly {contract.horizon} entries"
        )
    rows: list[list[float]] = []
    for step, vector in enumerate(raw):
        if not isinstance(vector, list):
            raise ValueError(f"control[{step}] must be a vector")
        if len(vector) != contract.m:
            raise ValueError(
                f"control[{step}] must have exactly {contract.m} entries"
            )
        numbers: list[float] = []
        for index, entry in enumerate(vector):
            numbers.append(
                _finite_number(entry, f"control[{step}][{index}]")
            )
        rows.append(numbers)
    return np.asarray(rows, dtype=np.float64)
