"""VES Modeling — Optimization demo (mock generator + trusted local fixture).

Usage:
  python examples/optimization_demo.py --drafts 2 --improves 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ves_modeling.optimization import run_optimization_search

logging.basicConfig(level=logging.WARNING)

LP_PROBLEM = {
    "version": 1,
    "sense": "minimize",
    "variables": {
        "x0": {"type": "continuous", "lower": 0.0, "upper": 10.0},
        "x1": {"type": "continuous", "lower": 0.0, "upper": 10.0},
    },
    "objective": {"coefficients": {"x0": 1.0, "x1": 2.0}, "constant": 3.0},
    "constraints": [
        {"coefficients": {"x0": 1.0, "x1": 1.0}, "sense": "<=", "rhs": 5.0},
        {"coefficients": {"x0": 2.0, "x1": -1.0}, "sense": ">=", "rhs": 0.0},
        {"coefficients": {"x1": 1.0}, "sense": "==", "rhs": 1.0},
    ],
}


def ensure_problem(root: Path) -> Path:
    public_dir = root / "data" / "optimization" / "public"
    if (public_dir / "problem.json").is_file():
        return public_dir
    public_dir.mkdir(parents=True)
    (public_dir / "problem.json").write_text(
        json.dumps(LP_PROBLEM), encoding="utf-8"
    )
    return public_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--drafts", type=int, default=2)
    parser.add_argument("--improves", type=int, default=3)
    args = parser.parse_args()

    root = args.root.resolve()
    public_dir = ensure_problem(root)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    result = run_optimization_search(
        public_dir,
        drafts=args.drafts,
        improves=args.improves,
        workspace=runs_dir,
        generator="mock",
    )
    print("status:", result.status)
    print("best_feasible:", result.best_feasible)
    print("best_objective:", result.best_objective)
    print("best_max_constraint_violation:", result.best_max_constraint_violation)
    print("run_dir:", result.run_dir)


if __name__ == "__main__":
    main()
