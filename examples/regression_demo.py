"""VES Modeling — Regression demo (mock or real LLM + Docker).

Usage:
  python examples/regression_demo.py --mock
  python examples/regression_demo.py --llm --drafts 2 --improves 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ves_modeling.regression import run_regression_search

logging.basicConfig(level=logging.WARNING)


def ensure_data(root: Path) -> tuple[Path, Path]:
    public_dir = root / "data" / "regression" / "public"
    host_dir = root / "data" / "regression" / "host"
    if not (public_dir / "train.csv").is_file():
        import sys

        sys.path.insert(0, str(root / "scripts"))
        from generate_regression_data import generate

        generate(public_dir, host_dir)
    return public_dir, host_dir


def observation(evidence, name: str) -> float:
    for item in evidence:
        if item.name == name:
            return item.value
    raise ValueError(f"missing {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="mock generator + local runner")
    group.add_argument("--llm", action="store_true", help="LLM generator + Docker runner")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--drafts", type=int, default=2)
    parser.add_argument("--improves", type=int, default=3)
    parser.add_argument("--image", default="ves-modeling-runner:0.1")
    args = parser.parse_args()

    root = args.root.resolve()
    public_dir, host_dir = ensure_data(root)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    result = run_regression_search(
        public_dir,
        host_dir,
        drafts=args.drafts,
        improves=args.improves,
        workspace=runs_dir,
        generator="mock" if args.mock else "llm",
        fixture_dir=root / "fixtures" / "candidates",
        image=args.image,
    )

    print("VES Modeling")
    print("task: regression")
    print("metric: RMSE")
    print(f"search: {args.drafts} drafts + {args.improves} improves")
    for index, record in enumerate(result.records):
        rmse = observation(record.evidence, "rmse")
        mae = observation(record.evidence, "mae")
        label = f"draft{index}" if index < args.drafts else f"improve{index - args.drafts}"
        print(f"[{label}] verification: VERIFIED rmse: {rmse:.3f} mae: {mae:.3f}")
    print()
    if result.best_code is None:
        print("BEST VERIFIED: none (all candidates rejected)")
    else:
        print("BEST VERIFIED")
        print(f"candidate: {result.best_candidate_id[:8] if result.best_candidate_id else '?'}")
        print(f"rmse: {result.best_rmse:.3f}")
        print(f"mae: {result.best_mae:.3f}")
        print(f"rejected: {result.rejected}")
    print(f"run artifacts: {result.run_dir}")


if __name__ == "__main__":
    main()
