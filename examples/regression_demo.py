"""VES Modeling — Regression demo (mock or real LLM + Docker).

Usage:
  python examples/regression_demo.py --mock
  python examples/regression_demo.py --llm --drafts 2 --improves 3
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

from ves.search import GreedyTop1Policy

from ves_modeling.llm import OpenAICompatibleClient
from ves_modeling.regression.generator import (
    LLMRegressionGenerator,
    MockRegressionGenerator,
)
from ves_modeling.regression.problem import build_regression_problem
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
    LocalRegressionRunner,
)

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
    problem = build_regression_problem(public_dir, host_dir)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]

    if args.mock:
        generator = MockRegressionGenerator(root / "fixtures" / "candidates")
        runner = LocalRegressionRunner(workspace=runs_dir, data_dir=public_dir)
    else:
        fallback = (root / "fixtures" / "candidates" / "linear_regression.py").read_text(
            encoding="utf-8"
        )
        generator = LLMRegressionGenerator(
            OpenAICompatibleClient(), fallback_code=fallback
        )
        runner = DockerRegressionRunner(
            DockerRunnerConfig(
                workspace=runs_dir, data_dir=public_dir, image=args.image
            )
        )

    from ves.search_engine import SearchEngine

    engine = SearchEngine(
        problem=problem,
        generator=generator,
        runner=runner,
        anchor_policy=GreedyTop1Policy(),
        drafts=args.drafts,
        improves=args.improves,
    )
    result = engine.search()

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
        print(f"candidate: {result.best_record.candidate_id[:8] if result.best_record else '?'}")
        print(f"rmse: {observation(result.best_evidence, 'rmse'):.3f}")
        print(f"mae: {observation(result.best_evidence, 'mae'):.3f}")
        print(f"rejected: {result.rejected}")

    # Save run artifacts (never API keys or hidden labels).
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_solution.py").write_text(result.best_code or "", encoding="utf-8")
    summary = {
        "run_id": run_id,
        "task": "regression",
        "dataset": "regression",
        "drafts": args.drafts,
        "improves": args.improves,
        "best_candidate_id": result.best_record.candidate_id if result.best_record else None,
        "best_rmse": observation(result.best_evidence, "rmse") if result.best_evidence else None,
        "best_mae": observation(result.best_evidence, "mae") if result.best_evidence else None,
        "rejected": result.rejected,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "mock": args.mock,
                "drafts": args.drafts,
                "improves": args.improves,
                "image": args.image,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
