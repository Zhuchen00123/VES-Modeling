"""Stable Regression search API (Workflow-facing entry point).

idea.md (next phase) requires a concrete, stable Regression call entry so the
VES-MathModeling-Skill adapter never depends on examples/demos or on VES Core
SearchEngine internals.  The API keeps the vertical slice concrete: no
universal task abstractions until a second domain proves them necessary.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ves.search import GreedyTop1Policy
from ves.search_engine import SearchEngine

from ves_modeling.llm import OpenAICompatibleClient
from ves_modeling.regression.generator import (
    LlmClient,
    LLMRegressionGenerator,
    MockRegressionGenerator,
)
from ves_modeling.regression.problem import build_regression_problem
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
    LocalRegressionRunner,
)

GeneratorKind = Literal["mock", "llm"]

_REPO_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "candidates"


def _observation(evidence: Any, name: str) -> float | None:
    for item in evidence or ():
        if item.name == name:
            return item.value
    return None


@dataclass(frozen=True)
class RegressionSearchResult:
    """Host-verified regression search outcome.

    ``best_rmse`` / ``best_mae`` come from the host verifier evidence, never
    from candidate self-reports.  ``records`` preserves the full search
    transcript for Workflow-side reporting.
    """

    run_id: str
    dataset_name: str
    generator: str
    status: str
    drafts: int
    improves: int
    best_code: str | None
    best_candidate_id: str | None
    best_evidence: Any | None
    best_rmse: float | None
    best_mae: float | None
    rejected: int
    run_dir: Path
    records: tuple[Any, ...]


def _default_fixtures() -> Path:
    if _REPO_FIXTURES.is_dir():
        return _REPO_FIXTURES
    return Path.cwd() / "fixtures" / "candidates"


def run_regression_search(
    public_dir: Path,
    host_dir: Path,
    *,
    drafts: int = 2,
    improves: int = 3,
    workspace: Path | None = None,
    generator: GeneratorKind = "mock",
    dataset_name: str = "regression",
    labels: np.ndarray | None = None,
    fixture_dir: Path | None = None,
    fallback_code: str | None = None,
    client: LlmClient | None = None,
    image: str = "ves-modeling-runner:0.1",
    timeout_seconds: float = 900.0,
) -> RegressionSearchResult:
    """Search a tabular regression task and return the host-verified best.

    ``public_dir`` must contain ``train.csv`` / ``test_features.csv``
    (candidate-visible); ``host_dir`` must contain ``hidden_test_labels.csv``
    (host-only, never exposed to candidates).  ``labels`` may be injected
    instead for tests.

    ``generator="mock"`` uses trusted fixtures + the local subprocess runner
    (tests/fixtures only); ``generator="llm"`` uses the LLM generator plus the
    Docker runner (the only execution boundary for real candidates).

    Returns a :class:`RegressionSearchResult`; artifacts (best_solution.py,
    summary.json, config.json) are persisted under ``workspace/<run_id>``.
    """
    public_dir = Path(public_dir)
    host_dir = Path(host_dir)
    workspace = Path(workspace or Path.cwd() / "runs")
    workspace.mkdir(parents=True, exist_ok=True)
    fixture_dir = Path(fixture_dir) if fixture_dir else _default_fixtures()

    problem = build_regression_problem(
        public_dir, host_dir, dataset_name=dataset_name, labels=labels
    )

    if generator == "mock":
        candidate = MockRegressionGenerator(fixture_dir)
        runner = LocalRegressionRunner(workspace=workspace, data_dir=public_dir)
    elif generator == "llm":
        if fallback_code is None:
            fallback_path = fixture_dir / "linear_regression.py"
            if not fallback_path.is_file():
                raise FileNotFoundError(f"fallback fixture missing: {fallback_path}")
            fallback_code = fallback_path.read_text(encoding="utf-8")
        candidate = LLMRegressionGenerator(
            client or OpenAICompatibleClient(), fallback_code=fallback_code
        )
        runner = DockerRegressionRunner(
            DockerRunnerConfig(
                workspace=workspace,
                data_dir=public_dir,
                image=image,
                timeout_seconds=timeout_seconds,
            )
        )
    else:
        raise ValueError(f"unknown generator: {generator!r}")

    engine = SearchEngine(
        problem=problem,
        generator=candidate,
        runner=runner,
        anchor_policy=GreedyTop1Policy(),
        drafts=drafts,
        improves=improves,
    )
    result = engine.search()

    status = "verified" if result.best_code is not None else "no_verified"
    run_id = uuid.uuid4().hex[:12]
    run_dir = workspace / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_evidence = result.best_evidence
    best_rmse = (
        _observation(best_evidence, "rmse") if best_evidence else None
    )
    best_mae = (
        _observation(best_evidence, "mae") if best_evidence else None
    )
    best_candidate_id = result.best_record.candidate_id if result.best_record else None

    (run_dir / "best_solution.py").write_text(
        result.best_code or "", encoding="utf-8"
    )
    summary = {
        "run_id": run_id,
        "task": "regression",
        "dataset": dataset_name,
        "generator": generator,
        "status": status,
        "drafts": drafts,
        "improves": improves,
        "best_candidate_id": best_candidate_id,
        "best_rmse": best_rmse,
        "best_mae": best_mae,
        "rejected": result.rejected,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "generator": generator,
                "drafts": drafts,
                "improves": improves,
                "image": image if generator == "llm" else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return RegressionSearchResult(
        run_id=run_id,
        dataset_name=dataset_name,
        generator=generator,
        status=status,
        drafts=drafts,
        improves=improves,
        best_code=result.best_code,
        best_candidate_id=best_candidate_id,
        best_evidence=best_evidence,
        best_rmse=best_rmse,
        best_mae=best_mae,
        rejected=result.rejected,
        run_dir=run_dir,
        records=tuple(result.records),
    )
