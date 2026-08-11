"""R6 pre-check: full closed loop with a scripted LLM + real Docker sandbox.

The loop is real (SearchEngine -> LLMRegressionGenerator -> Docker runner ->
host verifier -> Judge -> improve); only the "model" is deterministic
(scripted fixture code).  A real API key would substitute the scripted client.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
from ves.search import GreedyTop1Policy
from ves.search_engine import SearchEngine

from ves_modeling.regression.generator import LLMRegressionGenerator
from ves_modeling.regression.problem import build_regression_problem
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
)


class FakeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.responses: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)

IMAGE = os.environ.get("VES_MODELING_DOCKER_IMAGE", "ves-modeling-runner:0.1")

pytestmark = pytest.mark.docker


def docker_ready() -> bool:
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return False
    try:
        version = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        image = subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return version.returncode == 0 and image.returncode == 0


@pytest.mark.skipif(not docker_ready(), reason="Docker or image unavailable")
def test_full_closed_loop_with_scripted_llm(tmp_path: Path) -> None:
    # Build a small dataset directly.
    import pandas as pd
    from sklearn.datasets import make_regression

    public = tmp_path / "public"
    host = tmp_path / "host"
    public.mkdir()
    host.mkdir()
    X, y = make_regression(n_samples=120, n_features=5, noise=10.0, random_state=5)
    feature_names = [f"feature_{i}" for i in range(5)]
    split = 25
    train = pd.DataFrame(X[split:], columns=feature_names)
    train["target"] = y[split:]
    test_features = pd.DataFrame(X[:split], columns=feature_names)
    hidden = pd.DataFrame({"target": y[:split]})
    train.to_csv(public / "train.csv", index=False)
    test_features.to_csv(public / "test_features.csv", index=False)
    hidden.to_csv(host / "hidden_test_labels.csv", index=False)

    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "candidates"
    problem = build_regression_problem(public, host)

    llm = FakeLlm()
    llm.responses = [
        (fixtures / "linear_regression.py").read_text(encoding="utf-8"),
        (fixtures / "random_forest.py").read_text(encoding="utf-8"),
        (fixtures / "gradient_boosting.py").read_text(encoding="utf-8"),
    ]
    generator = LLMRegressionGenerator(llm)

    runner = DockerRegressionRunner(
        DockerRunnerConfig(
            workspace=tmp_path / "runs",
            data_dir=public,
            image=IMAGE,
            timeout_seconds=300,
        )
    )
    engine = SearchEngine(
        problem=problem,
        generator=generator,
        runner=runner,
        anchor_policy=GreedyTop1Policy(),
        drafts=2,
        improves=1,
    )
    result = engine.search()

    assert result.rejected == 0
    assert len(result.records) == 3
    assert result.drafts == 2
    assert result.improves == 1
    assert result.best_feasible is True
    assert result.best_evidence is not None
    rmse = {o.name: o.value for o in result.best_evidence}["rmse"]
    assert rmse > 0.0 and math.isfinite(rmse)
    # Every record carries a distinct candidate id (lineage kept).
    assert len({r.candidate_id for r in result.records}) == 3
    # Docker actually produced the artifact.
    assert len(llm.prompts) == 3
    # Improve prompt carried host evidence from the anchor.
    assert "RMSE:" in llm.prompts[2]
