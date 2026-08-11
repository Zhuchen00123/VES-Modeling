"""R10: read-only problem mount in the Docker runner (real attack when available)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ves_modeling.optimization.problem import build_optimization_problem
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
)

IMAGE = os.environ.get("VES_MODELING_DOCKER_IMAGE", "ves-modeling-runner:0.1")

pytestmark = pytest.mark.docker

PROBLEM = {
    "version": 1,
    "sense": "minimize",
    "variables": {
        "x0": {"type": "continuous", "lower": 0.0, "upper": 10.0},
        "x1": {"type": "continuous", "lower": 0.0, "upper": 10.0},
    },
    "objective": {"coefficients": {"x0": 1.0, "x1": 2.0}, "constant": 3.0},
    "constraints": [],
}

ATTACK_CANDIDATE = '''\
import json, os

found = {}
for p in ["/data/hidden_test_labels.csv", "/host/hidden_test_labels.csv",
          "/data/train.csv", "/data/test_features.csv",
          "/output/problem.json", "/readonly/problem.json"]:
    found[p] = os.path.exists(p)
for d in ["/data", "/output", "/readonly", "/"]:
    try:
        found[f"ls:{d}"] = sorted(os.listdir(d))
    except OSError as exc:
        found[f"ls:{d}"] = f"err:{exc}"

with open("/data/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)
values = {}
for name, spec in problem["variables"].items():
    values[name] = (spec["lower"] + spec["upper"]) / 2.0
with open("/output/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"variables": values, "probe": found}, fh)
'''


@pytest.fixture
def docker_env(tmp_path: Path) -> tuple[Path, Path]:
    public = tmp_path / "public"
    public.mkdir()
    (public / "problem.json").write_text(
        json.dumps(PROBLEM), encoding="utf-8"
    )
    workspace = tmp_path / "runs"
    return public, workspace


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def test_runner_command_mounts_only_problem_json(docker_env) -> None:
    public, workspace = docker_env
    config = DockerRunnerConfig(
        workspace=workspace,
        data_dir=public,
        image=IMAGE,
        public_files=("problem.json",),
    )
    runner = DockerRegressionRunner(config)
    command = runner.build_command(Path("/tmp/code"), Path("/tmp/out"), "attack0")
    command_text = " ".join(command)
    assert "/data/problem.json" in command_text
    for forbidden in ("train.csv", "test_features.csv", "hidden_test_labels"):
        assert forbidden not in command_text
    assert "--read-only" in command_text
    assert "--network" in command_text and "none" in command_text
    assert "--cap-drop" in command_text and "ALL" in command_text
    assert "no-new-privileges" in command_text


@pytest.mark.skipif(not docker_available(), reason="Docker daemon unavailable")
def test_read_only_problem_mount_real_container(docker_env) -> None:
    """Real attack: only problem.json is visible; solution still verified."""
    public, workspace = docker_env
    if subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        check=False,
    ).returncode != 0:
        pytest.skip(f"image {IMAGE} not built; run scripts/build_runner_image.sh")

    problem = build_optimization_problem(public)
    config = DockerRunnerConfig(
        workspace=workspace,
        data_dir=public,
        image=IMAGE,
        timeout_seconds=180,
        public_files=("problem.json",),
    )
    runner = DockerRegressionRunner(config)
    result = runner.run(ATTACK_CANDIDATE, "attack1")
    assert result.succeeded, result.stderr
    artifact_path = result.run_dir / "solution.json"
    assert artifact_path.is_file()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    probe = payload["probe"]
    for key, value in probe.items():
        if key.startswith("ls:"):
            continue
        assert value is False, f"unexpected file visible at {key}: {value}"

    from ves.artifact import SafeArtifactLoader
    from ves.problem import VerificationPipeline

    artifact = SafeArtifactLoader(root=result.run_dir).load("solution.json")
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(artifact)
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["objective"] == pytest.approx(3.0 + 5.0 + 2.0 * 5.0)
    assert values["max_bound_violation"] == 0.0
