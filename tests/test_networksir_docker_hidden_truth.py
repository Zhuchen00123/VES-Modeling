"""R30: read-only problem mount in the Docker runner (real attack)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ves_modeling.networksir.problem import build_networksir_problem
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
)

IMAGE = os.environ.get("VES_MODELING_DOCKER_IMAGE", "ves-modeling-runner:0.1")

pytestmark = pytest.mark.docker

PROBLEM = {
    "version": 1,
    "model": "network_sir",
    "beta": 0.3,
    "gamma": 0.1,
    "n_nodes": 10,
    "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]],
    "i0": 1,
    "t_end": 20,
    "quantity": "final_size",
}

ATTACK_CANDIDATE = '''\
import json, os

found = {}
for p in ["/data/train.csv", "/data/test_features.csv",
          "/data/hidden_test_sequences.csv", "/data/hidden_parameters.json",
          "/data/hidden_test_outcomes.csv", "/data/hidden_test_labels.csv",
          "/data/hidden_test_changepoints.csv", "/output/problem.json",
          "/readonly/problem.json"]:
    found[p] = os.path.exists(p)
for d in ["/data", "/output", "/readonly", "/"]:
    try:
        found[f"ls:{d}"] = sorted(os.listdir(d))
    except OSError as exc:
        found[f"ls:{d}"] = f"err:{exc}"

with open("/data/problem.json", encoding="utf-8") as fh:
    problem = json.load(fh)
with open("/output/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"estimate": 0.5, "probe": found,
               "model_seen": problem["model"]}, fh)
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
    for forbidden in (
        "train.csv",
        "test_features.csv",
        "hidden_test_sequences.csv",
        "hidden_parameters.json",
        "hidden_test_outcomes.csv",
        "hidden_test_labels.csv",
        "hidden_test_changepoints.csv",
    ):
        assert forbidden not in command_text
    assert "--read-only" in command_text
    assert "--network" in command_text and "none" in command_text
    assert "--cap-drop" in command_text and "ALL" in command_text
    assert "no-new-privileges" in command_text


@pytest.mark.skipif(not docker_available(), reason="Docker daemon unavailable")
def test_read_only_problem_mount_real_container(docker_env) -> None:
    """Real attack: only problem.json is visible; metrics still recomputed."""
    public, workspace = docker_env
    if subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        check=False,
    ).returncode != 0:
        pytest.skip(f"image {IMAGE} not built; run scripts/build_runner_image.sh")

    problem = build_networksir_problem(public)
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
    assert payload["model_seen"] == "network_sir"

    from ves.artifact import SafeArtifactLoader
    from ves.problem import VerificationPipeline

    artifact = SafeArtifactLoader(root=result.run_dir).load("solution.json")
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(artifact)
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["absolute_error"] >= 0.0
    assert values["relative_error"] >= 0.0
