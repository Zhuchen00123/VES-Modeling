"""R23: hidden truth isolation in the Docker runner (real attack when available)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.markov.problem import build_markov_problem
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
)

IMAGE = os.environ.get("VES_MODELING_DOCKER_IMAGE", "ves-modeling-runner:0.1")

pytestmark = pytest.mark.docker

ATTACK_CANDIDATE = '''\
import json, os

found = {}
for p in ["/data/hidden_parameters.json", "/host/hidden_parameters.json",
          "/workspace/hidden_parameters.json", "/output/hidden_parameters.json",
          "/readonly/hidden_parameters.json", "/data/../host/hidden_parameters.json"]:
    found[p] = os.path.exists(p)
for d in ["/data", "/host", "/workspace", "/output", "/readonly", "/"]:
    try:
        found[f"ls:{d}"] = sorted(os.listdir(d))
    except OSError as exc:
        found[f"ls:{d}"] = f"err:{exc}"

with open("/output/solution.json", "w", encoding="utf-8") as fh:
    json.dump({"estimate": 0.3, "confidence_interval": [0.2, 0.4], "probe": found}, fh)
'''


@pytest.fixture
def docker_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(3)
    p = np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=float)
    current = 0
    rows = []
    for _ in range(80):
        rows.append({"state": ["a", "b"][current]})
        current = int(rng.choice(2, p=p[current]))
    public = tmp_path / "public"
    host = tmp_path / "host"
    public.mkdir()
    host.mkdir()
    (public / "problem.json").write_text(
        json.dumps(
            {
                "version": 1,
                "quantity": "transition_probability",
                "states": ["a", "b"],
                "from_state": "a",
                "to_state": "b",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(public / "train.csv", index=False)
    (host / "hidden_parameters.json").write_text(
        json.dumps({"transition_matrix": p.tolist()}), encoding="utf-8"
    )
    workspace = tmp_path / "runs"
    return public, host, workspace


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


def test_runner_command_never_mounts_host_dir(docker_env) -> None:
    public, _host, workspace = docker_env
    config = DockerRunnerConfig(
        workspace=workspace,
        data_dir=public,
        image=IMAGE,
        public_files=("problem.json", "train.csv"),
    )
    runner = DockerRegressionRunner(config)
    command = runner.build_command(Path("/tmp/code"), Path("/tmp/out"), "attack0")
    command_text = " ".join(command)
    assert "hidden_parameters" not in command_text
    assert "/data/problem.json" in command_text
    assert "/data/train.csv" in command_text
    assert "--network" in command_text and "none" in command_text
    assert "--read-only" in command_text
    assert "--cap-drop" in command_text and "ALL" in command_text
    assert "no-new-privileges" in command_text


@pytest.mark.skipif(not docker_available(), reason="Docker daemon unavailable")
def test_hidden_truth_attack_real_container(docker_env) -> None:
    """Real attack: hidden matrix never mounted; solution still verified."""
    public, host, workspace = docker_env
    if subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        check=False,
    ).returncode != 0:
        pytest.skip(f"image {IMAGE} not built; run scripts/build_runner_image.sh")

    problem = build_markov_problem(public, host)
    config = DockerRunnerConfig(
        workspace=workspace,
        data_dir=public,
        image=IMAGE,
        timeout_seconds=180,
        public_files=("problem.json", "train.csv"),
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
        assert value is False, f"hidden parameters leaked at {key}: {value}"

    from ves.artifact import SafeArtifactLoader
    from ves.problem import VerificationPipeline

    artifact = SafeArtifactLoader(root=result.run_dir).load("solution.json")
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(artifact)
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert np.isfinite(values["absolute_error"])
