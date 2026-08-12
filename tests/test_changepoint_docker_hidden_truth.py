"""R25: read-only public CSV mount in the Docker runner (real attack)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from ves_modeling.changepoint.problem import build_changepoint_problem
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
)

IMAGE = os.environ.get("VES_MODELING_DOCKER_IMAGE", "ves-modeling-runner:0.1")

pytestmark = pytest.mark.docker

ATTACK_CANDIDATE = '''\
import json, os

found = {}
for p in ["/data/hidden_test_changepoints.csv", "/host/hidden_test_changepoints.csv",
          "/data/hidden_parameters.json", "/data/hidden_test_outcomes.csv",
          "/data/problem.json", "/output/train.csv",
          "/readonly/train.csv"]:
    found[p] = os.path.exists(p)
for d in ["/data", "/output", "/readonly", "/"]:
    try:
        found[f"ls:{d}"] = sorted(os.listdir(d))
    except OSError as exc:
        found[f"ls:{d}"] = f"err:{exc}"

with open("/data/train.csv", encoding="utf-8") as fh:
    train = fh.read()
with open("/data/test_features.csv", encoding="utf-8") as fh:
    test = fh.read()
with open("/output/changepoints.json", "w", encoding="utf-8") as fh:
    json.dump({"changepoints": [30], "probe": found,
               "train_seen": "t" in train, "test_seen": "y" in test}, fh)
'''


def _write_data(root: Path) -> tuple[Path, Path]:
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(
        {"t": [float(i) for i in range(40)], "y": [0.0] * 40}
    ).to_csv(public / "train.csv", index=False)
    y = [0.0] * 30 + [10.0] * 30
    pd.DataFrame(
        {"t": [float(i) for i in range(60)], "y": y}
    ).to_csv(public / "test_features.csv", index=False)
    pd.DataFrame({"changepoint": [30]}).to_csv(
        host / "hidden_test_changepoints.csv", index=False
    )
    return public, host


@pytest.fixture
def docker_env(tmp_path: Path) -> tuple[Path, Path]:
    public, host = _write_data(tmp_path)
    return public, host


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


def test_runner_command_mounts_only_public_csvs(docker_env) -> None:
    public, _ = docker_env
    config = DockerRunnerConfig(
        workspace=public.parent / "runs",
        data_dir=public,
        image=IMAGE,
        public_files=("train.csv", "test_features.csv"),
    )
    runner = DockerRegressionRunner(config)
    command = runner.build_command(Path("/tmp/code"), Path("/tmp/out"), "attack0")
    command_text = " ".join(command)
    assert "/data/train.csv" in command_text
    assert "/data/test_features.csv" in command_text
    for forbidden in (
        "hidden_test_changepoints.csv",
        "problem.json",
        "hidden_test_outcomes.csv",
        "hidden_parameters.json",
        "hidden_test_labels.csv",
    ):
        assert forbidden not in command_text
    assert "--read-only" in command_text
    assert "--network" in command_text and "none" in command_text
    assert "--cap-drop" in command_text and "ALL" in command_text
    assert "no-new-privileges" in command_text


@pytest.mark.skipif(not docker_available(), reason="Docker daemon unavailable")
def test_read_only_public_csv_mount_real_container(docker_env) -> None:
    """Real attack: only the two public CSVs are visible."""
    public, host = docker_env
    if subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        check=False,
    ).returncode != 0:
        pytest.skip(f"image {IMAGE} not built; run scripts/build_runner_image.sh")

    problem = build_changepoint_problem(public, host)
    config = DockerRunnerConfig(
        workspace=public.parent / "runs",
        data_dir=public,
        image=IMAGE,
        timeout_seconds=180,
        public_files=("train.csv", "test_features.csv"),
    )
    runner = DockerRegressionRunner(config)
    result = runner.run(ATTACK_CANDIDATE, "attack1")
    assert result.succeeded, result.stderr
    artifact_path = result.run_dir / "changepoints.json"
    assert artifact_path.is_file()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    probe = payload["probe"]
    for key, value in probe.items():
        if key.startswith("ls:"):
            continue
        assert value is False, f"unexpected file visible at {key}: {value}"
    assert payload["train_seen"] is True
    assert payload["test_seen"] is True

    from ves.artifact import SafeArtifactLoader
    from ves.problem import VerificationPipeline

    artifact = SafeArtifactLoader(root=result.run_dir).load(
        "changepoints.json"
    )
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(artifact)
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["f1"] == 1.0
    assert values["mean_distance"] == 0.0
