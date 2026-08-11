"""R8: hidden truth isolation in the Docker runner (real attack when available)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

from ves_modeling.forecasting.problem import build_forecasting_problem
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
)

IMAGE = os.environ.get("VES_MODELING_DOCKER_IMAGE", "ves-modeling-runner:0.1")

pytestmark = pytest.mark.docker

ATTACK_CANDIDATE = '''\
import json, os
import pandas as pd

found = {}
for p in ["/data/hidden_test_labels.csv", "/host/hidden_test_labels.csv",
          "/workspace/hidden_test_labels.csv", "/output/hidden_test_labels.csv",
          "/readonly/hidden_test_labels.csv", "/data/../host/hidden_test_labels.csv"]:
    found[p] = os.path.exists(p)
for d in ["/data", "/host", "/workspace", "/output", "/readonly", "/"]:
    try:
        found[f"ls:{d}"] = sorted(os.listdir(d))
    except OSError as exc:
        found[f"ls:{d}"] = f"err:{exc}"

test = pd.read_csv("/data/test_features.csv")
rows = [
    {
        "series_id": str(row["series_id"]),
        "timestamp": str(row["timestamp"]),
        "prediction": 0.0,
    }
    for _, row in test.iterrows()
]
with open("/output/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": rows, "probe": found}, fh)
'''


@pytest.fixture
def docker_env(tmp_path: Path) -> tuple[Path, Path, Path]:
    public = tmp_path / "public"
    host = tmp_path / "host"
    public.mkdir()
    host.mkdir()
    offset = to_offset("D")
    train_times = pd.date_range("2024-01-01", periods=10, freq=offset)
    test_times = pd.date_range(
        train_times[-1] + offset, periods=5, freq=offset
    )
    train = pd.DataFrame(
        {
            "series_id": ["s0"] * 10,
            "timestamp": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in train_times],
            "target": [float(i) for i in range(10)],
        }
    )
    test_features = pd.DataFrame(
        {
            "series_id": ["s0"] * 5,
            "timestamp": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in test_times],
        }
    )
    hidden = pd.DataFrame(
        {
            "series_id": ["s0"] * 5,
            "timestamp": [
                t.strftime("%Y-%m-%dT%H:%M:%S") for t in test_times
            ],
            "target": [float(i) for i in range(10, 15)],
        }
    )
    train.to_csv(public / "train.csv", index=False)
    test_features.to_csv(public / "test_features.csv", index=False)
    hidden.to_csv(host / "hidden_test_labels.csv", index=False)
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
    config = DockerRunnerConfig(workspace=workspace, data_dir=public, image=IMAGE)
    runner = DockerRegressionRunner(config)
    command = runner.build_command(Path("/tmp/code"), Path("/tmp/out"), "attack0")
    command_text = " ".join(command)
    assert "host" not in command_text.lower() or "hidden_test" not in command_text
    assert "/data/train.csv" in command_text
    assert "/data/test_features.csv" in command_text
    assert "--network" in command_text and "none" in command_text
    assert "--read-only" in command_text
    assert "--cap-drop" in command_text and "ALL" in command_text
    assert "no-new-privileges" in command_text
    assert "VES_DATA_DIR=/data" in command_text
    assert "VES_OUTPUT_DIR=/output" in command_text


@pytest.mark.skipif(not docker_available(), reason="Docker daemon unavailable")
def test_hidden_truth_attack_real_container(docker_env) -> None:
    """Real attack: candidate must not read hidden labels; output verified."""
    public, host, workspace = docker_env
    if subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        check=False,
    ).returncode != 0:
        pytest.skip(f"image {IMAGE} not built; run scripts/build_runner_image.sh")

    problem = build_forecasting_problem(public, host)
    config = DockerRunnerConfig(
        workspace=workspace, data_dir=public, image=IMAGE, timeout_seconds=180
    )
    runner = DockerRegressionRunner(config)
    result = runner.run(ATTACK_CANDIDATE, "attack1")
    assert result.succeeded, result.stderr
    artifact_path = result.run_dir / "predictions.json"
    assert artifact_path.is_file()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    probe = payload["probe"]
    for key, value in probe.items():
        if key.startswith("ls:"):
            continue
        assert value is False, f"hidden labels leaked at {key}: {value}"

    from ves.artifact import SafeArtifactLoader
    from ves.problem import VerificationPipeline

    artifact = SafeArtifactLoader(root=result.run_dir).load("predictions.json")
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(artifact)
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    assert values["rmse"] > 0.0
    assert values["rmse"] == values["rmse"]  # finite
    assert values["smape"] == values["smape"]  # finite
