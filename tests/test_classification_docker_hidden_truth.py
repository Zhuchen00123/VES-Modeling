"""R9: hidden truth isolation in the Docker runner (real attack when available)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ves_modeling.classification.problem import build_classification_problem
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
)

IMAGE = os.environ.get("VES_MODELING_DOCKER_IMAGE", "ves-modeling-runner:0.1")

pytestmark = pytest.mark.docker

ATTACK_CANDIDATE = '''\
import json, os
import numpy as np
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
    {"label": 0, "probabilities": [1.0, 0.0, 0.0]}
    for _ in range(len(test))
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
    X, y = make_classification(
        n_samples=90,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=3,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=30, stratify=y, random_state=3
    )
    feature_names = [f"f{i}" for i in range(4)]
    train = pd.DataFrame(X_train, columns=feature_names)
    train["target"] = y_train
    test_features = pd.DataFrame(X_test, columns=feature_names)
    hidden = pd.DataFrame({"target": y_test})
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

    problem = build_classification_problem(public, host)
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
    assert np.isfinite(values["accuracy"])
    assert np.isfinite(values["macro_f1"])
    assert np.isfinite(values["auroc"])
