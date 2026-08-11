"""R7.3 rework: Docker image digest resolution (P1.4) and RunResult shape."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
    RunResult,
)

_DIGEST = "sha256:" + "c" * 64


def _runner(tmp_path: Path, **kwargs) -> DockerRegressionRunner:
    workspace = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = DockerRunnerConfig(
        workspace=workspace, data_dir=data_dir, **kwargs
    )
    return DockerRegressionRunner(config)


def test_configured_digest_skips_inspect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path, image_digest=_DIGEST)

    def boom(*args, **kwargs):
        raise AssertionError("image inspect must not run when configured")

    monkeypatch.setattr(subprocess, "run", boom)
    runner.resolve_image_digest()
    assert runner.image_digest_status == "configured"
    assert runner.effective_image_digest == _DIGEST
    assert runner.image_digest_error is None


def test_resolve_image_digest_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0, stdout=_DIGEST + "\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner.resolve_image_digest()
    assert runner.image_digest_status == "resolved"
    assert runner.effective_image_digest == _DIGEST
    assert runner.image_digest_error is None


def test_resolve_image_digest_failure_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="", stderr="no such image: ves-modeling-runner"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner.resolve_image_digest()
    assert runner.image_digest_status == "unresolved"
    assert runner.effective_image_digest is None
    assert "no such image" in (runner.image_digest_error or "")


def test_resolve_image_digest_bad_format_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="not-a-digest\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner.resolve_image_digest()
    assert runner.image_digest_status == "unresolved"
    assert runner.effective_image_digest is None
    assert "unexpected image id format" in (runner.image_digest_error or "")


def test_run_result_third_positional_remains_stdout() -> None:
    result = RunResult(True, Path("run"), "hello-stdout")
    assert result.stdout == "hello-stdout"
    assert result.stderr == ""
    assert result.run_root is None
