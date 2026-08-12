"""Applying a change-point solution to caller-provided data (R25).

Untrusted/LLM code is executed in the Docker sandbox by default (only
``train.csv`` and ``test_features.csv`` are mounted, read-only); local
execution requires an explicit ``trusted_code=True`` opt-in.  The apply
status remains ``produced_unverified`` and no quality metric is produced.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ves.artifact import SafeArtifactLoader

from ves_modeling.changepoint.data_contract import (
    validate_changepoint_data,
    validate_changepoints,
)
from ves_modeling.changepoint.diagnostics import sha256_text, write_run_json
from ves_modeling.changepoint.schema import API_SCHEMA_VERSION
from ves_modeling.regression.provenance import (
    file_hashes,
    package_versions,
    sha256_file,
)
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
    LocalRegressionRunner,
    validate_run_id,
)

APPLY_SUCCESS_STATUS = "produced_unverified"


@dataclass(frozen=True)
class ApplyChangepointResult:
    """Outcome of applying a solution; structural checks only."""

    status: str
    run_id: str
    run_dir: Path
    candidate_dir: Path
    solutions_path: Path | None
    code_sha256: str
    data_sha256: dict[str, str]
    solutions_sha256: str | None
    stdout_log: Path
    stderr_log: Path
    runner: str
    docker_image: str | None
    docker_digest: str | None
    timeout_seconds: float | None
    resources: dict[str, Any] | None
    data_contract: dict[str, Any] | None = None
    error: str | None = None

    def to_summary(self) -> dict[str, Any]:
        """JSON-serializable summary (relative artifact paths only)."""
        return {
            "schema_version": API_SCHEMA_VERSION,
            "run_id": self.run_id,
            "task": "changepoint",
            "operation": "apply",
            "status": self.status,
            "error": self.error,
            "runner": self.runner,
            "code_sha256": self.code_sha256,
            "data_sha256": self.data_sha256,
            "solutions_sha256": self.solutions_sha256,
            "stdout_log": self._relative_to_run_dir(self.stdout_log),
            "stderr_log": self._relative_to_run_dir(self.stderr_log),
            "candidate_dir": self._relative_to_run_dir(self.candidate_dir),
            "solutions": (
                self._relative_to_run_dir(self.solutions_path)
                if self.solutions_path is not None
                else None
            ),
            "docker_image": self.docker_image,
            "docker_digest": self.docker_digest,
            "timeout_seconds": self.timeout_seconds,
            "resources": self.resources,
            "data_contract": self.data_contract,
        }

    def _relative_to_run_dir(self, path: Path) -> str:
        return os.path.relpath(str(path), str(self.run_dir))


def _load_solutions_payload(artifact_path: Path) -> dict[str, Any]:
    """Race-safely load the artifact and require a JSON object root."""
    artifact = SafeArtifactLoader(root=artifact_path.parent).load(
        artifact_path.name
    )
    try:
        data = json.loads(artifact.content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"changepoints.json is not valid JSON: {exc}"
        ) from None
    if not isinstance(data, dict):
        raise ValueError("changepoints.json root must be an object")
    return data


def apply_changepoint_solution(
    solution: str | Path,
    public_dir: Path,
    *,
    workspace: Path | None = None,
    run_id: str | None = None,
    trusted_code: bool = False,
    image: str = "ves-modeling-runner:0.1",
    image_digest: str | None = None,
    timeout_seconds: float = 900.0,
    docker_executable: str = "docker",
) -> ApplyChangepointResult:
    """Apply ``solution`` to ``public_dir`` (train/test CSVs)."""
    code = (
        Path(solution).read_text(encoding="utf-8")
        if isinstance(solution, Path)
        else solution
    )
    if not code.strip():
        raise ValueError("solution must be non-empty")
    public_dir = Path(public_dir)
    file_hashes(public_dir, ("train.csv", "test_features.csv"), strict=True)
    contract = validate_changepoint_data(public_dir)
    workspace = Path(workspace or Path.cwd() / "runs")
    workspace.mkdir(parents=True, exist_ok=True)
    run_id = run_id or uuid.uuid4().hex[:12]
    validate_run_id(run_id)
    run_dir = workspace / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    if trusted_code:
        runner = LocalRegressionRunner(
            workspace=run_dir,
            data_dir=public_dir,
            timeout_seconds=timeout_seconds,
            run_layout="flat",
        )
        runner_kind = "local"
        docker_image: str | None = None
        docker_digest: str | None = None
        resources: dict[str, Any] | None = None
    else:
        config = DockerRunnerConfig(
            workspace=run_dir,
            data_dir=public_dir,
            image=image,
            image_digest=image_digest,
            timeout_seconds=timeout_seconds,
            docker_executable=docker_executable,
            run_layout="flat",
            public_files=("train.csv", "test_features.csv"),
        )
        runner = DockerRegressionRunner(config)
        runner_kind = "docker"
        docker_image = config.image
        docker_digest = config.image_digest
        resources = {
            "memory": config.memory,
            "cpus": config.cpus,
            "pids_limit": config.pids_limit,
            "tmpfs_size": config.tmpfs_size,
            "public_files": list(config.public_files),
        }

    run_result = runner.run(code, run_id="candidate")
    candidate_dir = run_result.run_root or run_result.run_dir
    artifact_path = candidate_dir / "changepoints.json"
    if not artifact_path.is_file():
        artifact_path = candidate_dir / "output" / "changepoints.json"
    code_sha256 = sha256_text(code)
    issues: tuple[str, ...] = ()
    solutions_sha256: str | None = None
    if not run_result.succeeded:
        status = "timeout" if run_result.timed_out else "execution_failed"
        issues = (f"returncode={run_result.returncode}",)
    elif not artifact_path.is_file():
        status = "artifact_missing"
    else:
        try:
            payload = _load_solutions_payload(artifact_path)
            validate_changepoints(payload, n=contract.test_rows)
            solutions_sha256 = sha256_file(artifact_path)
            status = APPLY_SUCCESS_STATUS
        except ValueError as exc:
            status = "artifact_invalid"
            issues = (str(exc),)

    write_run_json(
        candidate_dir,
        candidate="candidate",
        status=status,
        code_sha256=code_sha256,
        artifact_sha256=solutions_sha256,
        issues=issues,
    )
    data_hashes = file_hashes(
        public_dir, ("train.csv", "test_features.csv")
    )
    if runner_kind == "docker":
        docker_digest = runner.effective_image_digest
        digest_status = runner.image_digest_status
        digest_error = runner.image_digest_error
    else:
        digest_status = None
        digest_error = None
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "schema_version": API_SCHEMA_VERSION,
                "operation": "apply",
                "runner": runner_kind,
                "trusted_code": trusted_code,
                "image": docker_image,
                "image_digest": docker_digest,
                "image_digest_status": digest_status,
                "image_digest_error": digest_error,
                "timeout_seconds": timeout_seconds,
                "resources": resources,
                "data_contract": contract.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    error_message = issues[0] if issues else None
    summary = ApplyChangepointResult(
        status=status,
        run_id=run_id,
        run_dir=run_dir,
        candidate_dir=candidate_dir,
        solutions_path=(
            artifact_path if status == APPLY_SUCCESS_STATUS else None
        ),
        code_sha256=code_sha256,
        data_sha256=data_hashes,
        solutions_sha256=solutions_sha256,
        stdout_log=candidate_dir / "stdout.log",
        stderr_log=candidate_dir / "stderr.log",
        runner=runner_kind,
        docker_image=docker_image,
        docker_digest=docker_digest,
        timeout_seconds=timeout_seconds,
        resources=resources,
        data_contract=contract.to_dict(),
        error=error_message,
    )
    summary_payload = summary.to_summary()
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": API_SCHEMA_VERSION,
                "run_id": run_id,
                "operation": "apply",
                "created_at": datetime.datetime.now(
                    datetime.UTC
                ).isoformat(),
                "inputs": {"public": data_hashes},
                "code_sha256": code_sha256,
                "solutions_sha256": solutions_sha256,
                "versions": package_versions(),
                "runner": {
                    "kind": runner_kind,
                    "image": docker_image,
                    "image_digest": docker_digest,
                    "image_digest_status": digest_status,
                    "image_digest_error": digest_error,
                    "timeout_seconds": timeout_seconds,
                    "resources": resources,
                },
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                },
                "data_contract": contract.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if status != APPLY_SUCCESS_STATUS:
        detail = f": {error_message}" if error_message else ""
        raise RuntimeError(f"candidate {status}{detail}")
    return summary
