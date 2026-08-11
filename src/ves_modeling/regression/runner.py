"""CodeRunner adapters for the regression slice.

Trust boundary (idea.md Rule 6):
- LocalRegressionRunner: NOT a security sandbox; trusted fixtures/tests only.
- DockerRegressionRunner: the only execution boundary for real LLM candidates;
  mirrors the VES DockerProcessRunner security parameters while mounting the
  candidate output directory at /output (Gap 1 workaround, docs/core-gaps.md).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_HEX_RE = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class RunResult:
    succeeded: bool
    run_dir: Path
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    run_root: Path | None = None


def validate_run_id(run_id: str) -> None:
    """Reject run ids that could escape a workspace or confuse tooling."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run_id may contain only letters, digits, '_' and '-'")


def _validate_run_id(run_id: str) -> None:
    """Backwards-compatible alias for :func:`validate_run_id`."""
    validate_run_id(run_id)



def _prepare_run_dir(run_root: Path) -> Path:
    """Create a fresh run root; remove a stale one so workspaces are reusable."""
    import shutil as _shutil

    resolved = run_root.resolve()
    if resolved.exists():
        _shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _persist_logs(run_root: Path, stdout: str, stderr: str) -> None:
    """Write full candidate stdout/stderr into the run root for diagnostics.

    Local runs write them at ``runs/<run_id>/stdout.log`` and
    ``runs/<run_id>/stderr.log``; Docker runs write them at
    ``runs/<run_id>/`` (same level as ``code/`` and ``output/``).  Content is
    persisted unfiltered even though ``RunResult`` truncates its copies.
    """
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "stdout.log").write_text(stdout, encoding="utf-8", errors="replace")
    (run_root / "stderr.log").write_text(stderr, encoding="utf-8", errors="replace")

def _normalize_docker_host_path(path: Path) -> Path:
    """Rewrite a /codexprojects prefix to its /mnt/f equivalent (VES pattern)."""
    s = str(path)
    if s.startswith("/codexprojects"):
        alt = "/mnt/f/codexprojects" + s[len("/codexprojects") :]
        logger.warning("docker bind source %s -> %s", s, alt)
        return Path(alt)
    return path


class LocalRegressionRunner:
    """Runs a candidate in a fresh subprocess (trusted fixtures/tests only)."""

    def __init__(
        self,
        workspace: Path,
        data_dir: Path,
        *,
        timeout_seconds: float = 300.0,
        python_executable: str = sys.executable,
        run_layout: Literal["nested", "flat"] = "nested",
    ) -> None:
        self.workspace = Path(workspace)
        self.data_dir = Path(data_dir)
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable
        if run_layout not in ("nested", "flat"):
            raise ValueError("run_layout must be 'nested' or 'flat'")
        self.run_layout = run_layout
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, code: str, run_id: str) -> RunResult:
        _validate_run_id(run_id)
        base = (
            self.workspace
            if self.run_layout == "flat"
            else self.workspace / "runs"
        )
        run_dir = _prepare_run_dir(base / run_id)
        code_path = run_dir / "solution.py"
        code_path.write_text(code, encoding="utf-8")

        environment = os.environ.copy()
        environment.setdefault("PYTHONHASHSEED", "0")
        environment["REGRESSION_DATA_DIR"] = str(self.data_dir)
        environment["REGRESSION_OUTPUT_DIR"] = str(run_dir)

        timed_out = False
        try:
            completed = subprocess.run(
                [self.python_executable, str(code_path)],
                cwd=run_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                timeout=self.timeout_seconds,
                check=False,
            )
            stdout, stderr = completed.stdout, completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = error.stdout or ""
            stderr = (error.stderr or "") + "\n<timed out>"
            returncode = None

        _persist_logs(run_dir, stdout, stderr)
        return RunResult(
            succeeded=not timed_out and returncode == 0,
            run_dir=run_dir,
            run_root=run_dir,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=timed_out,
        )


@dataclass(frozen=True)
class DockerRunnerConfig:
    workspace: Path
    data_dir: Path
    image: str = "ves-modeling-runner:0.1"
    image_digest: str | None = None
    timeout_seconds: float = 900.0
    memory: str = "4g"
    cpus: float = 4.0
    pids_limit: int = 512
    tmpfs_size: str = "128m"
    output_size_limit_bytes: int = 64 * 1024 * 1024
    max_output_chars: int = 50_000
    docker_executable: str = "docker"
    public_files: tuple[str, ...] = ("train.csv", "test_features.csv")
    run_layout: Literal["nested", "flat"] = "nested"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.cpus <= 0:
            raise ValueError("timeout_seconds and cpus must be positive")
        if self.image_digest is not None and not (
            self.image_digest.startswith("sha256:")
            and len(self.image_digest) == 7 + 64
            and _HEX_RE.fullmatch(self.image_digest[7:])
        ):
            raise ValueError(
                "image_digest must be a sha256:... digest"
            )
        if self.pids_limit <= 0 or self.max_output_chars <= 0:
            raise ValueError("pids_limit and max_output_chars must be positive")
        if not self.memory.strip() or not self.tmpfs_size.strip():
            raise ValueError("memory and tmpfs_size cannot be empty")
        for name in self.public_files:
            if not name or Path(name).name != name or "\\" in name:
                raise ValueError(
                    f"public_files entries must be plain file names, got {name!r}"
                )
        if self.run_layout not in ("nested", "flat"):
            raise ValueError("run_layout must be 'nested' or 'flat'")


class DockerRegressionRunner:
    """Resource-limited, network-disabled Docker sandbox for LLM candidates.

    Mounts:
      code      -> /readonly (ro)
      public data files -> /data/train.csv, /data/test_features.csv (ro, whitelist)
      output    -> /output (writable)
    Hidden labels are never mounted (public_files whitelist).

    ``public_files`` is a strict whitelist: only the listed files are mounted
    (one bind per file), never the whole data directory.  When it is empty no
    /data mount is created at all (no legacy whole-directory fallback), and
    every missing file raises before the container starts.
    """

    def __init__(self, config: DockerRunnerConfig) -> None:
        self.config = config
        self._image_digest: str | None = None
        self._image_digest_error: str | None = None
        try:
            self._workspace = _normalize_docker_host_path(
                config.workspace.resolve()
            )
            self._data_dir = _normalize_docker_host_path(config.data_dir.resolve())
        except OSError as error:
            raise RuntimeError(f"cannot resolve runner paths: {error}") from error
        self._workspace.mkdir(parents=True, exist_ok=True)

    @property
    def image_ref(self) -> str:
        if self.config.image_digest is not None:
            return f"{self.config.image}@{self.config.image_digest}"
        return self.config.image

    @property
    def effective_image_digest(self) -> str | None:
        """Configured digest, or the digest resolved via image inspect."""
        return self.config.image_digest or self._image_digest

    @property
    def image_digest_error(self) -> str | None:
        """Why digest resolution failed (None when configured/resolved)."""
        return None if self.config.image_digest is not None else self._image_digest_error

    @property
    def image_digest_status(self) -> str:
        """'configured', 'resolved' or 'unresolved'."""
        if self.config.image_digest is not None:
            return "configured"
        if self._image_digest is not None:
            return "resolved"
        return "unresolved"

    def resolve_image_digest(self) -> None:
        """Best-effort resolve the image ID via ``docker image inspect``.

        Only runs when no digest was configured.  Failures are recorded in
        ``image_digest_error`` so callers never mistake an unresolved digest
        for a recorded one.
        """
        if self.config.image_digest is not None:
            return
        if self._image_digest is not None or self._image_digest_error is not None:
            return
        try:
            completed = subprocess.run(
                [
                    self.config.docker_executable,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    self.config.image,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._image_digest_error = f"image inspect failed: {exc}"
            return
        if completed.returncode != 0:
            self._image_digest_error = (
                completed.stderr.strip()
                or f"image inspect failed with rc={completed.returncode}"
            )
            return
        digest = completed.stdout.strip()
        if digest.startswith("sha256:") and _HEX_RE.fullmatch(digest[7:]):
            self._image_digest = digest
        else:
            self._image_digest_error = (
                f"unexpected image id format from inspect: {digest!r}"
            )

    def is_available(self) -> bool:
        executable = shutil.which(self.config.docker_executable)
        if executable is None:
            return False
        try:
            completed = subprocess.run(
                [
                    self.config.docker_executable,
                    "version",
                    "--format",
                    "{{.Server.Version}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    def build_command(
        self, code_dir: Path, output_dir: Path, run_id: str
    ) -> list[str]:
        _validate_run_id(run_id)
        user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "65534:65534"
        command = [
            self.config.docker_executable,
            "run",
            "--rm",
            "--name",
            f"ves-modeling-{run_id}",
            "--network",
            "none",
            "--memory",
            self.config.memory,
            "--cpus",
            str(self.config.cpus),
            "--pids-limit",
            str(self.config.pids_limit),
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            user,
            "--ulimit",
            "nofile=256:256",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,size={self.config.tmpfs_size}",
            "--mount",
            f"type=bind,src={code_dir.resolve()},dst=/readonly,ro",
            "--mount",
            f"type=bind,src={output_dir.resolve()},dst=/output",
        ]
        for name in self.config.public_files:
            source = self._data_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"public data file missing: {source}")
            command.extend(
                [
                    "--mount",
                    f"type=bind,src={source.resolve()},dst=/data/{name},ro",
                ]
            )
        thread_count = max(1, round(self.config.cpus))
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "JOBLIB_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            command.extend(["--env", f"{var}={thread_count}"])
        command.extend(
            [
                "--env",
                "REGRESSION_DATA_DIR=/data",
                "--env",
                "REGRESSION_OUTPUT_DIR=/output",
                "--workdir",
                "/output",
                self.image_ref,
                "python",
                "/readonly/solution.py",
            ]
        )
        return command

    def run(self, code: str, run_id: str) -> RunResult:
        _validate_run_id(run_id)
        if not self.is_available():
            raise RuntimeError(
                "Docker daemon unavailable; enable Docker Desktop WSL integration"
            )
        if self.config.image_digest is None:
            self.resolve_image_digest()
        base = (
            self._workspace
            if self.config.run_layout == "flat"
            else self._workspace / "runs"
        )
        run_root = _prepare_run_dir(base / run_id)
        code_dir = run_root / "code"
        output_dir = run_root / "output"
        code_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        code_path = code_dir / "solution.py"
        code_path.write_text(code, encoding="utf-8")
        command = self.build_command(code_dir, output_dir, run_id)

        timed_out = False
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.timeout_seconds,
                check=False,
            )
            stdout, stderr = completed.stdout, completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = error.stdout or ""
            stderr = (error.stderr or "") + "\n<timed out>"
            returncode = None
            self._force_remove(run_id)

        if not timed_out and _dir_size(output_dir) > self.config.output_size_limit_bytes:
            self._force_remove(run_id)
            stderr = (
                stderr
                + f"\noutput exceeded {self.config.output_size_limit_bytes} bytes"
            )
            returncode = -1

        _persist_logs(run_root, stdout, stderr)
        return RunResult(
            succeeded=not timed_out and returncode == 0,
            run_dir=output_dir,
            run_root=run_root,
            stdout=self._limit(stdout),
            stderr=self._limit(stderr),
            returncode=returncode,
            timed_out=timed_out,
        )

    def _force_remove(self, run_id: str) -> None:
        container_name = f"ves-modeling-{run_id}"
        try:
            subprocess.run(
                [self.config.docker_executable, "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _limit(self, value: str) -> str:
        if len(value) <= self.config.max_output_chars:
            return value
        omitted = len(value) - self.config.max_output_chars
        return (
            value[: self.config.max_output_chars]
            + f"\n... <{omitted} chars omitted>"
        )


def _dir_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total
