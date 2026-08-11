"""Provenance helpers for search and application runs (R7.3 Batch A).

Persisted provenance never contains API keys, tokens, raw hidden labels or
absolute host paths; input files are recorded as SHA-256 fingerprints only.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(
    directory: Path, names: tuple[str, ...], *, strict: bool = False
) -> dict[str, str]:
    """SHA-256 fingerprints for the named files under ``directory``.

    With ``strict=True`` a missing file raises ``FileNotFoundError`` (input
    contract preflight); the default silently skips missing files to preserve
    compatibility with label-injection callers.
    """
    hashes: dict[str, str] = {}
    for name in names:
        path = Path(directory) / name
        if path.is_file():
            hashes[name] = sha256_file(path)
        elif strict:
            raise FileNotFoundError(f"required input file missing: {path}")
    return hashes


def _git_commit(repo_root: Path | None) -> str | None:
    """HEAD commit for a repository, when the working copy provides one."""
    if repo_root is None or not (repo_root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def package_versions() -> dict[str, dict[str, str | None]]:
    """VES-Modeling and VES Core versions/commits when obtainable."""
    import ves

    import ves_modeling

    def repo_root(package_module: Any, *, up: int) -> Path | None:
        try:
            path = Path(package_module.__file__).resolve()
        except AttributeError:
            return None
        for _ in range(up):
            path = path.parent
        return path

    return {
        "ves_modeling": {
            "version": getattr(ves_modeling, "__version__", None),
            "commit": _git_commit(repo_root(ves_modeling, up=3)),
        },
        "ves_core": {
            "version": getattr(ves, "__version__", None),
            "commit": _git_commit(repo_root(ves, up=2)),
        },
    }


def sanitize_provider(base_url: str | None, model: str | None) -> dict | None:
    """Sanitized provider identity (hostname + model, never credentials)."""
    if not base_url and not model:
        return None
    host: str | None = None
    if base_url:
        try:
            from urllib.parse import urlparse

            host = urlparse(base_url).hostname or None
        except ValueError:
            host = None
    return {"host": host, "model": model or None}
