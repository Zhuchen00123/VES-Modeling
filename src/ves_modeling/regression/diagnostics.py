"""Candidate diagnostic persistence for the Regression delivery API.

VES Core 0.1.0 exposes structured ``SearchResult.attempts``.  This module only
maps those public outcomes to VES-Modeling's persisted ``run.json`` contract;
it does not copy or override Core search control flow.

Candidate statuses (contract ``docs/r7.3-delivery-contract.md``):
``execution_failed`` / ``timeout`` / ``artifact_missing`` /
``artifact_invalid`` / ``verification_failed`` / ``verified``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ves.search_engine import AttemptStatus

# Preserve the JSON-facing v1 ordering while deriving every value from Core's
# public enum (no duplicated status strings/control-flow protocol).
CANDIDATE_STATUSES = tuple(
    status.value for status in AttemptStatus if status is not AttemptStatus.VERIFIED
) + (AttemptStatus.VERIFIED.value,)
APPLY_STATUSES = ("produced_unverified",)


def sha256_text(text: str) -> str:
    """SHA-256 of UTF-8 text (used for candidate code and summaries)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_solution_path(run_root: Path) -> str | None:
    """Relative solution path inside a candidate run root, if present."""
    if (run_root / "solution.py").is_file():
        return "solution.py"
    if (run_root / "code" / "solution.py").is_file():
        return "code/solution.py"
    return None


def detect_artifact_path(run_root: Path) -> str | None:
    """Relative artifact path inside a candidate run root, if present."""
    if (run_root / "predictions.json").is_file():
        return "predictions.json"
    if (run_root / "output" / "predictions.json").is_file():
        return "output/predictions.json"
    return None


def write_run_json(
    run_root: Path,
    *,
    candidate: str,
    status: str,
    code_sha256: str | None = None,
    artifact_sha256: str | None = None,
    evidence: dict[str, float] | None = None,
    issues: tuple[str, ...] = (),
    search_id: str | None = None,
) -> Path:
    """Persist a structured ``run.json`` for one candidate attempt."""
    if status not in CANDIDATE_STATUSES and status not in APPLY_STATUSES:
        raise ValueError(f"unknown candidate status: {status!r}")
    if candidate.startswith("improve"):
        generation = "improve"
        index = int(candidate[len("improve") :])
    elif candidate.startswith("draft"):
        generation = "draft"
        index = int(candidate[len("draft") :])
    else:
        generation = "apply"
        index = 0
    payload: dict[str, Any] = {
        "schema_version": 1,
        "candidate": candidate,
        "generation": generation,
        "index": index,
        "status": status,
        "issues": list(issues),
        "code_sha256": code_sha256,
        "solution": detect_solution_path(run_root),
        "stdout": "stdout.log",
        "stderr": "stderr.log",
    }
    if search_id is not None:
        payload["search_id"] = search_id
    if artifact_sha256 is not None:
        payload["artifact_sha256"] = artifact_sha256
    if evidence is not None:
        payload["evidence"] = {
            name: value for name, value in sorted(evidence.items())
        }
    payload["artifact"] = detect_artifact_path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "run.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
