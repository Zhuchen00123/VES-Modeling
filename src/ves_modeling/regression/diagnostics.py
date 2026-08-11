"""Application-layer candidate diagnostics (R7.3 Batch A).

VES Core's ``SearchEngine`` does not expose per-attempt rejection detail; this
module re-derives the classification in the application layer and persists a
structured ``run.json`` per candidate.  Core is never modified.

Candidate statuses (contract ``docs/r7.3-delivery-contract.md``):
``execution_failed`` / ``timeout`` / ``artifact_missing`` /
``artifact_invalid`` / ``verification_failed`` / ``verified``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ves.artifact import SafeArtifactLoader
from ves.problem import VerificationStatus
from ves.record import Candidate, VerifiedCandidate
from ves.search_engine import SearchEngine

from ves_modeling.regression.runner import RunResult

CANDIDATE_STATUSES = (
    "execution_failed",
    "timeout",
    "artifact_missing",
    "artifact_invalid",
    "verification_failed",
    "verified",
)
APPLY_STATUSES = ("produced_unverified",)


def sha256_text(text: str) -> str:
    """SHA-256 of UTF-8 text (used for candidate code and summaries)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateOutcome:
    """One attempted candidate: code, execution result and classification."""

    candidate: str
    code: str
    run_result: RunResult
    status: str
    issues: tuple[str, ...] = ()
    artifact_sha256: str | None = None
    evidence: dict[str, float] | None = None
    core_candidate_id: str | None = None


class ClassifyingSearchEngine(SearchEngine):
    """SearchEngine that records every attempted candidate's outcome.

    The override mirrors ``SearchEngine._run_and_verify`` and additionally
    captures the exact rejection reason.  It runs entirely in the application
    layer; the vendored Core is not modified.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.outcomes: dict[str, CandidateOutcome] = {}

    def _run_and_verify(  # type: ignore[override]
        self,
        code: str,
        run_id: str,
        parent: VerifiedCandidate | None = None,
    ) -> tuple[VerifiedCandidate, Any] | None:
        run_result = self.runner.run(code, run_id=run_id)
        if not run_result.succeeded:
            status = "timeout" if run_result.timed_out else "execution_failed"
            self.outcomes[run_id] = CandidateOutcome(
                candidate=run_id,
                code=code,
                run_result=run_result,
                status=status,
                issues=(f"returncode={run_result.returncode}",),
            )
            return None
        artifact_path = Path(run_result.run_dir) / self._artifact_filename
        try:
            artifact = SafeArtifactLoader(root=run_result.run_dir).load(
                self._artifact_filename
            )
        except ValueError as exc:
            status = (
                "artifact_missing"
                if not artifact_path.exists()
                else "artifact_invalid"
            )
            self.outcomes[run_id] = CandidateOutcome(
                candidate=run_id,
                code=code,
                run_result=run_result,
                status=status,
                issues=(str(exc),),
            )
            return None
        result = self._pipeline.verify(artifact)
        if (
            result.status is not VerificationStatus.VERIFIED
            or result.evidence is None
            or result.record is None
        ):
            status = (
                "artifact_invalid"
                if result.status is VerificationStatus.INVALID_ARTIFACT
                else "verification_failed"
            )
            self.outcomes[run_id] = CandidateOutcome(
                candidate=run_id,
                code=code,
                run_result=run_result,
                status=status,
                issues=tuple(result.issues),
            )
            return None
        candidate = (
            Candidate.draft(code)
            if parent is None
            else Candidate.improve(parent.candidate, code)
        )
        record = replace(
            result.record,
            candidate_id=candidate.id,
            candidate_sha256=sha256_text(code),
        )
        verified = VerifiedCandidate(
            candidate=candidate, artifact=artifact, record=record
        )
        self.outcomes[run_id] = CandidateOutcome(
            candidate=run_id,
            code=code,
            run_result=run_result,
            status="verified",
            artifact_sha256=record.artifact_sha256,
            evidence={
                observation.name: observation.value
                for observation in result.evidence
            },
            core_candidate_id=record.candidate_id,
        )
        return verified, record


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
