"""Stable Multi-objective search API (R16)."""

from __future__ import annotations

import datetime
import json
import platform
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ves.search import GreedyTop1Policy
from ves.search_engine import SearchEngine

from ves_modeling.llm import OpenAICompatibleClient
from ves_modeling.multiobjective.data_contract import validate_moo_data
from ves_modeling.multiobjective.diagnostics import sha256_text, write_run_json
from ves_modeling.multiobjective.generator import (
    LlmClient,
    LLMMooGenerator,
    MockMooGenerator,
)
from ves_modeling.multiobjective.problem import build_multiobjective_problem
from ves_modeling.multiobjective.schema import API_SCHEMA_VERSION
from ves_modeling.regression.provenance import (
    file_hashes,
    package_versions,
    sanitize_provider,
)
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    DockerRunnerConfig,
    LocalRegressionRunner,
)

GeneratorKind = Literal["mock", "llm"]

_REPO_FIXTURES = (
    Path(__file__).resolve().parents[3] / "fixtures" / "multiobjective"
)


def _observation(evidence: Any, name: str) -> float | None:
    for item in evidence or ():
        if item.name == name:
            return item.value
    return None


@dataclass(frozen=True)
class MooSearchResult:
    """Host-verified bi-objective search outcome."""

    run_id: str
    dataset_name: str
    generator: str
    status: str
    drafts: int
    improves: int
    best_code: str | None
    best_candidate_id: str | None
    best_evidence: Any | None
    best_hypervolume: float | None
    best_non_dominated_count: float | None
    best_feasible_count: float | None
    best_total_count: float | None
    tolerance: float
    rejected: int
    run_dir: Path
    records: tuple[Any, ...]
    candidates: tuple[dict[str, Any], ...] = ()
    data_contract: dict[str, Any] | None = None

    def to_summary(self) -> dict[str, Any]:
        """JSON-serializable summary (no Core objects, no absolute paths)."""
        return {
            "schema_version": API_SCHEMA_VERSION,
            "run_id": self.run_id,
            "task": "multiobjective",
            "dataset": self.dataset_name,
            "generator": self.generator,
            "status": self.status,
            "drafts": self.drafts,
            "improves": self.improves,
            "best_candidate_id": self.best_candidate_id,
            "best_hypervolume": self.best_hypervolume,
            "best_non_dominated_count": self.best_non_dominated_count,
            "best_feasible_count": self.best_feasible_count,
            "best_total_count": self.best_total_count,
            "tolerance": self.tolerance,
            "rejected": self.rejected,
            "candidates": [dict(item) for item in self.candidates],
            "best_solution": "best_solution.py",
            "summary": "summary.json",
            "provenance": "provenance.json",
            "data_contract": self.data_contract,
        }


_SENSITIVE_KEY_PARTS = ("api_key", "token", "password", "secret")


def _validate_split_metadata(split_metadata: dict[str, Any] | None) -> dict:
    """Fail fast and return canonical JSON for split metadata."""
    if split_metadata is None:
        return {}
    if not isinstance(split_metadata, dict):
        raise ValueError("split_metadata must be a dict")
    try:
        encoded = json.dumps(split_metadata, allow_nan=False)
        canonical = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"split_metadata must be JSON-serializable: {exc}"
        ) from exc

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            items = value.items()
        elif isinstance(value, list):
            items = ((index, item) for index, item in enumerate(value))
        else:
            return
        for key, child in items:
            lowered = str(key).lower()
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                raise ValueError(
                    f"split_metadata must not contain sensitive key: {key!r}"
                )
            _walk(child)

    _walk(canonical)
    return canonical


def _preflight_public_files(public_dir: Path) -> None:
    file_hashes(public_dir, ("problem.json",), strict=True)


def _default_fixtures() -> Path:
    if _REPO_FIXTURES.is_dir():
        return _REPO_FIXTURES
    return Path.cwd() / "fixtures" / "multiobjective"


def run_multiobjective_search(
    public_dir: Path,
    *,
    drafts: int = 2,
    improves: int = 3,
    workspace: Path | None = None,
    generator: GeneratorKind = "mock",
    dataset_name: str = "multiobjective",
    fixture_dir: Path | None = None,
    fallback_code: str | None = None,
    client: LlmClient | None = None,
    image: str = "ves-modeling-runner:0.1",
    image_digest: str | None = None,
    timeout_seconds: float = 900.0,
    split_metadata: dict[str, Any] | None = None,
    tolerance: float = 1e-6,
) -> MooSearchResult:
    """Search a bi-objective problem and return the host-verified best."""
    public_dir = Path(public_dir)
    split_metadata = _validate_split_metadata(split_metadata)
    workspace = Path(workspace or Path.cwd() / "runs")
    workspace.mkdir(parents=True, exist_ok=True)
    fixture_dir = Path(fixture_dir) if fixture_dir else _default_fixtures()
    _preflight_public_files(public_dir)
    contract = validate_moo_data(public_dir, tolerance=tolerance)
    problem = build_multiobjective_problem(
        public_dir, dataset_name=dataset_name, tolerance=tolerance
    )

    run_id = uuid.uuid4().hex[:12]
    run_dir = workspace / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = run_dir / "candidates"

    if generator == "mock":
        candidate = MockMooGenerator(fixture_dir)
        runner = LocalRegressionRunner(
            workspace=candidate_root, data_dir=public_dir, run_layout="flat"
        )
        runner_info: dict[str, Any] = {
            "kind": "local",
            "timeout_seconds": runner.timeout_seconds,
        }
    elif generator == "llm":
        if fallback_code is None:
            fallback_path = fixture_dir / "scalarization_linprog.py"
            if not fallback_path.is_file():
                raise FileNotFoundError(
                    f"fallback fixture missing: {fallback_path}"
                )
            fallback_code = fallback_path.read_text(encoding="utf-8")
        llm_client = client or OpenAICompatibleClient()
        candidate = LLMMooGenerator(
            llm_client,
            fallback_code=fallback_code,
            variable_names=contract.variable_names,
            n_constraints=contract.n_constraints,
        )
        docker_config = DockerRunnerConfig(
            workspace=candidate_root,
            data_dir=public_dir,
            image=image,
            image_digest=image_digest,
            timeout_seconds=timeout_seconds,
            run_layout="flat",
            public_files=("problem.json",),
        )
        runner = DockerRegressionRunner(docker_config)
        runner_info = {
            "kind": "docker",
            "image": docker_config.image,
            "image_digest": docker_config.image_digest,
            "image_digest_status": (
                docker_config.image_digest and "configured" or "unresolved"
            ),
            "image_digest_error": None,
            "timeout_seconds": docker_config.timeout_seconds,
            "memory": docker_config.memory,
            "cpus": docker_config.cpus,
            "pids_limit": docker_config.pids_limit,
            "tmpfs_size": docker_config.tmpfs_size,
            "public_files": list(docker_config.public_files),
        }
    else:
        raise ValueError(f"unknown generator: {generator!r}")

    engine = SearchEngine(
        problem=problem,
        generator=candidate,
        runner=runner,
        anchor_policy=GreedyTop1Policy(),
        drafts=drafts,
        improves=improves,
    )
    result = engine.search()

    if generator == "llm":
        runner_info["image_digest"] = runner.effective_image_digest
        runner_info["image_digest_status"] = runner.image_digest_status
        runner_info["image_digest_error"] = runner.image_digest_error

    best_evidence = result.best_evidence
    best_feasible_count = (
        _observation(best_evidence, "feasible_count") if best_evidence else None
    )
    status = (
        "verified"
        if result.best_code is not None
        and best_feasible_count is not None
        and best_feasible_count >= 1.0
        else "no_verified"
    )
    best_hypervolume = (
        _observation(best_evidence, "hypervolume") if best_evidence else None
    )
    best_non_dominated = (
        _observation(best_evidence, "non_dominated_count")
        if best_evidence
        else None
    )
    best_total_count = (
        _observation(best_evidence, "total_count") if best_evidence else None
    )
    best_candidate_id = (
        result.best_record.candidate_id if result.best_record else None
    )

    (run_dir / "best_solution.py").write_text(
        result.best_code or "", encoding="utf-8"
    )
    attempts = tuple(sorted(result.attempts, key=lambda item: item.attempt_id))
    for outcome in attempts:
        attempt = outcome.attempt_id
        run_root = candidate_root / attempt
        run_root.mkdir(parents=True, exist_ok=True)
        for log_name in ("stdout.log", "stderr.log"):
            (run_root / log_name).touch(exist_ok=True)
        evidence = (
            {
                observation.name: observation.value
                for observation in outcome.record.evidence
            }
            if outcome.record is not None
            else None
        )
        write_run_json(
            run_root,
            candidate=attempt,
            status=outcome.status.value,
            code_sha256=outcome.candidate_sha256,
            artifact_sha256=outcome.artifact_sha256,
            evidence=evidence,
            issues=outcome.issues,
            search_id=run_id,
        )

    result_object = MooSearchResult(
        run_id=run_id,
        dataset_name=dataset_name,
        generator=generator,
        status=status,
        drafts=drafts,
        improves=improves,
        best_code=result.best_code,
        best_candidate_id=best_candidate_id,
        best_evidence=best_evidence,
        best_hypervolume=best_hypervolume,
        best_non_dominated_count=best_non_dominated,
        best_feasible_count=best_feasible_count,
        best_total_count=best_total_count,
        tolerance=tolerance,
        rejected=result.rejected,
        run_dir=run_dir,
        records=tuple(result.records),
        candidates=tuple(
            {"candidate": outcome.attempt_id, "status": outcome.status.value}
            for outcome in attempts
        ),
        data_contract=contract.to_dict(),
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            result_object.to_summary(), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "schema_version": API_SCHEMA_VERSION,
                "generator": generator,
                "drafts": drafts,
                "improves": improves,
                "runner": runner_info["kind"],
                "image": runner_info.get("image"),
                "image_digest": runner_info.get("image_digest"),
                "image_digest_status": runner_info.get("image_digest_status"),
                "image_digest_error": runner_info.get("image_digest_error"),
                "timeout_seconds": runner_info.get("timeout_seconds"),
                "tolerance": tolerance,
                "data_contract": contract.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    provider: dict[str, Any] | None = None
    if generator == "llm":
        provider = sanitize_provider(
            getattr(llm_client, "base_url", None),
            getattr(llm_client, "model", None),
        )
    best_attempt = next(
        (
            outcome.attempt_id
            for outcome in attempts
            if outcome.status.value == "verified"
            and outcome.record is not None
            and outcome.record.candidate_id == best_candidate_id
        ),
        None,
    )
    provenance = {
        "schema_version": API_SCHEMA_VERSION,
        "run_id": run_id,
        "dataset": dataset_name,
        "generator": generator,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "inputs": {
            "public": file_hashes(public_dir, ("problem.json",)),
        },
        "best": (
            {
                "candidate": best_attempt,
                "candidate_id": best_candidate_id,
                "code_sha256": sha256_text(result.best_code),
                "artifact_sha256": (
                    result.best_record.artifact_sha256
                    if result.best_record is not None
                    else None
                ),
            }
            if result.best_code is not None
            else None
        ),
        "versions": package_versions(),
        "provider": provider,
        "runner": runner_info,
        "split_metadata": split_metadata or {},
        "data_contract": contract.to_dict(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result_object
