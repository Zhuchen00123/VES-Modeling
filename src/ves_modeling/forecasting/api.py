"""Stable Forecasting search API (R8).

Mirrors the regression API shape: one concrete ``run_forecasting_search``
entry that returns a host-verified result and persists the whole search under
``workspace/<run_id>``.  No universal task abstractions are introduced.
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ves.search import GreedyTop1Policy
from ves.search_engine import SearchEngine

from ves_modeling.forecasting.data_contract import validate_forecasting_data
from ves_modeling.forecasting.diagnostics import sha256_text, write_run_json
from ves_modeling.forecasting.generator import (
    LlmClient,
    LLMForecastingGenerator,
    MockForecastingGenerator,
)
from ves_modeling.forecasting.problem import build_forecasting_problem
from ves_modeling.forecasting.schema import API_SCHEMA_VERSION
from ves_modeling.llm import OpenAICompatibleClient
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

_REPO_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "forecasting"


def _observation(evidence: Any, name: str) -> float | None:
    for item in evidence or ():
        if item.name == name:
            return item.value
    return None


@dataclass(frozen=True)
class ForecastingSearchResult:
    """Host-verified forecasting search outcome.

    ``best_rmse`` / ``best_mae`` / ``best_smape`` come from the host verifier
    evidence, never from candidate self-reports.  ``to_summary()`` provides a
    stable JSON-serializable view without VES Core objects or filesystem
    paths.
    """

    run_id: str
    dataset_name: str
    generator: str
    status: str
    drafts: int
    improves: int
    best_code: str | None
    best_candidate_id: str | None
    best_evidence: Any | None
    best_rmse: float | None
    best_mae: float | None
    best_smape: float | None
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
            "task": "forecasting",
            "dataset": self.dataset_name,
            "generator": self.generator,
            "status": self.status,
            "drafts": self.drafts,
            "improves": self.improves,
            "best_candidate_id": self.best_candidate_id,
            "best_rmse": self.best_rmse,
            "best_mae": self.best_mae,
            "best_smape": self.best_smape,
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
    """Fail fast before candidate execution when inputs are missing."""
    file_hashes(public_dir, ("train.csv", "test_features.csv"), strict=True)


def _default_fixtures() -> Path:
    if _REPO_FIXTURES.is_dir():
        return _REPO_FIXTURES
    return Path.cwd() / "fixtures" / "forecasting"


def run_forecasting_search(
    public_dir: Path,
    host_dir: Path,
    *,
    drafts: int = 2,
    improves: int = 3,
    workspace: Path | None = None,
    generator: GeneratorKind = "mock",
    dataset_name: str = "forecasting",
    labels: np.ndarray | None = None,
    fixture_dir: Path | None = None,
    fallback_code: str | None = None,
    client: LlmClient | None = None,
    image: str = "ves-modeling-runner:0.1",
    image_digest: str | None = None,
    timeout_seconds: float = 900.0,
    split_metadata: dict[str, Any] | None = None,
    time_column: str = "timestamp",
    series_id_column: str = "series_id",
    target_column: str = "target",
    frequency: str = "D",
    row_order: str = "key",
    method_family: str | None = None,
) -> ForecastingSearchResult:
    """Search a long-format forecasting task and return the host-verified best.

    ``public_dir`` must contain ``train.csv`` / ``test_features.csv``
    (candidate-visible); ``host_dir`` must contain ``hidden_test_labels.csv``
    (host-only, never exposed to candidates).  ``labels`` may be injected
    instead for tests (input mode only).

    ``generator="mock"`` uses trusted fixtures + the local subprocess runner
    (tests/fixtures only); ``generator="llm"`` uses the LLM generator plus the
    Docker runner (the only execution boundary for real candidates).
    """
    public_dir = Path(public_dir)
    host_dir = Path(host_dir)
    split_metadata = _validate_split_metadata(split_metadata)
    workspace = Path(workspace or Path.cwd() / "runs")
    workspace.mkdir(parents=True, exist_ok=True)
    fixture_dir = Path(fixture_dir) if fixture_dir else _default_fixtures()
    _preflight_public_files(public_dir)
    contract = validate_forecasting_data(
        public_dir,
        time_column=time_column,
        series_id_column=series_id_column,
        target_column=target_column,
        frequency=frequency,
        row_order=row_order,
    )
    # Host-side contract errors must not leave a run directory behind.
    problem = build_forecasting_problem(
        public_dir,
        host_dir,
        dataset_name=dataset_name,
        labels=labels,
        time_column=time_column,
        series_id_column=series_id_column,
        target_column=target_column,
        frequency=frequency,
        row_order=row_order,
    )

    run_id = uuid.uuid4().hex[:12]
    run_dir = workspace / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = run_dir / "candidates"

    if generator == "mock":
        candidate = MockForecastingGenerator(
            fixture_dir, method_family=method_family
        )
        runner = LocalRegressionRunner(
            workspace=candidate_root, data_dir=public_dir, run_layout="flat"
        )
        runner_info: dict[str, Any] = {
            "kind": "local",
            "timeout_seconds": runner.timeout_seconds,
        }
    elif generator == "llm":
        if fallback_code is None:
            fallback_path = fixture_dir / "linear_forecast.py"
            if not fallback_path.is_file():
                raise FileNotFoundError(
                    f"fallback fixture missing: {fallback_path}"
                )
            fallback_code = fallback_path.read_text(encoding="utf-8")
        llm_client = client or OpenAICompatibleClient()
        candidate = LLMForecastingGenerator(
            llm_client,
            fallback_code=fallback_code,
            time_column=time_column,
            series_id_column=series_id_column,
            target_column=target_column,
            frequency=frequency,
            horizon=contract.horizon,
            feature_columns=contract.feature_columns,
            row_order=row_order,
            method_family=method_family,
        )
        docker_config = DockerRunnerConfig(
            workspace=candidate_root,
            data_dir=public_dir,
            image=image,
            image_digest=image_digest,
            timeout_seconds=timeout_seconds,
            run_layout="flat",
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

    status = "verified" if result.best_code is not None else "no_verified"
    best_evidence = result.best_evidence
    best_rmse = (
        _observation(best_evidence, "rmse") if best_evidence else None
    )
    best_mae = (
        _observation(best_evidence, "mae") if best_evidence else None
    )
    best_smape = (
        _observation(best_evidence, "smape") if best_evidence else None
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

    result_object = ForecastingSearchResult(
        run_id=run_id,
        dataset_name=dataset_name,
        generator=generator,
        status=status,
        drafts=drafts,
        improves=improves,
        best_code=result.best_code,
        best_candidate_id=best_candidate_id,
        best_evidence=best_evidence,
        best_rmse=best_rmse,
        best_mae=best_mae,
        best_smape=best_smape,
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
            "public": file_hashes(
                public_dir, ("train.csv", "test_features.csv")
            ),
            "host": file_hashes(host_dir, ("hidden_test_labels.csv",)),
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
