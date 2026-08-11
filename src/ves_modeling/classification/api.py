"""Stable Classification search API (R9).

Mirrors the regression/forecasting API shape: one concrete
``run_classification_search`` entry that returns a host-verified result and
persists the whole search under ``workspace/<run_id>``.  No universal task
abstractions are introduced.
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ves.search import GreedyTop1Policy
from ves.search_engine import SearchEngine

from ves_modeling.classification.data_contract import (
    validate_classification_data,
)
from ves_modeling.classification.diagnostics import sha256_text, write_run_json
from ves_modeling.classification.generator import (
    LLMClassificationGenerator,
    LlmClient,
    MockClassificationGenerator,
)
from ves_modeling.classification.problem import build_classification_problem
from ves_modeling.classification.schema import API_SCHEMA_VERSION
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

_REPO_FIXTURES = (
    Path(__file__).resolve().parents[3] / "fixtures" / "classification"
)


def _observation(evidence: Any, name: str) -> float | None:
    for item in evidence or ():
        if item.name == name:
            return item.value
    return None


def _confusion_matrix(evidence: Any, n_classes: int) -> list[list[float]] | None:
    if evidence is None:
        return None
    values: dict[str, float] = {
        item.name: item.value for item in evidence
    }
    matrix = [
        [values.get(f"confusion_{i}_{j}", 0.0) for j in range(n_classes)]
        for i in range(n_classes)
    ]
    return matrix


@dataclass(frozen=True)
class ClassificationSearchResult:
    """Host-verified classification search outcome.

    ``best_*`` metrics come from the host verifier evidence, never from
    candidate self-reports.  ``classes`` is the host-fixed class order.
    ``to_summary()`` provides a stable JSON-serializable view without VES Core
    objects or filesystem paths.
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
    best_accuracy: float | None
    best_macro_f1: float | None
    best_log_loss: float | None
    best_auroc: float | None
    best_multiclass_brier: float | None
    best_calibration_ece: float | None
    best_confusion_matrix: list[list[float]] | None
    classes: tuple[Any, ...]
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
            "task": "classification",
            "dataset": self.dataset_name,
            "generator": self.generator,
            "status": self.status,
            "drafts": self.drafts,
            "improves": self.improves,
            "best_candidate_id": self.best_candidate_id,
            "best_accuracy": self.best_accuracy,
            "best_macro_f1": self.best_macro_f1,
            "best_log_loss": self.best_log_loss,
            "best_auroc": self.best_auroc,
            "best_multiclass_brier": self.best_multiclass_brier,
            "best_calibration_ece": self.best_calibration_ece,
            "best_confusion_matrix": self.best_confusion_matrix,
            "classes": list(self.classes),
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
    return Path.cwd() / "fixtures" / "classification"


def run_classification_search(
    public_dir: Path,
    host_dir: Path,
    *,
    drafts: int = 2,
    improves: int = 3,
    workspace: Path | None = None,
    generator: GeneratorKind = "mock",
    dataset_name: str = "classification",
    labels: np.ndarray | None = None,
    fixture_dir: Path | None = None,
    fallback_code: str | None = None,
    client: LlmClient | None = None,
    image: str = "ves-modeling-runner:0.1",
    image_digest: str | None = None,
    timeout_seconds: float = 900.0,
    split_metadata: dict[str, Any] | None = None,
    label_column: str = "target",
    id_column: str | None = None,
    row_order: str = "input",
    classes: Sequence[Any] | None = None,
) -> ClassificationSearchResult:
    """Search a classification task and return the host-verified best.

    ``public_dir`` must contain ``train.csv`` / ``test_features.csv``
    (candidate-visible); ``host_dir`` must contain ``hidden_test_labels.csv``
    (host-only, never exposed to candidates).  ``labels`` may be injected
    instead for tests (input mode only).

    ``classes`` optionally fixes the host class order (list/tuple, >=2
    unique); the default is first appearance in train.
    """
    public_dir = Path(public_dir)
    host_dir = Path(host_dir)
    split_metadata = _validate_split_metadata(split_metadata)
    workspace = Path(workspace or Path.cwd() / "runs")
    workspace.mkdir(parents=True, exist_ok=True)
    fixture_dir = Path(fixture_dir) if fixture_dir else _default_fixtures()
    _preflight_public_files(public_dir)
    contract = validate_classification_data(
        public_dir,
        label_column=label_column,
        id_column=id_column,
        row_order=row_order,
        classes=classes,
    )
    problem = build_classification_problem(
        public_dir,
        host_dir,
        dataset_name=dataset_name,
        labels=labels,
        label_column=label_column,
        id_column=id_column,
        row_order=row_order,
        classes=classes,
    )

    run_id = uuid.uuid4().hex[:12]
    run_dir = workspace / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = run_dir / "candidates"

    if generator == "mock":
        candidate = MockClassificationGenerator(fixture_dir)
        runner = LocalRegressionRunner(
            workspace=candidate_root, data_dir=public_dir, run_layout="flat"
        )
        runner_info: dict[str, Any] = {
            "kind": "local",
            "timeout_seconds": runner.timeout_seconds,
        }
    elif generator == "llm":
        if fallback_code is None:
            fallback_path = fixture_dir / "random_forest_balanced.py"
            if not fallback_path.is_file():
                raise FileNotFoundError(
                    f"fallback fixture missing: {fallback_path}"
                )
            fallback_code = fallback_path.read_text(encoding="utf-8")
        llm_client = client or OpenAICompatibleClient()
        candidate = LLMClassificationGenerator(
            llm_client,
            fallback_code=fallback_code,
            label_column=label_column,
            id_column=id_column,
            row_order=row_order,
            classes=contract.classes,
            n_features=len(contract.feature_columns),
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
    metric_names = (
        "accuracy",
        "macro_f1",
        "log_loss",
        "auroc",
        "multiclass_brier",
        "calibration_ece",
    )
    metric_values = {
        name: (_observation(best_evidence, name) if best_evidence else None)
        for name in metric_names
    }
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

    result_object = ClassificationSearchResult(
        run_id=run_id,
        dataset_name=dataset_name,
        generator=generator,
        status=status,
        drafts=drafts,
        improves=improves,
        best_code=result.best_code,
        best_candidate_id=best_candidate_id,
        best_evidence=best_evidence,
        best_accuracy=metric_values["accuracy"],
        best_macro_f1=metric_values["macro_f1"],
        best_log_loss=metric_values["log_loss"],
        best_auroc=metric_values["auroc"],
        best_multiclass_brier=metric_values["multiclass_brier"],
        best_calibration_ece=metric_values["calibration_ece"],
        best_confusion_matrix=_confusion_matrix(
            best_evidence, contract.n_classes
        ),
        classes=contract.classes,
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
