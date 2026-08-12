"""Anomaly VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.anomaly.context import AnomalyVerificationContext
from ves_modeling.anomaly.data_contract import (
    OUTPUT_MODES,
    load_host_labels,
    validate_anomaly_data,
)
from ves_modeling.anomaly.verifier import AnomalyVerifier

verifier = AnomalyVerifier()


def build_anomaly_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "anomaly",
    output_mode: str = "score",
    label_column: str = "label",
) -> VerifiedProblem:
    """Assemble the anomaly VerifiedProblem for the requested artifact mode."""
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of {OUTPUT_MODES}")
    data_contract = validate_anomaly_data(
        public_dir, label_column=label_column
    )
    hidden = load_host_labels(host_dir, data_contract)

    def make_context() -> AnomalyVerificationContext:
        return AnomalyVerificationContext(
            hidden,
            dataset_name=dataset_name,
            expected_count=int(hidden.size),
            output_mode=output_mode,
        )

    if output_mode == "score":
        objectives = (
            ObjectiveSpec(
                observation="auroc", direction=Direction.MAXIMIZE
            ),
            ObjectiveSpec(
                observation="average_precision",
                direction=Direction.MAXIMIZE,
            ),
        )
        gates = (
            Gate(name="auroc_finite", observation="auroc", finite=True),
            Gate(
                name="average_precision_finite",
                observation="average_precision",
                finite=True,
            ),
        )
        required_fields = ("scores",)
    else:
        objectives = (
            ObjectiveSpec(observation="f1", direction=Direction.MAXIMIZE),
            ObjectiveSpec(
                observation="balanced_accuracy",
                direction=Direction.MAXIMIZE,
            ),
        )
        gates = (
            Gate(name="f1_finite", observation="f1", finite=True),
            Gate(
                name="balanced_accuracy_finite",
                observation="balanced_accuracy",
                finite=True,
            ),
        )
        required_fields = ("labels",)

    contract = ArtifactContract(
        filename="predictions.json",
        media_type="application/json",
        required_fields=required_fields,
    )

    return VerifiedProblem(
        contract=contract,
        context_factory=make_context,
        verifier=verifier,
        judge_spec=JudgeSpec(objectives=objectives, gates=gates),
        name=f"anomaly:{dataset_name}",
        problem_ref="ves_modeling.anomaly.problem:build_anomaly_problem",
        verifier_module="ves_modeling.anomaly.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.anomaly.problem:context_factory",
    )


def context_factory() -> AnomalyVerificationContext:
    """Module-level factory used by ``ves replay`` (requires env config).

    Env: ``VES_MODELING_HOST_DIR`` (required), ``VES_MODELING_DATASET``,
    ``VES_MODELING_LABEL_COLUMN``, ``VES_MODELING_OUTPUT_MODE``.
    """
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not host_dir:
        raise RuntimeError(
            "VES_MODELING_HOST_DIR must be set to replay an anomaly record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "anomaly")
    label_column = os.environ.get("VES_MODELING_LABEL_COLUMN", "label")
    output_mode = os.environ.get("VES_MODELING_OUTPUT_MODE", "score")
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay an anomaly record"
        )
    data_contract = validate_anomaly_data(
        Path(public_dir), label_column=label_column
    )
    hidden = load_host_labels(Path(host_dir), data_contract)
    return AnomalyVerificationContext(
        hidden,
        dataset_name=dataset_name,
        expected_count=int(hidden.size),
        output_mode=output_mode,
    )
