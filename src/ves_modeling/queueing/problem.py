"""Queueing VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.queueing.context import QueueingVerificationContext
from ves_modeling.queueing.data_contract import (
    compute_reference,
    validate_queueing_data,
)
from ves_modeling.queueing.verifier import QueueingVerifier

verifier = QueueingVerifier()


def build_queueing_problem(
    public_dir: Path,
    *,
    dataset_name: str = "queueing",
) -> VerifiedProblem:
    """Assemble the queueing VerifiedProblem."""
    contract = validate_queueing_data(public_dir)
    reference = compute_reference(contract)

    def make_context() -> QueueingVerificationContext:
        return QueueingVerificationContext(
            reference, contract, dataset_name=dataset_name
        )

    artifact_contract = ArtifactContract(
        filename="solution.json",
        media_type="application/json",
        required_fields=("estimate",),
    )

    return VerifiedProblem(
        contract=artifact_contract,
        context_factory=make_context,
        verifier=verifier,
        judge_spec=JudgeSpec(
            objectives=(
                ObjectiveSpec(
                    observation="relative_error",
                    direction=Direction.MINIMIZE,
                ),
                ObjectiveSpec(
                    observation="absolute_error",
                    direction=Direction.MINIMIZE,
                ),
            ),
            gates=(
                Gate(
                    name="relative_error_finite",
                    observation="relative_error",
                    finite=True,
                ),
                Gate(
                    name="absolute_error_finite",
                    observation="absolute_error",
                    finite=True,
                ),
            ),
        ),
        name=f"queueing:{dataset_name}",
        problem_ref="ves_modeling.queueing.problem:build_queueing_problem",
        verifier_module="ves_modeling.queueing.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.queueing.problem:context_factory",
    )


def context_factory() -> QueueingVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required, contains problem.json),
    ``VES_MODELING_DATASET``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay a queueing record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "queueing")
    contract = validate_queueing_data(Path(public_dir))
    reference = compute_reference(contract)
    return QueueingVerificationContext(
        reference, contract, dataset_name=dataset_name
    )
