"""Assignment/TSP VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.assignment.context import AssignVerificationContext
from ves_modeling.assignment.data_contract import validate_assign_data
from ves_modeling.assignment.verifier import AssignVerifier

verifier = AssignVerifier()


def build_assignment_problem(
    public_dir: Path,
    *,
    dataset_name: str = "assignment",
) -> VerifiedProblem:
    """Assemble the assignment/TSP VerifiedProblem."""
    contract = validate_assign_data(public_dir)

    def make_context() -> AssignVerificationContext:
        return AssignVerificationContext(contract, dataset_name=dataset_name)

    artifact_contract = ArtifactContract(
        filename="solution.json",
        media_type="application/json",
        required_fields=(),
    )

    return VerifiedProblem(
        contract=artifact_contract,
        context_factory=make_context,
        verifier=verifier,
        judge_spec=JudgeSpec(
            objectives=(
                ObjectiveSpec(
                    observation="total_cost", direction=Direction.MINIMIZE
                ),
            ),
            gates=(
                Gate(
                    name="total_cost_finite",
                    observation="total_cost",
                    finite=True,
                ),
            ),
        ),
        name=f"assignment:{dataset_name}",
        problem_ref=(
            "ves_modeling.assignment.problem:build_assignment_problem"
        ),
        verifier_module="ves_modeling.assignment.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.assignment.problem:context_factory",
    )


def context_factory() -> AssignVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required, contains problem.json),
    ``VES_MODELING_DATASET``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay an "
            "assignment/TSP record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "assignment")
    contract = validate_assign_data(Path(public_dir))
    return AssignVerificationContext(contract, dataset_name=dataset_name)
