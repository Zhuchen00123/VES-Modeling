"""LQR VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.lqr.context import LqrVerificationContext
from ves_modeling.lqr.data_contract import validate_lqr_data
from ves_modeling.lqr.verifier import LqrVerifier

verifier = LqrVerifier()


def build_lqr_problem(
    public_dir: Path,
    *,
    dataset_name: str = "lqr",
) -> VerifiedProblem:
    """Assemble the LQR VerifiedProblem."""
    contract = validate_lqr_data(public_dir)

    def make_context() -> LqrVerificationContext:
        return LqrVerificationContext(contract, dataset_name=dataset_name)

    artifact_contract = ArtifactContract(
        filename="solution.json",
        media_type="application/json",
        required_fields=("control",),
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
        name=f"lqr:{dataset_name}",
        problem_ref="ves_modeling.lqr.problem:build_lqr_problem",
        verifier_module="ves_modeling.lqr.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.lqr.problem:context_factory",
    )


def context_factory() -> LqrVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required), ``VES_MODELING_DATASET``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay an LQR record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "lqr")
    contract = validate_lqr_data(Path(public_dir))
    return LqrVerificationContext(contract, dataset_name=dataset_name)
