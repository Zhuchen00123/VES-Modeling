"""SIR VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.sir.context import SirVerificationContext
from ves_modeling.sir.data_contract import validate_sir_data
from ves_modeling.sir.verifier import (
    SirVerifier,
    reference_sir_value,
)

verifier = SirVerifier()


def build_sir_problem(
    public_dir: Path,
    *,
    dataset_name: str = "sir",
) -> VerifiedProblem:
    """Assemble the SIR VerifiedProblem (reference is host-computed)."""
    contract = validate_sir_data(public_dir)
    reference = reference_sir_value(contract)

    def make_context() -> SirVerificationContext:
        return SirVerificationContext(
            contract, reference, dataset_name=dataset_name
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
        name=f"sir:{dataset_name}",
        problem_ref="ves_modeling.sir.problem:build_sir_problem",
        verifier_module="ves_modeling.sir.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.sir.problem:context_factory",
    )


def context_factory() -> SirVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required), ``VES_MODELING_DATASET``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay an SIR record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "sir")
    contract = validate_sir_data(Path(public_dir))
    reference = reference_sir_value(contract)
    return SirVerificationContext(
        contract, reference, dataset_name=dataset_name
    )
