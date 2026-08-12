"""Cellular VerifiedProblem assembly (contract + context + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.cellular.context import CellularVerificationContext
from ves_modeling.cellular.data_contract import validate_cellular_data
from ves_modeling.cellular.verifier import (
    CellularVerifier,
    reference_ca_value,
)

verifier = CellularVerifier()


def build_cellular_problem(
    public_dir: Path,
    *,
    dataset_name: str = "cellular",
) -> VerifiedProblem:
    """Assemble the cellular-automaton VerifiedProblem."""
    contract = validate_cellular_data(public_dir)
    reference = reference_ca_value(contract)

    def make_context() -> CellularVerificationContext:
        return CellularVerificationContext(
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
        name=f"cellular:{dataset_name}",
        problem_ref="ves_modeling.cellular.problem:build_cellular_problem",
        verifier_module="ves_modeling.cellular.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.cellular.problem:context_factory",
    )


def context_factory() -> CellularVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required), ``VES_MODELING_DATASET``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay a cellular record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "cellular")
    contract = validate_cellular_data(Path(public_dir))
    reference = reference_ca_value(contract)
    return CellularVerificationContext(
        contract, reference, dataset_name=dataset_name
    )
