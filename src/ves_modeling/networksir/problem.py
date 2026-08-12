"""Network-SIR VerifiedProblem assembly (contract + context + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.networksir.context import NetworkSirVerificationContext
from ves_modeling.networksir.data_contract import validate_networksir_data
from ves_modeling.networksir.verifier import (
    NetworkSirVerifier,
    reference_networksir_value,
)

verifier = NetworkSirVerifier()


def build_networksir_problem(
    public_dir: Path,
    *,
    dataset_name: str = "networksir",
) -> VerifiedProblem:
    """Assemble the network-SIR VerifiedProblem."""
    contract = validate_networksir_data(public_dir)
    reference = reference_networksir_value(contract)

    def make_context() -> NetworkSirVerificationContext:
        return NetworkSirVerificationContext(
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
        name=f"networksir:{dataset_name}",
        problem_ref="ves_modeling.networksir.problem:build_networksir_problem",
        verifier_module="ves_modeling.networksir.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.networksir.problem:context_factory",
    )


def context_factory() -> NetworkSirVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required), ``VES_MODELING_DATASET``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay a network-SIR "
            "record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "networksir")
    contract = validate_networksir_data(Path(public_dir))
    reference = reference_networksir_value(contract)
    return NetworkSirVerificationContext(
        contract, reference, dataset_name=dataset_name
    )
