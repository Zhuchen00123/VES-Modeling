"""Probabilistic VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.probabilistic.context import (
    ProbabilisticVerificationContext,
)
from ves_modeling.probabilistic.data_contract import (
    compute_reference,
    load_hidden_parameters,
    validate_probabilistic_data,
)
from ves_modeling.probabilistic.verifier import ProbabilisticVerifier

verifier = ProbabilisticVerifier()


def build_probabilistic_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "probabilistic",
    sample_column: str = "value",
) -> VerifiedProblem:
    """Assemble the probabilistic inference VerifiedProblem."""
    contract = validate_probabilistic_data(
        public_dir, sample_column=sample_column
    )
    parameters = load_hidden_parameters(host_dir, contract)
    reference = compute_reference(contract, parameters)

    def make_context() -> ProbabilisticVerificationContext:
        return ProbabilisticVerificationContext(
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
        name=f"probabilistic:{dataset_name}",
        problem_ref=(
            "ves_modeling.probabilistic.problem:"
            "build_probabilistic_problem"
        ),
        verifier_module="ves_modeling.probabilistic.problem",
        verifier_attr="verifier",
        context_factory_ref=(
            "ves_modeling.probabilistic.problem:context_factory"
        ),
    )


def context_factory() -> ProbabilisticVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` and ``VES_MODELING_HOST_DIR`` (required),
    ``VES_MODELING_DATASET``, ``VES_MODELING_SAMPLE_COLUMN``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not public_dir or not host_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR and VES_MODELING_HOST_DIR must be set "
            "to replay a probabilistic record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "probabilistic")
    sample_column = os.environ.get("VES_MODELING_SAMPLE_COLUMN", "value")
    contract = validate_probabilistic_data(
        Path(public_dir), sample_column=sample_column
    )
    parameters = load_hidden_parameters(Path(host_dir), contract)
    reference = compute_reference(contract, parameters)
    return ProbabilisticVerificationContext(
        reference, contract, dataset_name=dataset_name
    )
