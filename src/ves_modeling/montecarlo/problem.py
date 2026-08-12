"""Monte Carlo VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.montecarlo.context import MonteCarloVerificationContext
from ves_modeling.montecarlo.data_contract import (
    compute_reference,
    validate_montecarlo_data,
)
from ves_modeling.montecarlo.verifier import MonteCarloVerifier

verifier = MonteCarloVerifier()


def build_montecarlo_problem(
    public_dir: Path,
    *,
    dataset_name: str = "montecarlo",
) -> VerifiedProblem:
    """Assemble the Monte Carlo VerifiedProblem.

    ``problem.json`` is the complete (public) instance; the analytic
    reference is host-computed and held only inside the context.
    """
    contract = validate_montecarlo_data(public_dir)
    reference = compute_reference(contract)

    def make_context() -> MonteCarloVerificationContext:
        return MonteCarloVerificationContext(
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
        name=f"montecarlo:{dataset_name}",
        problem_ref="ves_modeling.montecarlo.problem:build_montecarlo_problem",
        verifier_module="ves_modeling.montecarlo.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.montecarlo.problem:context_factory",
    )


def context_factory() -> MonteCarloVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required, contains problem.json),
    ``VES_MODELING_DATASET``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay a Monte Carlo "
            "record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "montecarlo")
    contract = validate_montecarlo_data(Path(public_dir))
    reference = compute_reference(contract)
    return MonteCarloVerificationContext(
        reference, contract, dataset_name=dataset_name
    )
