"""Optimization VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.optimization.context import OptimizationVerificationContext
from ves_modeling.optimization.data_contract import validate_optimization_data
from ves_modeling.optimization.verifier import OptimizationVerifier

verifier = OptimizationVerifier()


def build_optimization_problem(
    public_dir: Path,
    *,
    dataset_name: str = "optimization",
    tolerance: float = 1e-6,
) -> VerifiedProblem:
    """Assemble the optimization VerifiedProblem.

    ``problem.json`` is the complete (public) instance, so no host directory
    exists.  Feasibility gates use the documented tolerance; the objective is
    ranked in the problem sense.  Global optimality is never claimed.
    """
    contract = validate_optimization_data(
        public_dir, tolerance=tolerance
    )
    objective_direction = (
        Direction.MINIMIZE
        if contract.sense == "minimize"
        else Direction.MAXIMIZE
    )

    def make_context() -> OptimizationVerificationContext:
        return OptimizationVerificationContext(
            contract, dataset_name=dataset_name
        )

    artifact_contract = ArtifactContract(
        filename="solution.json",
        media_type="application/json",
        required_fields=("variables",),
    )

    return VerifiedProblem(
        contract=artifact_contract,
        context_factory=make_context,
        verifier=verifier,
        judge_spec=JudgeSpec(
            objectives=(
                ObjectiveSpec(
                    observation="objective",
                    direction=objective_direction,
                ),
            ),
            gates=(
                Gate(
                    name="objective_finite",
                    observation="objective",
                    finite=True,
                ),
                Gate(
                    name="bound_feasible",
                    observation="max_bound_violation",
                    maximum=contract.tolerance,
                ),
                Gate(
                    name="constraint_feasible",
                    observation="max_constraint_violation",
                    maximum=contract.tolerance,
                ),
                Gate(
                    name="integrality_feasible",
                    observation="integrality_violation",
                    maximum=contract.tolerance,
                ),
            ),
        ),
        name=f"optimization:{dataset_name}",
        problem_ref=(
            "ves_modeling.optimization.problem:build_optimization_problem"
        ),
        verifier_module="ves_modeling.optimization.problem",
        verifier_attr="verifier",
        context_factory_ref=(
            "ves_modeling.optimization.problem:context_factory"
        ),
    )


def context_factory() -> OptimizationVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required, contains problem.json),
    ``VES_MODELING_DATASET``, ``VES_MODELING_TOLERANCE``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay an "
            "optimization record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "optimization")
    tolerance = float(os.environ.get("VES_MODELING_TOLERANCE", "1e-6"))
    contract = validate_optimization_data(
        Path(public_dir), tolerance=tolerance
    )
    return OptimizationVerificationContext(
        contract, dataset_name=dataset_name
    )
