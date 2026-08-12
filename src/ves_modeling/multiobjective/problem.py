"""Multi-objective VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.multiobjective.context import MooVerificationContext
from ves_modeling.multiobjective.data_contract import validate_moo_data
from ves_modeling.multiobjective.verifier import MooVerifier

verifier = MooVerifier()


def build_multiobjective_problem(
    public_dir: Path,
    *,
    dataset_name: str = "multiobjective",
    tolerance: float = 1e-6,
) -> VerifiedProblem:
    """Assemble the bi-objective VerifiedProblem.

    ``problem.json`` is the complete (public) instance.  Hypervolume is a
    relative quality measure of the delivered solution set; global optimality
    is never claimed.
    """
    contract = validate_moo_data(public_dir, tolerance=tolerance)

    def make_context() -> MooVerificationContext:
        return MooVerificationContext(contract, dataset_name=dataset_name)

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
                    observation="hypervolume",
                    direction=Direction.MAXIMIZE,
                ),
            ),
            gates=(
                Gate(
                    name="hypervolume_finite",
                    observation="hypervolume",
                    finite=True,
                ),
                Gate(
                    name="feasible_solutions",
                    observation="feasible_count",
                    minimum=1.0,
                ),
            ),
        ),
        name=f"multiobjective:{dataset_name}",
        problem_ref=(
            "ves_modeling.multiobjective.problem:"
            "build_multiobjective_problem"
        ),
        verifier_module="ves_modeling.multiobjective.problem",
        verifier_attr="verifier",
        context_factory_ref=(
            "ves_modeling.multiobjective.problem:context_factory"
        ),
    )


def context_factory() -> MooVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required, contains problem.json),
    ``VES_MODELING_DATASET``, ``VES_MODELING_TOLERANCE``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay a "
            "multi-objective record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "multiobjective")
    tolerance = float(os.environ.get("VES_MODELING_TOLERANCE", "1e-6"))
    contract = validate_moo_data(Path(public_dir), tolerance=tolerance)
    return MooVerificationContext(contract, dataset_name=dataset_name)
