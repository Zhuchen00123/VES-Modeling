"""Game VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.game.context import GameVerificationContext
from ves_modeling.game.data_contract import validate_game_data
from ves_modeling.game.verifier import GameVerifier

verifier = GameVerifier()


def build_game_problem(
    public_dir: Path,
    *,
    dataset_name: str = "game",
) -> VerifiedProblem:
    """Assemble the LQ game VerifiedProblem."""
    contract = validate_game_data(public_dir)

    def make_context() -> GameVerificationContext:
        return GameVerificationContext(contract, dataset_name=dataset_name)

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
        name=f"game:{dataset_name}",
        problem_ref="ves_modeling.game.problem:build_game_problem",
        verifier_module="ves_modeling.game.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.game.problem:context_factory",
    )


def context_factory() -> GameVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required), ``VES_MODELING_DATASET``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay a game record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "game")
    contract = validate_game_data(Path(public_dir))
    return GameVerificationContext(contract, dataset_name=dataset_name)
