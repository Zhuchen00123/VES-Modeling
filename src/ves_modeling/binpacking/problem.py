"""Bin packing VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.binpacking.context import BinVerificationContext
from ves_modeling.binpacking.data_contract import validate_bin_data
from ves_modeling.binpacking.verifier import BinVerifier

verifier = BinVerifier()


def build_binpacking_problem(
    public_dir: Path,
    *,
    dataset_name: str = "binpacking",
) -> VerifiedProblem:
    """Assemble the bin packing VerifiedProblem."""
    contract = validate_bin_data(public_dir)

    def make_context() -> BinVerificationContext:
        return BinVerificationContext(contract, dataset_name=dataset_name)

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
                    observation="bin_count", direction=Direction.MINIMIZE
                ),
            ),
            gates=(
                Gate(
                    name="bin_count_finite",
                    observation="bin_count",
                    finite=True,
                ),
                Gate(
                    name="capacity_feasible",
                    observation="capacity_violation",
                    maximum=0.0,
                ),
            ),
        ),
        name=f"binpacking:{dataset_name}",
        problem_ref="ves_modeling.binpacking.problem:build_binpacking_problem",
        verifier_module="ves_modeling.binpacking.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.binpacking.problem:context_factory",
    )


def context_factory() -> BinVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required, contains problem.json),
    ``VES_MODELING_DATASET``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay a bin packing "
            "record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "binpacking")
    contract = validate_bin_data(Path(public_dir))
    return BinVerificationContext(contract, dataset_name=dataset_name)
