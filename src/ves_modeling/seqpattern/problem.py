"""Sequential-pattern VerifiedProblem assembly (contract + context + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.seqpattern.context import SeqPatternVerificationContext
from ves_modeling.seqpattern.data_contract import (
    load_hidden_sequences,
    validate_seqpattern_data,
)
from ves_modeling.seqpattern.verifier import SeqPatternVerifier

verifier = SeqPatternVerifier()


def build_seqpattern_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "seqpattern",
) -> VerifiedProblem:
    """Assemble the sequential-pattern VerifiedProblem."""
    contract = validate_seqpattern_data(public_dir)
    hidden = load_hidden_sequences(host_dir)

    def make_context() -> SeqPatternVerificationContext:
        return SeqPatternVerificationContext(
            hidden,
            event_set=contract.event_set,
            dataset_name=dataset_name,
        )

    artifact_contract = ArtifactContract(
        filename="patterns.json",
        media_type="application/json",
        required_fields=("patterns",),
    )

    return VerifiedProblem(
        contract=artifact_contract,
        context_factory=make_context,
        verifier=verifier,
        judge_spec=JudgeSpec(
            objectives=(
                ObjectiveSpec(
                    observation="mean_lift", direction=Direction.MAXIMIZE
                ),
                ObjectiveSpec(
                    observation="mean_confidence",
                    direction=Direction.MAXIMIZE,
                ),
            ),
            gates=(
                Gate(
                    name="mean_lift_finite",
                    observation="mean_lift",
                    finite=True,
                ),
                Gate(
                    name="mean_confidence_finite",
                    observation="mean_confidence",
                    finite=True,
                ),
                Gate(
                    name="at_least_one_evaluable",
                    observation="evaluable_pattern_count",
                    minimum=1.0,
                ),
            ),
        ),
        name=f"seqpattern:{dataset_name}",
        problem_ref=(
            "ves_modeling.seqpattern.problem:build_seqpattern_problem"
        ),
        verifier_module="ves_modeling.seqpattern.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.seqpattern.problem:context_factory",
    )


def context_factory() -> SeqPatternVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` and ``VES_MODELING_HOST_DIR`` (required),
    ``VES_MODELING_DATASET``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not public_dir or not host_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR and VES_MODELING_HOST_DIR must be set "
            "to replay a sequential-pattern record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "seqpattern")
    contract = validate_seqpattern_data(Path(public_dir))
    hidden = load_hidden_sequences(Path(host_dir))
    return SeqPatternVerificationContext(
        hidden,
        event_set=contract.event_set,
        dataset_name=dataset_name,
    )
