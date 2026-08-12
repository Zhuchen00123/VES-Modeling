"""Markov VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.markov.context import MarkovVerificationContext
from ves_modeling.markov.data_contract import (
    compute_reference,
    load_hidden_parameters,
    validate_markov_data,
)
from ves_modeling.markov.verifier import MarkovVerifier

verifier = MarkovVerifier()


def build_markov_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "markov",
    state_column: str = "state",
    sequence_id_column: str | None = "sequence_id",
) -> VerifiedProblem:
    """Assemble the Markov estimation VerifiedProblem."""
    contract = validate_markov_data(
        public_dir,
        state_column=state_column,
        sequence_id_column=sequence_id_column,
    )
    transition_matrix = load_hidden_parameters(host_dir, contract)
    reference = compute_reference(contract, transition_matrix)

    def make_context() -> MarkovVerificationContext:
        return MarkovVerificationContext(
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
        name=f"markov:{dataset_name}",
        problem_ref="ves_modeling.markov.problem:build_markov_problem",
        verifier_module="ves_modeling.markov.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.markov.problem:context_factory",
    )


def context_factory() -> MarkovVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` and ``VES_MODELING_HOST_DIR`` (required),
    ``VES_MODELING_DATASET``, ``VES_MODELING_STATE_COLUMN``,
    ``VES_MODELING_SEQUENCE_ID_COLUMN``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not public_dir or not host_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR and VES_MODELING_HOST_DIR must be set "
            "to replay a markov record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "markov")
    state_column = os.environ.get("VES_MODELING_STATE_COLUMN", "state")
    sequence_id_column = os.environ.get("VES_MODELING_SEQUENCE_ID_COLUMN")
    contract = validate_markov_data(
        Path(public_dir),
        state_column=state_column,
        sequence_id_column=sequence_id_column or "sequence_id",
    )
    transition_matrix = load_hidden_parameters(Path(host_dir), contract)
    reference = compute_reference(contract, transition_matrix)
    return MarkovVerificationContext(
        reference, contract, dataset_name=dataset_name
    )
