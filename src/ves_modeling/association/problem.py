"""Association VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.association.context import AssociationVerificationContext
from ves_modeling.association.data_contract import (
    load_hidden_transactions,
    validate_association_data,
)
from ves_modeling.association.verifier import AssociationVerifier

verifier = AssociationVerifier()


def build_association_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "association",
    lift_cap: float = 1e6,
    transaction_id_column: str = "transaction_id",
    item_column: str = "item",
) -> VerifiedProblem:
    """Assemble the association rule VerifiedProblem."""
    data_contract = validate_association_data(
        public_dir,
        transaction_id_column=transaction_id_column,
        item_column=item_column,
    )
    hidden = load_hidden_transactions(host_dir, data_contract)

    def make_context() -> AssociationVerificationContext:
        return AssociationVerificationContext(
            hidden,
            data_contract,
            dataset_name=dataset_name,
            lift_cap=lift_cap,
        )

    contract = ArtifactContract(
        filename="rules.json",
        media_type="application/json",
        required_fields=("rules",),
    )

    return VerifiedProblem(
        contract=contract,
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
                    name="evaluable_rules",
                    observation="evaluable_rule_count",
                    minimum=1.0,
                ),
            ),
        ),
        name=f"association:{dataset_name}",
        problem_ref=(
            "ves_modeling.association.problem:build_association_problem"
        ),
        verifier_module="ves_modeling.association.problem",
        verifier_attr="verifier",
        context_factory_ref=(
            "ves_modeling.association.problem:context_factory"
        ),
    )


def context_factory() -> AssociationVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` and ``VES_MODELING_HOST_DIR`` (required),
    ``VES_MODELING_DATASET``, ``VES_MODELING_LIFT_CAP``,
    ``VES_MODELING_TRANSACTION_ID_COLUMN``, ``VES_MODELING_ITEM_COLUMN``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not public_dir or not host_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR and VES_MODELING_HOST_DIR must be set "
            "to replay an association record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "association")
    lift_cap = float(os.environ.get("VES_MODELING_LIFT_CAP", "1e6"))
    transaction_id_column = os.environ.get(
        "VES_MODELING_TRANSACTION_ID_COLUMN", "transaction_id"
    )
    item_column = os.environ.get("VES_MODELING_ITEM_COLUMN", "item")
    data_contract = validate_association_data(
        Path(public_dir),
        transaction_id_column=transaction_id_column,
        item_column=item_column,
    )
    hidden = load_hidden_transactions(Path(host_dir), data_contract)
    return AssociationVerificationContext(
        hidden,
        data_contract,
        dataset_name=dataset_name,
        lift_cap=lift_cap,
    )
