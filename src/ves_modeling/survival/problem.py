"""Survival VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.survival.context import SurvivalVerificationContext
from ves_modeling.survival.data_contract import (
    load_hidden_outcomes,
    validate_survival_data,
)
from ves_modeling.survival.verifier import SurvivalVerifier

verifier = SurvivalVerifier()


def build_survival_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "survival",
    time_column: str = "time",
    event_column: str = "event",
    id_column: str | None = None,
    row_order: str = "input",
    output_kind: str = "risk_score",
) -> VerifiedProblem:
    """Assemble the survival VerifiedProblem."""
    data_contract = validate_survival_data(
        public_dir,
        time_column=time_column,
        event_column=event_column,
        id_column=id_column,
        row_order=row_order,
        output_kind=output_kind,
    )
    hidden_times, hidden_events = load_hidden_outcomes(
        host_dir, data_contract
    )

    def make_context() -> SurvivalVerificationContext:
        return SurvivalVerificationContext(
            hidden_times,
            hidden_events,
            dataset_name=dataset_name,
            expected_count=int(hidden_times.size),
            output_kind=output_kind,
            id_column=id_column,
            prediction_ids=(
                data_contract.test_ids if row_order == "id" else None
            ),
            row_order=row_order,
        )

    objectives = [
        ObjectiveSpec(
            observation="c_index", direction=Direction.MAXIMIZE
        )
    ]
    gates = [
        Gate(name="c_index_finite", observation="c_index", finite=True)
    ]
    if output_kind == "time":
        objectives.append(
            ObjectiveSpec(observation="mae", direction=Direction.MINIMIZE)
        )
        gates.append(Gate(name="mae_finite", observation="mae", finite=True))

    contract = ArtifactContract(
        filename="predictions.json",
        media_type="application/json",
        required_fields=("predictions",),
    )

    return VerifiedProblem(
        contract=contract,
        context_factory=make_context,
        verifier=verifier,
        judge_spec=JudgeSpec(
            objectives=tuple(objectives), gates=tuple(gates)
        ),
        name=f"survival:{dataset_name}",
        problem_ref="ves_modeling.survival.problem:build_survival_problem",
        verifier_module="ves_modeling.survival.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.survival.problem:context_factory",
    )


def context_factory() -> SurvivalVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` and ``VES_MODELING_HOST_DIR`` (required),
    ``VES_MODELING_DATASET``, ``VES_MODELING_TIME_COLUMN``,
    ``VES_MODELING_EVENT_COLUMN``, ``VES_MODELING_ID_COLUMN``,
    ``VES_MODELING_ROW_ORDER``, ``VES_MODELING_OUTPUT_KIND``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not public_dir or not host_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR and VES_MODELING_HOST_DIR must be set "
            "to replay a survival record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "survival")
    time_column = os.environ.get("VES_MODELING_TIME_COLUMN", "time")
    event_column = os.environ.get("VES_MODELING_EVENT_COLUMN", "event")
    id_column = os.environ.get("VES_MODELING_ID_COLUMN") or None
    row_order = os.environ.get("VES_MODELING_ROW_ORDER", "input")
    output_kind = os.environ.get("VES_MODELING_OUTPUT_KIND", "risk_score")
    data_contract = validate_survival_data(
        Path(public_dir),
        time_column=time_column,
        event_column=event_column,
        id_column=id_column,
        row_order=row_order,
        output_kind=output_kind,
    )
    hidden_times, hidden_events = load_hidden_outcomes(
        Path(host_dir), data_contract
    )
    return SurvivalVerificationContext(
        hidden_times,
        hidden_events,
        dataset_name=dataset_name,
        expected_count=int(hidden_times.size),
        output_kind=output_kind,
        id_column=id_column,
        prediction_ids=(
            data_contract.test_ids if row_order == "id" else None
        ),
        row_order=row_order,
    )
