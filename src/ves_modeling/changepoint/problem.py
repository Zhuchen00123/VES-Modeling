"""Change-point VerifiedProblem assembly (contract + context + verifier)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.changepoint.context import ChangepointVerificationContext
from ves_modeling.changepoint.data_contract import (
    load_hidden_changepoints,
    validate_changepoint_data,
)
from ves_modeling.changepoint.verifier import ChangepointVerifier

verifier = ChangepointVerifier()


def build_changepoint_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "changepoint",
    t_column: str = "t",
    y_column: str = "y",
    tolerance_window: int = 3,
) -> VerifiedProblem:
    """Assemble the change-point detection VerifiedProblem."""
    contract = validate_changepoint_data(
        public_dir,
        t_column=t_column,
        y_column=y_column,
        tolerance_window=tolerance_window,
    )
    hidden = load_hidden_changepoints(host_dir, contract)

    def make_context() -> ChangepointVerificationContext:
        return ChangepointVerificationContext(
            hidden,
            dataset_name=dataset_name,
            n=contract.test_rows,
            tolerance_window=contract.tolerance_window,
        )

    artifact_contract = ArtifactContract(
        filename="changepoints.json",
        media_type="application/json",
        required_fields=("changepoints",),
    )

    return VerifiedProblem(
        contract=artifact_contract,
        context_factory=make_context,
        verifier=verifier,
        judge_spec=JudgeSpec(
            objectives=(
                ObjectiveSpec(observation="f1", direction=Direction.MAXIMIZE),
                ObjectiveSpec(
                    observation="mean_distance",
                    direction=Direction.MINIMIZE,
                ),
            ),
            gates=(
                Gate(name="f1_finite", observation="f1", finite=True),
                Gate(
                    name="mean_distance_finite",
                    observation="mean_distance",
                    finite=True,
                ),
            ),
        ),
        name=f"changepoint:{dataset_name}",
        problem_ref=(
            "ves_modeling.changepoint.problem:build_changepoint_problem"
        ),
        verifier_module="ves_modeling.changepoint.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.changepoint.problem:context_factory",
    )


def context_factory() -> ChangepointVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` and ``VES_MODELING_HOST_DIR`` (required),
    ``VES_MODELING_DATASET``, ``VES_MODELING_T_COLUMN``,
    ``VES_MODELING_Y_COLUMN``, ``VES_MODELING_TOLERANCE_WINDOW``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not public_dir or not host_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR and VES_MODELING_HOST_DIR must be set "
            "to replay a change-point record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "changepoint")
    t_column = os.environ.get("VES_MODELING_T_COLUMN", "t")
    y_column = os.environ.get("VES_MODELING_Y_COLUMN", "y")
    tolerance_window = int(
        os.environ.get("VES_MODELING_TOLERANCE_WINDOW", "3")
    )
    contract = validate_changepoint_data(
        Path(public_dir),
        t_column=t_column,
        y_column=y_column,
        tolerance_window=tolerance_window,
    )
    hidden = load_hidden_changepoints(Path(host_dir), contract)
    return ChangepointVerificationContext(
        hidden,
        dataset_name=dataset_name,
        n=contract.test_rows,
        tolerance_window=contract.tolerance_window,
    )
