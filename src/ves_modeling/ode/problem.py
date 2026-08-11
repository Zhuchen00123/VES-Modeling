"""ODE VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.ode.context import OdeVerificationContext
from ves_modeling.ode.data_contract import (
    load_host_values,
    validate_ode_data,
)
from ves_modeling.ode.verifier import OdeVerifier

verifier = OdeVerifier()


def build_ode_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "ode",
    labels: np.ndarray | None = None,
    time_column: str = "t",
    value_column: str = "y",
    trajectory_id_column: str | None = None,
    row_order: str = "input",
) -> VerifiedProblem:
    """Assemble the ODE VerifiedProblem.

    ``labels`` may be injected for tests (input mode only); otherwise loaded
    from ``host_dir`` and aligned to public test keys.  ``host_dir`` must
    never be exposed to candidates.
    """
    data_contract = validate_ode_data(
        public_dir,
        time_column=time_column,
        value_column=value_column,
        trajectory_id_column=trajectory_id_column,
        row_order=row_order,
    )
    hidden = (
        labels
        if labels is not None
        else load_host_values(host_dir, data_contract)
    )
    if labels is not None and row_order == "key":
        raise ValueError(
            "labels injection is not supported for row_order='key': provide "
            "hidden_test_values.csv so values can be aligned by key"
        )
    if hidden.size == 0 or not np.isfinite(hidden).all():
        raise ValueError("hidden values must be non-empty and finite")
    if hidden.size != data_contract.test_rows:
        raise ValueError(
            f"hidden values count {hidden.size} != test rows "
            f"{data_contract.test_rows}"
        )
    expected_count = int(hidden.size)

    def make_context() -> OdeVerificationContext:
        return OdeVerificationContext(
            hidden,
            dataset_name=dataset_name,
            expected_count=expected_count,
            trajectory_keys=(
                tuple(key[0] for key in data_contract.test_keys)
                if row_order == "key"
                else None
            ),
            time_keys=(
                tuple(key[1] for key in data_contract.test_keys)
                if row_order == "key"
                else None
            ),
            trajectory_id_column=(
                trajectory_id_column or "trajectory_id"
            ),
            time_column=time_column,
            row_order=row_order,
        )

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
            objectives=(
                ObjectiveSpec(
                    observation="rmse", direction=Direction.MINIMIZE
                ),
                ObjectiveSpec(
                    observation="mae", direction=Direction.MINIMIZE
                ),
            ),
            gates=(
                Gate(
                    name="rmse_finite",
                    observation="rmse",
                    finite=True,
                ),
                Gate(
                    name="mae_finite",
                    observation="mae",
                    finite=True,
                ),
            ),
        ),
        name=f"ode:{dataset_name}",
        problem_ref="ves_modeling.ode.problem:build_ode_problem",
        verifier_module="ves_modeling.ode.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.ode.problem:context_factory",
    )


def context_factory() -> OdeVerificationContext:
    """Module-level factory used by ``ves replay`` (requires env config).

    Env: ``VES_MODELING_HOST_DIR`` (required), ``VES_MODELING_DATASET``,
    ``VES_MODELING_TIME_COLUMN``, ``VES_MODELING_VALUE_COLUMN``,
    ``VES_MODELING_TRAJECTORY_ID_COLUMN``, ``VES_MODELING_ROW_ORDER``.
    Key-mode replay additionally requires ``VES_MODELING_PUBLIC_DIR``.
    """
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not host_dir:
        raise RuntimeError(
            "VES_MODELING_HOST_DIR must be set to replay an ODE record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "ode")
    time_column = os.environ.get("VES_MODELING_TIME_COLUMN", "t")
    value_column = os.environ.get("VES_MODELING_VALUE_COLUMN", "y")
    trajectory_id_column = (
        os.environ.get("VES_MODELING_TRAJECTORY_ID_COLUMN") or None
    )
    row_order = os.environ.get("VES_MODELING_ROW_ORDER", "input")
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if row_order == "key":
        if not public_dir:
            raise RuntimeError(
                "VES_MODELING_PUBLIC_DIR must be set to replay "
                "row_order='key' records"
            )
        contract = validate_ode_data(
            Path(public_dir),
            time_column=time_column,
            value_column=value_column,
            trajectory_id_column=trajectory_id_column,
            row_order=row_order,
        )
        labels = load_host_values(Path(host_dir), contract)
        return OdeVerificationContext(
            labels,
            dataset_name=dataset_name,
            expected_count=int(labels.size),
            trajectory_keys=tuple(key[0] for key in contract.test_keys),
            time_keys=tuple(key[1] for key in contract.test_keys),
            trajectory_id_column=trajectory_id_column or "trajectory_id",
            time_column=time_column,
            row_order="key",
        )
    if public_dir:
        contract = validate_ode_data(
            Path(public_dir),
            time_column=time_column,
            value_column=value_column,
            trajectory_id_column=trajectory_id_column,
            row_order="input",
        )
        labels = load_host_values(Path(host_dir), contract)
    else:
        import pandas as pd

        frame = pd.read_csv(Path(host_dir) / "hidden_test_values.csv")
        labels = np.asarray(
            frame[value_column], dtype=np.float64
        ).reshape(-1)
    if labels.size == 0 or not np.isfinite(labels).all():
        raise ValueError("hidden values must be non-empty and finite")
    return OdeVerificationContext(
        labels,
        dataset_name=dataset_name,
        expected_count=int(labels.size),
        trajectory_id_column=trajectory_id_column or "trajectory_id",
        time_column=time_column,
        row_order="input",
    )
