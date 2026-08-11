"""Regression VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.regression.context import RegressionVerificationContext
from ves_modeling.regression.data_contract import (
    _check_ids,
    _check_no_duplicate_headers,
    _raw_headers,
    load_host_labels,
    validate_regression_data,
)
from ves_modeling.regression.verifier import RegressionVerifier

verifier = RegressionVerifier()


def load_hidden_labels(
    host_dir: Path,
    *,
    target_column: str = "target",
    id_column: str | None = None,
    test_ids: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Read hidden_test_labels.csv from the host-only directory.

    With ``id_column``/``test_ids`` the labels are validated against the
    public test ids and returned aligned to public test order.
    """
    path = host_dir / "hidden_test_labels.csv"
    _check_no_duplicate_headers(path, _raw_headers(path))
    frame = pd.read_csv(path)
    if target_column not in frame.columns:
        raise ValueError(
            f"hidden labels CSV must have a {target_column!r} column: {path}"
        )
    try:
        labels = frame[target_column].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"hidden labels target column {target_column!r} must be numeric"
        ) from exc
    if labels.size == 0 or not np.isfinite(labels).all():
        raise ValueError("hidden labels must be non-empty and finite")
    if test_ids is not None:
        if id_column is None:
            raise ValueError("id_column is required when test_ids is set")
        host_ids = _check_ids(frame, id_column, "hidden_test_labels.csv")
        if set(host_ids) != set(test_ids):
            raise ValueError(
                "hidden_test_labels.csv ids must match public test ids exactly"
            )
        order = {host_id: index for index, host_id in enumerate(host_ids)}
        labels = np.asarray(
            [labels[order[test_id]] for test_id in test_ids], dtype=np.float64
        )
    return labels


def context_factory() -> RegressionVerificationContext:
    """Module-level factory used by ``ves replay`` (requires env config).

    Env: ``VES_MODELING_HOST_DIR`` (required), ``VES_MODELING_DATASET``,
    ``VES_MODELING_TARGET_COLUMN``, ``VES_MODELING_ID_COLUMN``,
    ``VES_MODELING_ROW_ORDER``.  Any configured ID column additionally
    requires ``VES_MODELING_PUBLIC_DIR`` so labels can be re-aligned to public
    test order; custom-target replay without an ID works without it.
    """
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not host_dir:
        raise RuntimeError(
            "VES_MODELING_HOST_DIR must be set to replay a regression record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "regression")
    target_column = os.environ.get("VES_MODELING_TARGET_COLUMN", "target")
    id_column = os.environ.get("VES_MODELING_ID_COLUMN") or None
    row_order = os.environ.get("VES_MODELING_ROW_ORDER", "input")
    if row_order == "id":
        public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
        if not public_dir:
            raise RuntimeError(
                "VES_MODELING_PUBLIC_DIR must be set to replay "
                "row_order='id' records"
            )
        contract = validate_regression_data(
            Path(public_dir),
            target_column=target_column,
            id_column=id_column,
            row_order=row_order,
        )
        labels = load_host_labels(Path(host_dir), contract)
        return RegressionVerificationContext(
            labels,
            dataset_name=dataset_name,
            expected_count=int(labels.size),
            id_column=id_column,
            prediction_ids=contract.test_ids,
            row_order="id",
        )
    if id_column is not None:
        public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
        if not public_dir:
            raise RuntimeError(
                "VES_MODELING_PUBLIC_DIR must be set to replay records "
                "that declare an id column"
            )
        contract = validate_regression_data(
            Path(public_dir),
            target_column=target_column,
            id_column=id_column,
            row_order=row_order,
        )
        labels = load_host_labels(Path(host_dir), contract)
        return RegressionVerificationContext(
            labels,
            dataset_name=dataset_name,
            expected_count=int(labels.size),
            id_column=id_column,
            prediction_ids=None,
            row_order="input",
        )
    labels = load_hidden_labels(Path(host_dir), target_column=target_column)
    return RegressionVerificationContext(
        labels, dataset_name=dataset_name, expected_count=int(labels.size)
    )


def build_regression_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "regression",
    labels: np.ndarray | None = None,
    target_column: str = "target",
    id_column: str | None = None,
    row_order: str = "input",
) -> VerifiedProblem:
    """Assemble the regression VerifiedProblem.

    ``labels`` may be injected for tests; otherwise loaded from ``host_dir``.
    ``host_dir`` must never be exposed to candidates.
    """
    data_contract = validate_regression_data(
        public_dir,
        target_column=target_column,
        id_column=id_column,
        row_order=row_order,
    )
    hidden = (
        labels
        if labels is not None
        else load_hidden_labels(
            host_dir,
            target_column=target_column,
            id_column=id_column,
            test_ids=(
                data_contract.test_ids if id_column is not None else None
            ),
        )
    )
    if labels is not None and row_order == "id":
        raise ValueError(
            "labels injection is not supported for row_order='id': provide "
            "hidden_test_labels.csv so labels can be aligned by id"
        )
    if hidden.size == 0 or not np.isfinite(hidden).all():
        raise ValueError("hidden labels must be non-empty and finite")
    if hidden.size != data_contract.test_rows:
        raise ValueError(
            f"hidden labels count {hidden.size} != test rows "
            f"{data_contract.test_rows}"
        )
    expected_count = int(hidden.size)

    def make_context() -> RegressionVerificationContext:
        return RegressionVerificationContext(
            hidden,
            dataset_name=dataset_name,
            expected_count=expected_count,
            id_column=id_column,
            prediction_ids=(
                data_contract.test_ids if row_order == "id" else None
            ),
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
            ),
            gates=(Gate(name="rmse_finite", observation="rmse", finite=True),),
        ),
        name=f"regression:{dataset_name}",
        problem_ref="ves_modeling.regression.problem:build_regression_problem",
        verifier_module="ves_modeling.regression.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.regression.problem:context_factory",
    )
