"""Forecasting VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.forecasting.context import ForecastingVerificationContext
from ves_modeling.forecasting.data_contract import (
    load_host_labels,
    validate_forecasting_data,
)
from ves_modeling.forecasting.verifier import ForecastingVerifier

verifier = ForecastingVerifier()


def build_forecasting_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "forecasting",
    labels: np.ndarray | None = None,
    time_column: str = "timestamp",
    series_id_column: str = "series_id",
    target_column: str = "target",
    frequency: str = "D",
    row_order: str = "key",
) -> VerifiedProblem:
    """Assemble the forecasting VerifiedProblem.

    ``labels`` may be injected for tests; otherwise loaded from ``host_dir``
    and aligned to public test keys.  ``host_dir`` must never be exposed to
    candidates.
    """
    data_contract = validate_forecasting_data(
        public_dir,
        time_column=time_column,
        series_id_column=series_id_column,
        target_column=target_column,
        frequency=frequency,
        row_order=row_order,
    )
    hidden = (
        labels
        if labels is not None
        else load_host_labels(host_dir, data_contract)
    )
    if labels is not None and row_order == "key":
        raise ValueError(
            "labels injection is not supported for row_order='key': provide "
            "hidden_test_labels.csv so labels can be aligned by key"
        )
    if hidden.size == 0 or not np.isfinite(hidden).all():
        raise ValueError("hidden labels must be non-empty and finite")
    if hidden.size != data_contract.test_rows:
        raise ValueError(
            f"hidden labels count {hidden.size} != test rows "
            f"{data_contract.test_rows}"
        )
    expected_count = int(hidden.size)

    def make_context() -> ForecastingVerificationContext:
        return ForecastingVerificationContext(
            hidden,
            dataset_name=dataset_name,
            expected_count=expected_count,
            series_keys=(
                tuple(key[0] for key in data_contract.test_keys)
                if row_order == "key"
                else None
            ),
            time_keys=(
                tuple(key[1] for key in data_contract.test_keys)
                if row_order == "key"
                else None
            ),
            series_id_column=series_id_column,
            time_column=time_column,
            frequency=frequency,
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
                Gate(
                    name="smape_finite",
                    observation="smape",
                    finite=True,
                ),
            ),
        ),
        name=f"forecasting:{dataset_name}",
        problem_ref=(
            "ves_modeling.forecasting.problem:build_forecasting_problem"
        ),
        verifier_module="ves_modeling.forecasting.problem",
        verifier_attr="verifier",
        context_factory_ref=(
            "ves_modeling.forecasting.problem:context_factory"
        ),
    )


def context_factory() -> ForecastingVerificationContext:
    """Module-level factory used by ``ves replay`` (requires env config).

    Env: ``VES_MODELING_HOST_DIR`` (required), ``VES_MODELING_DATASET``,
    ``VES_MODELING_TIME_COLUMN``, ``VES_MODELING_SERIES_ID_COLUMN``,
    ``VES_MODELING_TARGET_COLUMN``, ``VES_MODELING_FREQUENCY``,
    ``VES_MODELING_ROW_ORDER``.  Key-mode replay additionally requires
    ``VES_MODELING_PUBLIC_DIR`` so labels can be re-aligned to public keys.
    """
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not host_dir:
        raise RuntimeError(
            "VES_MODELING_HOST_DIR must be set to replay a forecasting record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "forecasting")
    time_column = os.environ.get("VES_MODELING_TIME_COLUMN", "timestamp")
    series_id_column = os.environ.get(
        "VES_MODELING_SERIES_ID_COLUMN", "series_id"
    )
    target_column = os.environ.get("VES_MODELING_TARGET_COLUMN", "target")
    frequency = os.environ.get("VES_MODELING_FREQUENCY", "D")
    row_order = os.environ.get("VES_MODELING_ROW_ORDER", "key")
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if row_order == "key":
        if not public_dir:
            raise RuntimeError(
                "VES_MODELING_PUBLIC_DIR must be set to replay "
                "row_order='key' records"
            )
        contract = validate_forecasting_data(
            Path(public_dir),
            time_column=time_column,
            series_id_column=series_id_column,
            target_column=target_column,
            frequency=frequency,
            row_order=row_order,
        )
        labels = load_host_labels(Path(host_dir), contract)
        return ForecastingVerificationContext(
            labels,
            dataset_name=dataset_name,
            expected_count=int(labels.size),
            series_keys=tuple(key[0] for key in contract.test_keys),
            time_keys=tuple(key[1] for key in contract.test_keys),
            series_id_column=series_id_column,
            time_column=time_column,
            frequency=frequency,
            row_order="key",
        )
    if public_dir:
        contract = validate_forecasting_data(
            Path(public_dir),
            time_column=time_column,
            series_id_column=series_id_column,
            target_column=target_column,
            frequency=frequency,
            row_order="input",
        )
        labels = load_host_labels(Path(host_dir), contract)
    else:
        frame = np.genfromtxt(
            Path(host_dir) / "hidden_test_labels.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        labels = np.asarray(
            frame[target_column], dtype=np.float64
        ).reshape(-1)
    if labels.size == 0 or not np.isfinite(labels).all():
        raise ValueError("hidden labels must be non-empty and finite")
    return ForecastingVerificationContext(
        labels,
        dataset_name=dataset_name,
        expected_count=int(labels.size),
        series_id_column=series_id_column,
        time_column=time_column,
        frequency=frequency,
        row_order="input",
    )
