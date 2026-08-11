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
from ves_modeling.regression.verifier import RegressionVerifier

verifier = RegressionVerifier()


def load_hidden_labels(host_dir: Path) -> np.ndarray:
    """Read hidden_test_labels.csv from the host-only directory."""
    path = host_dir / "hidden_test_labels.csv"
    frame = pd.read_csv(path)
    if "target" not in frame.columns:
        raise ValueError(f"hidden labels CSV must have a 'target' column: {path}")
    labels = frame["target"].to_numpy(dtype=np.float64)
    if labels.size == 0 or np.isnan(labels).any():
        raise ValueError("hidden labels must be non-empty and finite")
    return labels


def context_factory() -> RegressionVerificationContext:
    """Module-level factory used by ``ves replay`` (requires env config).

    Normal search builds the context from in-memory labels; replay locates the
    host data through ``VES_MODELING_HOST_DIR`` so records stay replayable
    without embedding hidden truth.
    """
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not host_dir:
        raise RuntimeError(
            "VES_MODELING_HOST_DIR must be set to replay a regression record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "regression")
    labels = load_hidden_labels(Path(host_dir))
    return RegressionVerificationContext(
        labels, dataset_name=dataset_name, expected_count=int(labels.size)
    )


def build_regression_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "regression",
    labels: np.ndarray | None = None,
) -> VerifiedProblem:
    """Assemble the regression VerifiedProblem.

    ``labels`` may be injected for tests; otherwise loaded from ``host_dir``.
    ``host_dir`` must never be exposed to candidates.
    """
    hidden = labels if labels is not None else load_hidden_labels(host_dir)
    expected_count = int(hidden.size)

    def make_context() -> RegressionVerificationContext:
        return RegressionVerificationContext(
            hidden, dataset_name=dataset_name, expected_count=expected_count
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
