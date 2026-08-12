"""Recommendation VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.recommendation.context import (
    RecommendationVerificationContext,
)
from ves_modeling.recommendation.data_contract import (
    load_host_ratings,
    validate_recommendation_data,
)
from ves_modeling.recommendation.verifier import RecommendationVerifier

verifier = RecommendationVerifier()


def build_recommendation_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "recommendation",
    labels: np.ndarray | None = None,
    user_id_column: str = "user_id",
    item_id_column: str = "item_id",
    rating_column: str = "rating",
    row_order: str = "key",
) -> VerifiedProblem:
    """Assemble the recommendation VerifiedProblem."""
    data_contract = validate_recommendation_data(
        public_dir,
        user_id_column=user_id_column,
        item_id_column=item_id_column,
        rating_column=rating_column,
        row_order=row_order,
    )
    hidden = (
        labels
        if labels is not None
        else load_host_ratings(host_dir, data_contract)
    )
    if labels is not None and row_order == "key":
        raise ValueError(
            "labels injection is not supported for row_order='key': provide "
            "hidden_test_ratings.csv so ratings can be aligned by key"
        )
    if hidden.size == 0 or not np.isfinite(hidden).all():
        raise ValueError("hidden ratings must be non-empty and finite")
    if hidden.size != data_contract.test_rows:
        raise ValueError(
            f"hidden ratings count {hidden.size} != test rows "
            f"{data_contract.test_rows}"
        )
    expected_count = int(hidden.size)

    def make_context() -> RecommendationVerificationContext:
        return RecommendationVerificationContext(
            hidden,
            dataset_name=dataset_name,
            expected_count=expected_count,
            user_keys=(
                tuple(key[0] for key in data_contract.test_keys)
                if row_order == "key"
                else None
            ),
            item_keys=(
                tuple(key[1] for key in data_contract.test_keys)
                if row_order == "key"
                else None
            ),
            user_id_column=user_id_column,
            item_id_column=item_id_column,
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
        name=f"recommendation:{dataset_name}",
        problem_ref=(
            "ves_modeling.recommendation.problem:"
            "build_recommendation_problem"
        ),
        verifier_module="ves_modeling.recommendation.problem",
        verifier_attr="verifier",
        context_factory_ref=(
            "ves_modeling.recommendation.problem:context_factory"
        ),
    )


def context_factory() -> RecommendationVerificationContext:
    """Module-level factory used by ``ves replay`` (requires env config).

    Env: ``VES_MODELING_HOST_DIR`` (required), ``VES_MODELING_DATASET``,
    ``VES_MODELING_USER_ID_COLUMN``, ``VES_MODELING_ITEM_ID_COLUMN``,
    ``VES_MODELING_RATING_COLUMN``, ``VES_MODELING_ROW_ORDER``.
    Key-mode replay additionally requires ``VES_MODELING_PUBLIC_DIR``.
    """
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not host_dir:
        raise RuntimeError(
            "VES_MODELING_HOST_DIR must be set to replay a recommendation "
            "record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "recommendation")
    user_id_column = os.environ.get("VES_MODELING_USER_ID_COLUMN", "user_id")
    item_id_column = os.environ.get("VES_MODELING_ITEM_ID_COLUMN", "item_id")
    rating_column = os.environ.get("VES_MODELING_RATING_COLUMN", "rating")
    row_order = os.environ.get("VES_MODELING_ROW_ORDER", "key")
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay a "
            "recommendation record"
        )
    contract = validate_recommendation_data(
        Path(public_dir),
        user_id_column=user_id_column,
        item_id_column=item_id_column,
        rating_column=rating_column,
        row_order=row_order,
    )
    ratings = load_host_ratings(Path(host_dir), contract)
    return RecommendationVerificationContext(
        ratings,
        dataset_name=dataset_name,
        expected_count=int(ratings.size),
        user_keys=(
            tuple(key[0] for key in contract.test_keys)
            if row_order == "key"
            else None
        ),
        item_keys=(
            tuple(key[1] for key in contract.test_keys)
            if row_order == "key"
            else None
        ),
        user_id_column=user_id_column,
        item_id_column=item_id_column,
        row_order=row_order,
    )
