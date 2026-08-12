"""Host-computed recommendation metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.recommendation.context import (
    RecommendationVerificationContext,
)
from ves_modeling.recommendation.data_contract import validate_predictions


class RecommendationVerifier:
    """EvidenceVerifier for recommendation prediction artifacts.

    The host aligns predictions to hidden ratings by row order or
    ``(user_id, item_id)`` key and recomputes RMSE and MAE.  ``ndcg@5`` is an
    audit observation: per user, predicted-order DCG over ideal order with
    k = min(5, count); users with fewer than two test items contribute 1.0.
    Input mode carries no user keys, so ``ndcg@5`` is always 1.0 there.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, RecommendationVerificationContext):
            raise TypeError(
                "RecommendationVerifier requires "
                "RecommendationVerificationContext"
            )
        payload = self._parse(raw_artifact)
        test_keys = (
            tuple(zip(context.user_keys, context.item_keys))
            if context.row_order == "key"
            else None
        )
        predictions = validate_predictions(
            payload,
            expected_count=context.expected_count,
            test_keys=test_keys,
            key_columns=(context.user_id_column, context.item_id_column),
        )
        ratings = context.hidden_ratings()
        rmse = float(np.sqrt(np.mean((predictions - ratings) ** 2)))
        mae = float(np.mean(np.abs(predictions - ratings)))
        ndcg = _ndcg_at_5(test_keys, ratings, predictions) if test_keys else 1.0
        metrics = (rmse, mae, ndcg)
        for value in metrics:
            if not np.isfinite(value):
                raise ValueError("recommendation metrics must be finite")
        return Evidence(
            observations=(
                Observation(
                    value=rmse,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="rmse",
                ),
                Observation(
                    value=mae,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="mae",
                ),
                Observation(
                    value=ndcg,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="ndcg@5",
                ),
            )
        )

    @staticmethod
    def _parse(raw_artifact: RawArtifact) -> dict[str, Any]:
        text = (
            raw_artifact.content.decode("utf-8")
            if isinstance(raw_artifact.content, bytes)
            else raw_artifact.content
        )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ValueError("predictions.json root must be an object")
        return data


def _dcg(relevance: list[float]) -> float:
    value = 0.0
    for rank, rel in enumerate(relevance):
        value += (2.0**rel - 1.0) / np.log2(rank + 2)
    return float(value)


def _ndcg_at_5(
    test_keys: list[tuple[str, str]],
    ratings: np.ndarray,
    predictions: np.ndarray,
) -> float:
    by_user: dict[str, list[tuple[int, float, float]]] = {}
    for index, (user_key, _item_key) in enumerate(test_keys):
        by_user.setdefault(user_key, []).append(
            (index, float(predictions[index]), float(ratings[index]))
        )
    total = 0.0
    for user_key, entries in by_user.items():
        count = len(entries)
        if count < 2:
            total += 1.0
            continue
        k = min(5, count)
        predicted_order = sorted(
            entries, key=lambda entry: entry[1], reverse=True
        )[:k]
        ideal_order = sorted(
            entries, key=lambda entry: entry[2], reverse=True
        )[:k]
        dcg = _dcg([entry[2] for entry in predicted_order])
        idcg = _dcg([entry[2] for entry in ideal_order])
        total += dcg / idcg if idcg > 0.0 else 1.0
    return float(total / len(by_user))
