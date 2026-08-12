"""Recommendation / matrix factorization vertical slice for VES Modeling."""

from ves_modeling.recommendation.api import (
    RecommendationSearchResult,
    run_recommendation_search,
)
from ves_modeling.recommendation.apply import (
    APPLY_SUCCESS_STATUS,
    ApplyRecommendationResult,
    apply_recommendation_solution,
)
from ves_modeling.recommendation.schema import API_SCHEMA_VERSION, capabilities

__all__ = [
    "API_SCHEMA_VERSION",
    "APPLY_SUCCESS_STATUS",
    "ApplyRecommendationResult",
    "RecommendationSearchResult",
    "apply_recommendation_solution",
    "capabilities",
    "run_recommendation_search",
]
