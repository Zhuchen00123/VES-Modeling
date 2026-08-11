"""Tabular Regression vertical slice for VES Modeling."""

from ves_modeling.regression.api import (
    RegressionSearchResult,
    run_regression_search,
)
from ves_modeling.regression.apply import (
    APPLY_SUCCESS_STATUS,
    ApplyRegressionResult,
    apply_regression_solution,
)
from ves_modeling.regression.schema import API_SCHEMA_VERSION, capabilities

__all__ = [
    "API_SCHEMA_VERSION",
    "APPLY_SUCCESS_STATUS",
    "ApplyRegressionResult",
    "RegressionSearchResult",
    "apply_regression_solution",
    "capabilities",
    "run_regression_search",
]
