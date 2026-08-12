"""Network epidemic (graph SIR) slice for VES Modeling."""

from ves_modeling.networksir.api import (
    NetworkSirSearchResult,
    run_networksir_search,
)
from ves_modeling.networksir.apply import (
    APPLY_SUCCESS_STATUS,
    ApplyNetworkSirResult,
    apply_networksir_solution,
)
from ves_modeling.networksir.schema import API_SCHEMA_VERSION, capabilities

__all__ = [
    "API_SCHEMA_VERSION",
    "APPLY_SUCCESS_STATUS",
    "ApplyNetworkSirResult",
    "NetworkSirSearchResult",
    "apply_networksir_solution",
    "capabilities",
    "run_networksir_search",
]
