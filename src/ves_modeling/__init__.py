"""VES Modeling: verifier-first executable search for computational modeling."""

from ves_modeling.anomaly.problem import build_anomaly_problem
from ves_modeling.classification.problem import build_classification_problem
from ves_modeling.clustering.problem import build_clustering_problem
from ves_modeling.forecasting.problem import build_forecasting_problem
from ves_modeling.graph.problem import build_graph_problem
from ves_modeling.montecarlo.problem import build_montecarlo_problem
from ves_modeling.ode.problem import build_ode_problem
from ves_modeling.optimization.problem import build_optimization_problem
from ves_modeling.regression.problem import build_regression_problem

__all__ = [
    "build_anomaly_problem",
    "build_classification_problem",
    "build_clustering_problem",
    "build_forecasting_problem",
    "build_graph_problem",
    "build_montecarlo_problem",
    "build_ode_problem",
    "build_optimization_problem",
    "build_regression_problem",
]
__version__ = "0.1.0"
