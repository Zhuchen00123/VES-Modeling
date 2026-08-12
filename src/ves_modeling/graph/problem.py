"""Graph VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.graph.context import GraphVerificationContext
from ves_modeling.graph.data_contract import validate_graph_data
from ves_modeling.graph.verifier import GraphVerifier

verifier = GraphVerifier()


def build_graph_problem(
    public_dir: Path,
    *,
    dataset_name: str = "graph",
    tolerance: float = 1e-6,
) -> VerifiedProblem:
    """Assemble the graph VerifiedProblem.

    ``problem.json`` is the complete (public) instance, so no host directory
    exists.  Feasibility gates run before the type-specific objective; global
    optimality is never claimed.
    """
    contract = validate_graph_data(public_dir, tolerance=tolerance)
    if contract.problem_type == "max_flow":
        objective = ObjectiveSpec(
            observation="total_value", direction=Direction.MAXIMIZE
        )
        gates = (
            Gate(
                name="total_value_finite",
                observation="total_value",
                finite=True,
            ),
            Gate(
                name="capacity_feasible",
                observation="capacity_violation",
                maximum=contract.tolerance,
            ),
            Gate(
                name="conservation_feasible",
                observation="conservation_violation",
                maximum=contract.tolerance,
            ),
        )
    else:
        objective = ObjectiveSpec(
            observation="total_weight", direction=Direction.MINIMIZE
        )
        gates = [
            Gate(
                name="total_weight_finite",
                observation="total_weight",
                finite=True,
            ),
        ]
        if contract.problem_type == "shortest_path":
            gates.append(
                Gate(
                    name="path_feasible",
                    observation="path_violation",
                    maximum=0.0,
                )
            )
        else:
            gates.append(
                Gate(
                    name="tree_feasible",
                    observation="tree_violation",
                    maximum=0.0,
                )
            )
        gates = tuple(gates)

    def make_context() -> GraphVerificationContext:
        return GraphVerificationContext(contract, dataset_name=dataset_name)

    artifact_contract = ArtifactContract(
        filename="solution.json",
        media_type="application/json",
        required_fields=(),
    )

    return VerifiedProblem(
        contract=artifact_contract,
        context_factory=make_context,
        verifier=verifier,
        judge_spec=JudgeSpec(objectives=(objective,), gates=gates),
        name=f"graph:{dataset_name}",
        problem_ref="ves_modeling.graph.problem:build_graph_problem",
        verifier_module="ves_modeling.graph.problem",
        verifier_attr="verifier",
        context_factory_ref="ves_modeling.graph.problem:context_factory",
    )


def context_factory() -> GraphVerificationContext:
    """Module-level factory used by ``ves replay``.

    Env: ``VES_MODELING_PUBLIC_DIR`` (required, contains problem.json),
    ``VES_MODELING_DATASET``, ``VES_MODELING_TOLERANCE``.
    """
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay a graph record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "graph")
    tolerance = float(os.environ.get("VES_MODELING_TOLERANCE", "1e-6"))
    contract = validate_graph_data(Path(public_dir), tolerance=tolerance)
    return GraphVerificationContext(contract, dataset_name=dataset_name)
