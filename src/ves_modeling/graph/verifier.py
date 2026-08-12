"""Host-computed graph facts; candidate self-reports are ignored."""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.graph.context import GraphVerificationContext
from ves_modeling.graph.data_contract import validate_solution


class GraphVerifier:
    """EvidenceVerifier for per-type graph solution artifacts.

    The host recomputes the objective and feasibility facts:
    - shortest_path: total weight + path violation;
    - max_flow: total value + capacity/conservation violations;
    - min_spanning_tree: total weight + tree violation.
    Candidate-reported objective/feasibility/optimality/gap are never read;
    no optimality claim is ever made.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, GraphVerificationContext):
            raise TypeError(
                "GraphVerifier requires GraphVerificationContext"
            )
        payload = self._parse(raw_artifact)
        contract = context.contract
        if contract.problem_type == "shortest_path":
            return self._verify_shortest_path(payload, contract)
        if contract.problem_type == "max_flow":
            return self._verify_max_flow(payload, contract)
        return self._verify_mst(payload, contract)

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
            raise ValueError("solution.json root must be an object")
        return data

    def _verify_shortest_path(self, payload, contract) -> Evidence:
        path = validate_solution(payload, contract)
        edge_weights = contract.edge_weights()
        total_weight = 0.0
        for u, v in pairwise(path):
            total_weight += edge_weights[(u, v)]
        evidence = [
            Observation(
                value=float(total_weight),
                uncertainty=0.0,
                provenance="host:problem",
                name="total_weight",
            ),
            Observation(
                value=0.0,
                uncertainty=0.0,
                provenance="host:problem",
                name="path_violation",
            ),
        ]
        return _evidence_finite(evidence)

    def _verify_max_flow(self, payload, contract) -> Evidence:
        flow = validate_solution(payload, contract)
        outgoing: dict[str, float] = {node: 0.0 for node in contract.nodes}
        incoming: dict[str, float] = {node: 0.0 for node in contract.nodes}
        capacity_violation = 0.0
        for (u, v), value in flow.items():
            edge_capacity = next(
                edge.weight
                for edge in contract.edges
                if edge.u == u and edge.v == v
            )
            capacity_violation = max(
                capacity_violation, max(0.0, value - edge_capacity)
            )
            outgoing[u] += value
            incoming[v] += value
        conservation_violation = 0.0
        for node in contract.nodes:
            if node in (contract.source, contract.target):
                continue
            conservation_violation = max(
                conservation_violation,
                abs(incoming[node] - outgoing[node]),
            )
        total_value = outgoing[contract.source] - incoming[contract.source]
        evidence = [
            Observation(
                value=float(total_value),
                uncertainty=0.0,
                provenance="host:problem",
                name="total_value",
            ),
            Observation(
                value=float(capacity_violation),
                uncertainty=0.0,
                provenance="host:problem",
                name="capacity_violation",
            ),
            Observation(
                value=float(conservation_violation),
                uncertainty=0.0,
                provenance="host:problem",
                name="conservation_violation",
            ),
        ]
        return _evidence_finite(evidence)

    def _verify_mst(self, payload, contract) -> Evidence:
        selected = validate_solution(payload, contract)
        edge_weights = contract.edge_weights()
        total_weight = 0.0
        parent = {node: node for node in contract.nodes}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for u, v in selected:
            total_weight += edge_weights[(u, v)]
            root_u = find(u)
            root_v = find(v)
            if root_u == root_v:
                tree_violation = 1.0
                break
            parent[root_u] = root_v
        else:
            roots = {find(node) for node in contract.nodes}
            tree_violation = 0.0 if len(roots) == 1 else 1.0
        evidence = [
            Observation(
                value=float(total_weight),
                uncertainty=0.0,
                provenance="host:problem",
                name="total_weight",
            ),
            Observation(
                value=float(tree_violation),
                uncertainty=0.0,
                provenance="host:problem",
                name="tree_violation",
            ),
        ]
        return _evidence_finite(evidence)


def _evidence_finite(evidence: list[Observation]) -> Evidence:
    for observation in evidence:
        if not np.isfinite(observation.value):
            raise ValueError("graph metrics must be finite")
    return Evidence(observations=tuple(evidence))
