"""Candidate generators for the graph slice.

- MockGraphGenerator: returns the trusted fixture for the active problem
  type (Dijkstra / Ford-Fulkerson / Kruskal), no LLM.
- LLMGraphGenerator: provider-neutral adapter over an OpenAI-compatible
  client with draft/improve prompts for the active problem type.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

FIXTURES_BY_TYPE = {
    "shortest_path": "dijkstra_shortest_path.py",
    "max_flow": "ford_fulkerson_max_flow.py",
    "min_spanning_tree": "kruskal_mst.py",
}


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockGraphGenerator:
    """Returns the trusted fixture for the active problem type."""

    def __init__(
        self,
        fixture_dir: Path,
        *,
        problem_type: str,
    ) -> None:
        if problem_type not in FIXTURES_BY_TYPE:
            raise ValueError(
                f"problem_type must be one of {tuple(FIXTURES_BY_TYPE)}"
            )
        self._fixture_dir = Path(fixture_dir)
        self._name = FIXTURES_BY_TYPE[problem_type]

    def draft(self, problem: VerifiedProblem, index: int) -> str:
        return _read_fixture(self._fixture_dir, self._name)

    def improve(
        self, problem: VerifiedProblem, anchor: VerifiedCandidate
    ) -> str:
        return _read_fixture(self._fixture_dir, self._name)


def _artifact_format_prompt(problem_type: str) -> str:
    if problem_type == "shortest_path":
        return (
            "{{\n"
            '  "path": ["<node>", ...]\n'
            "}}\n\n"
            "A simple path from source to target using declared edges."
        )
    if problem_type == "max_flow":
        return (
            "{{\n"
            '  "flow": {\n'
            '    "<u>-><v>": <finite number>,\n'
            "    ...\n"
            "  }\n"
            "}\n\n"
            "Exactly one key per declared edge, non-negative, "
            "capacity-respecting with flow conservation at every internal "
            "node."
        )
    return (
        "{{\n"
        '  "edges": [["<u>", "<v>"], ...]\n'
        "}}\n\n"
        "Exactly n-1 declared edges forming a spanning tree."
    )


def _draft_prompt(
    *,
    problem_type: str,
    n_nodes: int,
    n_edges: int,
    source: str | None,
    target: str | None,
) -> str:
    lines = [
        "You are solving a graph/network optimization problem.",
        "",
        "Available files:",
        "/data/problem.json",
        "",
        f"problem_type: {problem_type}.",
        f"nodes: {n_nodes}. edges: {n_edges}.",
    ]
    if source is not None:
        lines += [
            f"source: {source}. target: {target}.",
        ]
    lines += [
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/problem.json.",
        (
            "2. Solve the graph problem with a correct algorithm "
            "(e.g. Dijkstra/BFS, Ford-Fulkerson, Kruskal)."
        ),
        "3. Write exactly one artifact:",
        "   /output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(problem_type),
        "",
        (
            "Do not report or rely on a self-computed objective, "
            "feasibility, optimality or gap."
        ),
        "The host verifier independently recomputes the facts.",
        "Never claim global optimality.",
        "",
        "Allowed libraries: numpy, pandas (pure Python is preferred).",
        "Do not use pip install, curl, wget, or network access.",
        "",
        "Your response must contain the complete solution.py only.",
        "Do not wrap the code in markdown fences.",
    ]
    return "\n".join(lines)


def _improve_prompt(
    code: str,
    evidence: dict[str, float],
    *,
    problem_type: str,
    n_nodes: int,
    n_edges: int,
    source: str | None,
    target: str | None,
) -> str:
    lines = [
        "Previous candidate:",
        "",
        code,
        "",
        "Host-verified evidence (independently recomputed by the host):",
        "",
    ]
    for name, value in evidence.items():
        lines.append(f"{name}: {value:.6g}")
    lines += [
        "",
        (
            "Improve the executable solution (better feasible objective "
            "for the graph problem)."
        ),
        "",
        (
            "You must still load /data/problem.json and write exactly one "
            "artifact:"
        ),
        "/output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(problem_type),
        "",
        f"nodes: {n_nodes}. edges: {n_edges}.",
    ]
    if source is not None:
        lines += [f"source: {source}. target: {target}."]
    lines += [
        "",
        (
            "Do not report or rely on a self-computed objective, "
            "feasibility, optimality or gap; the host verifier "
            "independently recomputes the facts. Never claim global "
            "optimality."
        ),
        "",
        "Allowed libraries: numpy, pandas (pure Python is preferred).",
        "Do not use pip install, curl, wget, or network access.",
        "",
        (
            "Return the complete solution.py only. Do not wrap the code in "
            "markdown fences."
        ),
    ]
    return "\n".join(lines)


class LLMGraphGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        problem_type: str = "shortest_path",
        n_nodes: int = 0,
        n_edges: int = 0,
        source: str | None = None,
        target: str | None = None,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        if problem_type not in ("shortest_path", "max_flow", "min_spanning_tree"):
            raise ValueError(
                "problem_type must be shortest_path, max_flow or "
                "min_spanning_tree"
            )
        self._problem_type = problem_type
        self._n_nodes = n_nodes
        self._n_edges = n_edges
        self._source = source
        self._target = target

    def _complete(self, prompt: str) -> str:
        try:
            return self._llm.complete(prompt)
        except Exception as exc:
            if self._fallback_code is None:
                raise
            logger.warning(
                "LLM call failed (%s: %s); using fallback candidate",
                type(exc).__name__,
                exc,
            )
            return self._fallback_code

    def draft(self, problem: VerifiedProblem, index: int) -> str:
        return self._complete(
            _draft_prompt(
                problem_type=self._problem_type,
                n_nodes=self._n_nodes,
                n_edges=self._n_edges,
                source=self._source,
                target=self._target,
            )
        )

    def improve(
        self, problem: VerifiedProblem, anchor: VerifiedCandidate
    ) -> str:
        evidence = {
            observation.name: observation.value
            for observation in anchor.record.evidence
        }
        return self._complete(
            _improve_prompt(
                anchor.program,
                evidence,
                problem_type=self._problem_type,
                n_nodes=self._n_nodes,
                n_edges=self._n_edges,
                source=self._source,
                target=self._target,
            )
        )
