"""Candidate generators for the assignment/TSP slice."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

FIXTURES_BY_TYPE = {
    "assignment": "assignment_hungarian.py",
    "tsp": "tsp_nearest_2opt.py",
}


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockAssignGenerator:
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


def _artifact_format_prompt(problem_type: str, start: int) -> str:
    if problem_type == "assignment":
        return (
            "{{\n"
            '  "assignment": [<j0>, <j1>, ..., <j_{n-1}>]\n'
            "}}\n\n"
            "A permutation of 0..n-1 (row i assigned to column j_i)."
        )
    return (
        "{{\n"
        '  "tour": [<node0>, <node1>, ..., <node_{n-1}>]\n'
        "}}\n\n"
        f"A permutation of 0..n-1 starting at {start} (cyclic tour)."
    )


def _draft_prompt(
    *,
    problem_type: str,
    size: int,
    start: int,
) -> str:
    lines = [
        "You are solving a combinatorial optimization problem.",
        "",
        "Available files:",
        "/data/problem.json",
        "",
        f"problem_type: {problem_type}. size: {size}.",
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/problem.json.",
        (
            "2. Solve the problem with a correct combinatorial algorithm "
            "(e.g. Hungarian for assignment; nearest-neighbor + 2-opt for "
            "TSP)."
        ),
        "3. Write exactly one artifact:",
        "   /output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(problem_type, start),
        "",
        (
            "Do not report or rely on a self-computed objective, "
            "feasibility, optimality or gap."
        ),
        "The host verifier independently recomputes the total cost.",
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
    size: int,
    start: int,
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
        "Improve the executable solution (lower total cost).",
        "",
        (
            "You must still load /data/problem.json and write exactly one "
            "artifact:"
        ),
        "/output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(problem_type, start),
        "",
        f"problem_type: {problem_type}. size: {size}.",
        "",
        (
            "Do not report or rely on a self-computed objective; the host "
            "independently recomputes the total cost. Never claim global "
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


class LLMAssignGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        problem_type: str = "assignment",
        size: int = 0,
        start: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._problem_type = problem_type
        self._size = size
        self._start = start

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
                size=self._size,
                start=self._start,
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
                size=self._size,
                start=self._start,
            )
        )
