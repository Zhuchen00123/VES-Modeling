"""Candidate generators for the multi-objective slice.

- MockMooGenerator: returns the trusted fixture (random-weight scalarization
  with scipy linprog/milp plus bounded random sampling).
- LLMMooGenerator: provider-neutral adapter over an OpenAI-compatible client
  with draft/improve prompts for the bi-objective solution-set artifact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "scalarization_linprog.py",
    "random_sampling.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockMooGenerator:
    """Returns preset trusted candidates for draft/improve steps."""

    def __init__(
        self,
        fixture_dir: Path,
        *,
        drafts: tuple[str, ...] = MOCK_FIXTURES[:2],
        improves: tuple[str, ...] = MOCK_FIXTURES[1:],
    ) -> None:
        self._fixture_dir = Path(fixture_dir)
        self._drafts = tuple(drafts)
        self._improves = tuple(improves)

    def draft(self, problem: VerifiedProblem, index: int) -> str:
        name = self._drafts[index % len(self._drafts)]
        return _read_fixture(self._fixture_dir, name)

    def improve(
        self, problem: VerifiedProblem, anchor: VerifiedCandidate
    ) -> str:
        generation = anchor.candidate.generation
        name = self._improves[generation % len(self._improves)]
        return _read_fixture(self._fixture_dir, name)


def _artifact_format_prompt() -> str:
    return (
        "{{\n"
        '  "solutions": [\n'
        '    {{"variables": {{"<variable name>": <finite number>, ...}}}},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Provide at least one feasible solution (a single "
        '{"variables": {...}} is also accepted); every solution must '
        "contain exactly one value per declared variable."
    )


def _draft_prompt(
    *,
    variable_names: tuple[str, ...],
    n_constraints: int,
) -> str:
    lines = [
        (
            "You are solving a bi-objective (Pareto) linear optimization "
            "problem."
        ),
        "",
        "Available files:",
        "/data/problem.json",
        "",
        (
            "problem.json declares variables with finite bounds, exactly "
            "two linear objectives and linear constraints."
        ),
        "",
        f"Declared variables: {', '.join(variable_names)}.",
        f"Number of constraints: {n_constraints}.",
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/problem.json.",
        (
            "2. Produce a spread of feasible solutions (e.g. random-weight "
            "scalarization with scipy linprog/milp plus bounded random "
            "sampling)."
        ),
        "3. Write exactly one artifact:",
        "   /output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        (
            "Do not report or rely on self-computed objective values, "
            "feasibility, optimality or gap."
        ),
        (
            "The host independently recomputes every solution and the "
            "Pareto frontier quality."
        ),
        "Never claim global optimality.",
        "",
        "Allowed libraries: numpy, pandas, scipy.",
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
    variable_names: tuple[str, ...],
    n_constraints: int,
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
            "Improve the executable solution (larger hypervolume, i.e. a "
            "better spread of non-dominated feasible solutions)."
        ),
        "",
        (
            "You must still load /data/problem.json and write exactly one "
            "artifact:"
        ),
        "/output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        f"Declared variables: {', '.join(variable_names)}.",
        f"Number of constraints: {n_constraints}.",
        "",
        (
            "Do not report or rely on self-computed objective values; the "
            "host independently recomputes everything. Never claim global "
            "optimality."
        ),
        "",
        "Allowed libraries: numpy, pandas, scipy.",
        "Do not use pip install, curl, wget, or network access.",
        "",
        (
            "Return the complete solution.py only. Do not wrap the code in "
            "markdown fences."
        ),
    ]
    return "\n".join(lines)


class LLMMooGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        variable_names: tuple[str, ...] = (),
        n_constraints: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._variable_names = tuple(variable_names)
        self._n_constraints = n_constraints

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
                variable_names=self._variable_names,
                n_constraints=self._n_constraints,
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
                variable_names=self._variable_names,
                n_constraints=self._n_constraints,
            )
        )
