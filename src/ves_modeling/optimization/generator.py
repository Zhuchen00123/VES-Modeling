"""Candidate generators for the optimization slice.

- MockOptimizationGenerator: returns trusted fixture code (no LLM).
- LLMOptimizationGenerator: provider-neutral adapter over an OpenAI-compatible
  client, with draft/improve prompts that follow the active problem contract
  (bounded linear/MILP, solution.json with all declared variables).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "linprog_solution.py",
    "milp_solution.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockOptimizationGenerator:
    """Returns preset trusted candidates for draft/improve steps.

    draft(0) -> linprog_solution, draft(1) -> milp_solution,
    improve -> milp_solution (cycled for later improves).
    """

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
        '  "variables": {\n'
        '    "<variable name>": <finite number>,\n'
        "    ...\n"
        "  }\n"
        "}\n\n"
        "The object must contain exactly one key per variable declared in "
        "problem.json (no missing and no extra variables), and every value "
        "must be a finite number."
    )


def _draft_prompt(
    *,
    sense: str,
    variable_names: tuple[str, ...],
    n_constraints: int,
    has_integer: bool,
) -> str:
    lines = [
        "You are solving a bounded linear/MILP optimization problem.",
        "",
        "Available files:",
        "/data/problem.json",
        "",
        (
            "problem.json is the complete instance: sense "
            f"'{sense}', variables with type and finite bounds, a linear "
            "objective, and linear constraints."
        ),
        "",
        f"Declared variables: {', '.join(variable_names)}.",
        f"Number of constraints: {n_constraints}.",
    ]
    if has_integer:
        lines += [
            (
                "The instance contains integer or binary variables; respect "
                "integrality in your solution."
            ),
        ]
    lines += [
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/problem.json.",
        (
            "2. Solve the optimization problem (you may use scipy.optimize "
            "linprog/milp)."
        ),
        "3. Write exactly one artifact:",
        "   /output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        (
            "Do not report or rely on a self-computed objective, feasibility, "
            "optimality or gap."
        ),
        (
            "The host verifier independently recomputes feasibility and the "
            "objective value."
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
    sense: str,
    variable_names: tuple[str, ...],
    n_constraints: int,
    has_integer: bool,
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
            "Improve the executable solution (better feasible objective in "
            f"sense '{sense}', respecting bounds, constraints and "
            "integrality)."
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
    ]
    if has_integer:
        lines += [
            (
                "The instance contains integer or binary variables; respect "
                "integrality."
            ),
        ]
    lines += [
        "",
        (
            "Do not report or rely on a self-computed objective, "
            "feasibility, optimality or gap; the host verifier "
            "independently recomputes the facts. Never claim global "
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


class LLMOptimizationGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        sense: str = "minimize",
        variable_names: tuple[str, ...] = (),
        n_constraints: int = 0,
        has_integer: bool = False,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        if sense not in ("minimize", "maximize"):
            raise ValueError("sense must be 'minimize' or 'maximize'")
        self._sense = sense
        self._variable_names = tuple(variable_names)
        self._n_constraints = n_constraints
        self._has_integer = has_integer

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
                sense=self._sense,
                variable_names=self._variable_names,
                n_constraints=self._n_constraints,
                has_integer=self._has_integer,
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
                sense=self._sense,
                variable_names=self._variable_names,
                n_constraints=self._n_constraints,
                has_integer=self._has_integer,
            )
        )
