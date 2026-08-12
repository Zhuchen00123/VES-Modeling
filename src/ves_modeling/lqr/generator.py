"""Candidate generators for the LQR slice."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "lqr_riccati.py",
    "lqr_proportional.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockLqrGenerator:
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
        '  "control": [\n'
        '    [<u_0_0>, ..., <u_0_{m-1}>],\n'
        "    ...\n"
        "  ]\n"
        "}}\n\n"
        "Exactly N control vectors, one per horizon step, each with m "
        "finite numbers."
    )


def _draft_prompt(*, n: int, m: int, horizon: int) -> str:
    lines = [
        (
            "You are solving a finite-horizon discrete LQR optimal control "
            "problem."
        ),
        "",
        "Available files:",
        "/data/problem.json",
        "",
        f"state dimension: {n}. control dimension: {m}. horizon: {horizon}.",
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/problem.json (A, B, Q, R, x0, horizon).",
        (
            "2. Produce a control sequence u_0..u_{N-1} (e.g. via the "
            "discrete Riccati recursion)."
        ),
        "3. Write exactly one artifact:",
        "   /output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        (
            "Do not report or rely on a self-computed total cost or "
            "optimality; the host verifier independently simulates the "
            "dynamics and recomputes the total cost."
        ),
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
    n: int,
    m: int,
    horizon: int,
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
        _artifact_format_prompt(),
        "",
        f"state dimension: {n}. control dimension: {m}. horizon: {horizon}.",
        "",
        (
            "Do not report or rely on a self-computed total cost; the host "
            "independently recomputes it. Never claim global optimality."
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


class LLMLqrGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        n: int = 0,
        m: int = 0,
        horizon: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._n = n
        self._m = m
        self._horizon = horizon

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
            _draft_prompt(n=self._n, m=self._m, horizon=self._horizon)
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
                n=self._n,
                m=self._m,
                horizon=self._horizon,
            )
        )
