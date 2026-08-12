"""Candidate generators for the network-SIR slice."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "networksir_stochastic.py",
    "networksir_meanfield.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockNetworkSirGenerator:
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
        '  "estimate": <finite number>,\n'
        '  "confidence_interval": [<lo>, <hi>]  # optional\n'
        "}}\n\n"
        "estimate is required and finite; confidence_interval (optional) "
        "must satisfy lo <= estimate <= hi."
    )


def _draft_prompt(*, quantity: str, n_nodes: int) -> str:
    lines = [
        "You are simulating an epidemic on a network (graph SIR).",
        "",
        "Available files:",
        "/data/problem.json",
        "",
        f"quantity: {quantity}. nodes: {n_nodes}.",
        "",
        "You must create one complete Python program. The program must:",
        (
            "1. Load /data/problem.json (beta, gamma, n_nodes, edges, i0, "
            "t_end, quantity)."
        ),
        (
            "2. Simulate the discrete-time network SIR process "
            "(e.g. stochastic replications or a mean-field fallback) and "
            "compute the requested quantity."
        ),
        "3. Write exactly one artifact:",
        "   /output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        (
            "Do not report or rely on a self-computed reference; the host "
            "verifier independently simulates many replications and "
            "computes the error metrics."
        ),
        "Never claim simulation optimality.",
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
    quantity: str,
    n_nodes: int,
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
            "Improve the executable solution (lower relative_error, then "
            "lower absolute_error)."
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
        f"quantity: {quantity}. nodes: {n_nodes}.",
        "",
        (
            "Do not report or rely on self-computed references; the host "
            "independently recomputes the errors. Never claim simulation "
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


class LLMNetworkSirGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        quantity: str = "final_size",
        n_nodes: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._quantity = quantity
        self._n_nodes = n_nodes

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
                quantity=self._quantity, n_nodes=self._n_nodes
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
                quantity=self._quantity,
                n_nodes=self._n_nodes,
            )
        )
