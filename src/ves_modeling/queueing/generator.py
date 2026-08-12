"""Candidate generators for the queueing slice."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

FIXTURES_BY_KIND = {
    "mm1": "des_mm1.py",
    "mmc": "des_mmc.py",
}


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockQueueingGenerator:
    """Returns the trusted DES fixture for the active queue kind."""

    def __init__(
        self,
        fixture_dir: Path,
        *,
        kind: str,
    ) -> None:
        if kind not in FIXTURES_BY_KIND:
            raise ValueError(f"kind must be one of {tuple(FIXTURES_BY_KIND)}")
        self._fixture_dir = Path(fixture_dir)
        self._name = FIXTURES_BY_KIND[kind]

    def draft(self, problem: VerifiedProblem, index: int) -> str:
        return _read_fixture(self._fixture_dir, self._name)

    def improve(
        self, problem: VerifiedProblem, anchor: VerifiedCandidate
    ) -> str:
        return _read_fixture(self._fixture_dir, self._name)


def _artifact_format_prompt() -> str:
    return (
        "{{\n"
        '  "estimate": <finite number>,\n'
        '  "confidence_interval": [<lo>, <hi>]\n'
        "}}\n\n"
        "estimate is required; confidence_interval is optional but if "
        "present lo <= estimate <= hi and both are finite."
    )


def _draft_prompt(*, kind: str, quantity: str) -> str:
    lines = [
        "You are solving a queueing theory simulation task.",
        "",
        "Available files:",
        "/data/problem.json",
        "",
        f"queue kind: {kind}. quantity to estimate: {quantity}.",
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/problem.json.",
        (
            "2. Simulate the queue with a discrete-event simulation "
            "(numpy exponential arrivals/service, event-driven FIFO) over "
            "multiple replications and estimate the quantity with a "
            "confidence interval."
        ),
        "3. Write exactly one artifact:",
        "   /output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        (
            "Do not report or rely on a self-computed reference value; the "
            "host holds the analytic reference and independently evaluates "
            "the estimate."
        ),
        "",
        "Allowed libraries: numpy, pandas.",
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
    kind: str,
    quantity: str,
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
            "Improve the executable solution (smaller error against the "
            "host reference, e.g. more customers/replications)."
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
        f"queue kind: {kind}. quantity: {quantity}.",
        "",
        (
            "Do not report or rely on a self-computed reference value; the "
            "host independently evaluates the estimate."
        ),
        "",
        "Allowed libraries: numpy, pandas.",
        "Do not use pip install, curl, wget, or network access.",
        "",
        (
            "Return the complete solution.py only. Do not wrap the code in "
            "markdown fences."
        ),
    ]
    return "\n".join(lines)


class LLMQueueingGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        kind: str = "mm1",
        quantity: str = "mean_wait",
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._kind = kind
        self._quantity = quantity

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
            _draft_prompt(kind=self._kind, quantity=self._quantity)
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
                kind=self._kind,
                quantity=self._quantity,
            )
        )
