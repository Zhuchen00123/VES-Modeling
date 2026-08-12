"""Candidate generators for the Markov slice."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

FIXTURES_BY_QUANTITY = {
    "transition_probability": "markov_frequency.py",
    "steady_state": "markov_power.py",
    "expected_recurrence_time": "markov_power.py",
}


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockMarkovGenerator:
    """Returns the trusted fixture for the active quantity."""

    def __init__(
        self,
        fixture_dir: Path,
        *,
        quantity: str,
    ) -> None:
        if quantity not in FIXTURES_BY_QUANTITY:
            raise ValueError(
                f"quantity must be one of {tuple(FIXTURES_BY_QUANTITY)}"
            )
        self._fixture_dir = Path(fixture_dir)
        self._name = FIXTURES_BY_QUANTITY[quantity]

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


def _draft_prompt(*, quantity: str, states: tuple[str, ...]) -> str:
    lines = [
        "You are solving a Markov chain estimation task.",
        "",
        "Available files:",
        "/data/problem.json",
        "/data/train.csv",
        "",
        f"quantity to estimate: {quantity}.",
        f"states: {', '.join(states)}.",
        "train.csv contains an observed state sequence.",
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/problem.json and /data/train.csv.",
        (
            "2. Estimate the quantity from the sample (e.g. empirical "
            "transition frequencies / power-iteration steady state) with a "
            "bootstrap confidence interval."
        ),
        "3. Write exactly one artifact:",
        "   /output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        (
            "Do not report or rely on a self-computed reference; the host "
            "holds the true transition matrix and independently evaluates "
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
    quantity: str,
    states: tuple[str, ...],
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
        "Improve the executable solution (smaller error).",
        "",
        (
            "You must still load /data/problem.json and /data/train.csv and "
            "write exactly one artifact:"
        ),
        "/output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        f"quantity: {quantity}. states: {', '.join(states)}.",
        "",
        (
            "Do not report or rely on a self-computed reference; the host "
            "independently evaluates the estimate."
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


class LLMMarkovGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        quantity: str = "transition_probability",
        states: tuple[str, ...] = (),
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._quantity = quantity
        self._states = tuple(states)

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
            _draft_prompt(quantity=self._quantity, states=self._states)
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
                states=self._states,
            )
        )
