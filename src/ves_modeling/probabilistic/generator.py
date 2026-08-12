"""Candidate generators for the probabilistic slice.

- MockProbabilisticGenerator: returns the trusted MLE fixture for the active
  quantity/family.
- LLMProbabilisticGenerator: provider-neutral adapter over an
  OpenAI-compatible client for parameter-estimation prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

FIXTURES_BY_QUANTITY = {
    "mean": "mle_mean_variance.py",
    "variance": "mle_mean_variance.py",
    "quantile": "mle_quantile.py",
    "probability_ge": "mle_family.py",
}


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockProbabilisticGenerator:
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


def _draft_prompt(*, family: str, quantity: str) -> str:
    lines = [
        (
            "You are solving a probabilistic inference (parameter "
            "estimation) task."
        ),
        "",
        "Available files:",
        "/data/problem.json",
        "/data/train.csv",
        "",
        f"distribution family: {family}.",
        f"quantity to estimate: {quantity}.",
        "train.csv contains observed samples (no labels).",
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/problem.json and /data/train.csv.",
        (
            "2. Estimate the quantity from the sample (e.g. MLE: sample "
            "mean/variance, empirical quantile, or gamma/beta "
            "method-of-moments / scipy.fit) with a bootstrap confidence "
            "interval."
        ),
        "3. Write exactly one artifact:",
        "   /output/solution.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        (
            "Do not report or rely on a self-computed reference or "
            "parameter values; the host holds the true parameters and "
            "independently evaluates the estimate."
        ),
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
    family: str,
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
            "host reference)."
        ),
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
        f"distribution family: {family}. quantity: {quantity}.",
        "",
        (
            "Do not report or rely on a self-computed reference or "
            "parameter values; the host independently evaluates the "
            "estimate."
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


class LLMProbabilisticGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        family: str = "normal",
        quantity: str = "mean",
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._family = family
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
            _draft_prompt(
                family=self._family, quantity=self._quantity
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
                family=self._family,
                quantity=self._quantity,
            )
        )
