"""Candidate generators for the change-point detection slice."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "changepoint_cusum.py",
    "changepoint_sliding.py",
)

FIXED_INTERVAL_FALLBACK = '''\
"""Fixed-interval change-point fallback (trusted LLM fallback only)."""

import json
import os

import pandas as pd

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
n = int(len(test))
step = max(1, n // 5)
changepoints = list(range(1, n - 1, step))
if not changepoints:
    changepoints = [max(1, n // 2)]
with open(f"{OUT_DIR}/changepoints.json", "w", encoding="utf-8") as fh:
    json.dump({"changepoints": changepoints}, fh)
'''


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockChangepointGenerator:
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
        '  "changepoints": [<index>, <index>, ...]\n'
        "}}\n\n"
        "One index per detected change point in the test series; indices "
        "are integers in [1, n-2], strictly increasing, at least one."
    )


def _draft_prompt(*, n_test: int) -> str:
    lines = [
        "You are detecting change points in a time series.",
        "",
        "Available files:",
        "/data/train.csv",
        "/data/test_features.csv",
        "",
        f"test series length: {n_test}.",
        "",
        "You must create one complete Python program. The program must:",
        (
            "1. Load /data/train.csv and /data/test_features.csv "
            "(columns t, y)."
        ),
        (
            "2. Detect change points (indices into the test series) with a "
            "method such as CUSUM or a sliding-window mean-shift statistic."
        ),
        "3. Write exactly one artifact:",
        "   /output/changepoints.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        (
            "Do not report or rely on self-computed precision/recall/f1; "
            "the host verifier independently matches your indices against "
            "its own truth within a tolerance window."
        ),
        "Never claim detection optimality.",
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
    n_test: int,
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
        "Improve the executable solution (higher f1, lower mean distance).",
        "",
        (
            "You must still load /data/train.csv and /data/test_features.csv "
            "and write exactly one artifact:"
        ),
        "/output/changepoints.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        f"test series length: {n_test}.",
        "",
        (
            "Do not report or rely on self-computed metrics; the host "
            "independently recomputes precision/recall/f1. Never claim "
            "detection optimality."
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


class LLMChangepointGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        n_test: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._n_test = n_test

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
        return self._complete(_draft_prompt(n_test=self._n_test))

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
                n_test=self._n_test,
            )
        )
