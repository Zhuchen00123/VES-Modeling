"""Candidate generators for the anomaly slice.

- MockAnomalyGenerator: returns trusted fixture code (no LLM); the fixture
  set is selected by the active ``output_mode``.
- LLMAnomalyGenerator: provider-neutral adapter over an OpenAI-compatible
  client with draft/improve prompts for scoring or labeling anomalies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

FIXTURES_BY_MODE = {
    "score": ("isolation_forest_score.py",),
    "label": ("zscore_threshold.py",),
}


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockAnomalyGenerator:
    """Returns preset trusted candidates for the active output mode."""

    def __init__(
        self,
        fixture_dir: Path,
        *,
        output_mode: str = "score",
    ) -> None:
        if output_mode not in FIXTURES_BY_MODE:
            raise ValueError(
                f"output_mode must be one of {tuple(FIXTURES_BY_MODE)}"
            )
        self._fixture_dir = Path(fixture_dir)
        self._fixtures = FIXTURES_BY_MODE[output_mode]

    def draft(self, problem: VerifiedProblem, index: int) -> str:
        name = self._fixtures[index % len(self._fixtures)]
        return _read_fixture(self._fixture_dir, name)

    def improve(
        self, problem: VerifiedProblem, anchor: VerifiedCandidate
    ) -> str:
        generation = anchor.candidate.generation
        name = self._fixtures[generation % len(self._fixtures)]
        return _read_fixture(self._fixture_dir, name)


def _artifact_format_prompt(output_mode: str) -> str:
    if output_mode == "score":
        return (
            "{{\n"
            '  "scores": [<number>, ...]\n'
            "}}\n\n"
            "One score per test row in test_features.csv row order; higher "
            "scores mean more anomalous. All scores must be finite numbers."
        )
    return (
        "{{\n"
        '  "labels": ["normal"|"anomaly"|0|1, ...]\n'
        "}}\n\n"
        "One label per test row in test_features.csv row order. Labels must "
        "be 'normal'/'anomaly' or 0/1 (1 = anomaly), with at least one of "
        "each class, and must not mix string and numeric encodings."
    )


def _draft_prompt(
    *,
    output_mode: str,
    n_features: int,
) -> str:
    lines = [
        "You are solving a binary anomaly detection task.",
        "",
        "Available files:",
        "/data/train.csv",
        "/data/test_features.csv",
        "",
        (
            "train.csv contains features of normal samples only (no "
            "labels). test_features.csv contains samples to score/label."
        ),
        f"Number of feature columns: {n_features}.",
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/train.csv.",
        "2. Load /data/test_features.csv.",
        (
            "3. Build an anomaly model from the normal train samples and "
            "produce one output per test row."
        ),
        "4. Write exactly one artifact:",
        "   /output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(output_mode),
        "",
        "You cannot access hidden evaluation labels.",
        "Do not report or rely on a self-computed score.",
        "The host verifier will independently evaluate the predictions.",
        "",
        "Allowed libraries: numpy, pandas, scikit-learn.",
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
    output_mode: str,
    n_features: int,
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
        lines.append(f"{name}: {value:.6f}")
    lines += [
        "",
        (
            "Improve the executable solution (better separation of normal "
            "vs anomalous test rows)."
        ),
        "",
        (
            "You must still load /data/train.csv and /data/test_features.csv, "
            "produce one output per test row, and write exactly one artifact:"
        ),
        "/output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(output_mode),
        "",
        f"Number of feature columns: {n_features}.",
        "",
        (
            "You cannot access hidden evaluation labels. Do not report a "
            "self-computed score; the host verifier independently evaluates "
            "the predictions."
        ),
        "",
        "Allowed libraries: numpy, pandas, scikit-learn.",
        "Do not use pip install, curl, wget, or network access.",
        "",
        (
            "Return the complete solution.py only. Do not wrap the code in "
            "markdown fences."
        ),
    ]
    return "\n".join(lines)


class LLMAnomalyGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        output_mode: str = "score",
        n_features: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        if output_mode not in ("score", "label"):
            raise ValueError("output_mode must be 'score' or 'label'")
        self._output_mode = output_mode
        self._n_features = n_features

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
                output_mode=self._output_mode,
                n_features=self._n_features,
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
                output_mode=self._output_mode,
                n_features=self._n_features,
            )
        )
