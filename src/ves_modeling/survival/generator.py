"""Candidate generators for the survival slice."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

FIXTURES_BY_KIND = {
    "risk_score": "cox_linear_risk.py",
    "time": "km_time.py",
}


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockSurvivalGenerator:
    """Returns the trusted fixture for the active output kind."""

    def __init__(
        self,
        fixture_dir: Path,
        *,
        output_kind: str,
    ) -> None:
        if output_kind not in FIXTURES_BY_KIND:
            raise ValueError(
                f"output_kind must be one of {tuple(FIXTURES_BY_KIND)}"
            )
        self._fixture_dir = Path(fixture_dir)
        self._name = FIXTURES_BY_KIND[output_kind]

    def draft(self, problem: VerifiedProblem, index: int) -> str:
        return _read_fixture(self._fixture_dir, self._name)

    def improve(
        self, problem: VerifiedProblem, anchor: VerifiedCandidate
    ) -> str:
        return _read_fixture(self._fixture_dir, self._name)


def _artifact_format_prompt(row_order: str) -> str:
    if row_order == "id":
        return (
            "{{\n"
            '  "predictions": [\n'
            '    {{"id": <test row id>, "prediction": <number>}},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Every test row must appear exactly once, using the id from "
            "test_features.csv."
        )
    return (
        "{{\n"
        '  "predictions": [<number>, ...]\n'
        "}}\n\n"
        "Predictions must be aligned to test_features.csv row order."
    )


def _draft_prompt(
    *,
    output_kind: str,
    row_order: str,
    n_features: int,
) -> str:
    lines = [
        "You are solving a survival analysis task.",
        "",
        "Available files:",
        "/data/train.csv",
        "/data/test_features.csv",
        "",
        (
            "train.csv contains time (positive), event (0/1 censoring) and "
            "features. test_features.csv contains individuals to score."
        ),
        (
            f"output_kind: {output_kind} (risk_score = higher risk; time = "
            "predicted time)."
        ),
        f"Number of feature columns: {n_features}.",
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/train.csv and /data/test_features.csv.",
        (
            "2. Fit a survival model (e.g. a simplified Cox PH risk score "
            "or a Kaplan-Meier median-time fallback) and produce one "
            "prediction per test row."
        ),
        "3. Write exactly one artifact:",
        "   /output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(row_order),
        "",
        "You cannot access hidden evaluation outcomes.",
        "Do not report or rely on a self-computed score.",
        "The host verifier will independently evaluate the predictions.",
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
    output_kind: str,
    row_order: str,
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
        lines.append(f"{name}: {value:.6g}")
    lines += [
        "",
        (
            "Improve the executable solution (higher c-index; in time mode "
            "also lower MAE)."
        ),
        "",
        (
            "You must still load /data/train.csv and /data/test_features.csv "
            "and write exactly one artifact:"
        ),
        "/output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(row_order),
        "",
        f"output_kind: {output_kind}. Number of features: {n_features}.",
        "",
        (
            "You cannot access hidden evaluation outcomes. Do not report a "
            "self-computed score; the host verifier independently evaluates "
            "the predictions."
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


class LLMSurvivalGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        output_kind: str = "risk_score",
        row_order: str = "input",
        n_features: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._output_kind = output_kind
        self._row_order = row_order
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
                output_kind=self._output_kind,
                row_order=self._row_order,
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
                output_kind=self._output_kind,
                row_order=self._row_order,
                n_features=self._n_features,
            )
        )
