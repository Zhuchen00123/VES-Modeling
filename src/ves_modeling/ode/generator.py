"""Candidate generators for the ODE slice.

- MockOdeGenerator: returns trusted fixture code (no LLM).
- LLMOdeGenerator: provider-neutral adapter over an OpenAI-compatible client
  with draft/improve prompts for ODE modeling of the active data contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "solve_ivp_fit.py",
    "numpy_fallback.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockOdeGenerator:
    """Returns preset trusted candidates for draft/improve steps.

    draft(0) -> solve_ivp_fit, draft(1) -> numpy_fallback,
    improve -> numpy_fallback (cycled for later improves).
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


def _artifact_format_prompt(
    trajectory_id_column: str | None, row_order: str
) -> str:
    if row_order == "key":
        return (
            "{{\n"
            '  "predictions": [\n'
            '    {{"<trajectory id column>": <trajectory id>, '
            '"<time column>": <number>, "prediction": <number>}},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Use the actual column names from test_features.csv. Every "
            "(trajectory, t) row must appear exactly once."
        )
    return (
        "{{\n"
        '  "predictions": [...]\n'
        "}}\n\n"
        "Predictions must be aligned to test_features.csv row order."
    )


def _draft_prompt(
    *,
    time_column: str,
    value_column: str,
    trajectory_id_column: str | None,
    row_order: str,
) -> str:
    lines = [
        (
            "You are solving an ordinary differential equation (ODE) "
            "modeling task."
        ),
        "",
        "Available files:",
        "/data/train.csv",
        "/data/test_features.csv",
        "",
        (
            f"train.csv contains observed points: {time_column} (independent "
            f"variable) and {value_column} (observed value)."
        ),
    ]
    if trajectory_id_column:
        lines += [
            (
                f"train.csv and test_features.csv contain the trajectory id "
                f"column: {trajectory_id_column}."
            ),
        ]
    lines += [
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/train.csv.",
        "2. Load /data/test_features.csv.",
        (
            "3. Fit an ODE model per trajectory (e.g. scipy.solve_ivp with "
            "a linear y' = a*y + b or y' = a form, or a numpy interpolation "
            "fallback)."
        ),
        "4. Predict every test row.",
        "5. Write exactly one artifact:",
        "   /output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(trajectory_id_column, row_order),
        "",
        "You cannot access hidden evaluation values.",
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
    rmse: float,
    mae: float,
    *,
    time_column: str,
    value_column: str,
    trajectory_id_column: str | None,
    row_order: str,
) -> str:
    lines = [
        "Previous candidate:",
        "",
        code,
        "",
        "Host-verified evidence (independently recomputed by the host):",
        "",
        f"RMSE: {rmse:.6f}",
        f"MAE: {mae:.6f}",
        "",
        (
            "Improve the executable solution (better ODE fit/extrapolation, "
            "per trajectory)."
        ),
        "",
        (
            "You must still load /data/train.csv and /data/test_features.csv, "
            "predict every row, and write exactly one artifact:"
        ),
        "/output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(trajectory_id_column, row_order),
        "",
        f"Time column: {time_column}. Value column: {value_column}.",
    ]
    if trajectory_id_column:
        lines += [
            "Trajectory id column: " + trajectory_id_column + ".",
        ]
    lines += [
        "",
        (
            "You cannot access hidden evaluation values. Do not report a "
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


class LLMOdeGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        time_column: str = "t",
        value_column: str = "y",
        trajectory_id_column: str | None = None,
        row_order: str = "input",
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        if row_order not in ("input", "key"):
            raise ValueError("row_order must be 'input' or 'key'")
        if row_order == "key" and not trajectory_id_column:
            raise ValueError(
                "row_order='key' requires trajectory_id_column"
            )
        self._time_column = time_column
        self._value_column = value_column
        self._trajectory_id_column = trajectory_id_column
        self._row_order = row_order

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
                time_column=self._time_column,
                value_column=self._value_column,
                trajectory_id_column=self._trajectory_id_column,
                row_order=self._row_order,
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
                evidence["rmse"],
                evidence["mae"],
                time_column=self._time_column,
                value_column=self._value_column,
                trajectory_id_column=self._trajectory_id_column,
                row_order=self._row_order,
            )
        )
