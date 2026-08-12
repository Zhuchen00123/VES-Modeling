"""Candidate generators for the recommendation slice.

- MockRecommendationGenerator: returns trusted fixture code (no LLM).
- LLMRecommendationGenerator: provider-neutral adapter over an
  OpenAI-compatible client for matrix-completion prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "bias_baseline.py",
    "svd_lowrank.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockRecommendationGenerator:
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


def _artifact_format_prompt(row_order: str) -> str:
    if row_order == "key":
        return (
            "{{\n"
            '  "predictions": [\n'
            '    {{"user_id": <user id>, "item_id": <item id>, '
            '"prediction": <number>}},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Every test (user_id, item_id) pair must appear exactly once, "
            "using the ids from test_features.csv."
        )
    return (
        "{{\n"
        '  "predictions": [...]\n'
        "}}\n\n"
        "Predictions must be aligned to test_features.csv row order."
    )


def _draft_prompt(
    *,
    user_id_column: str,
    item_id_column: str,
    row_order: str,
) -> str:
    lines = [
        "You are solving a recommendation / matrix completion task.",
        "",
        "Available files:",
        "/data/train.csv",
        "/data/test_features.csv",
        "",
        "train.csv contains user-item rating history:",
        (
            f"user id column {user_id_column!r}, item id column "
            f"{item_id_column!r}, rating column 'rating'."
        ),
        "test_features.csv contains (user, item) pairs to predict.",
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/train.csv.",
        "2. Load /data/test_features.csv.",
        (
            "3. Fit a rating model (e.g. global mean + user/item bias, or "
            "an SVD low-rank completion) and predict every test pair."
        ),
        "4. Write exactly one artifact:",
        "   /output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(row_order),
        "",
        "You cannot access hidden evaluation ratings.",
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
    ndcg: float,
    *,
    user_id_column: str,
    item_id_column: str,
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
        f"NDCG@5: {ndcg:.6f}",
        "",
        (
            "Improve the executable solution (better rating prediction, "
            "e.g. bias + low-rank factorization)."
        ),
        "",
        (
            "You must still load /data/train.csv and /data/test_features.csv, "
            "predict every pair, and write exactly one artifact:"
        ),
        "/output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(row_order),
        "",
        f"User id column: {user_id_column}. Item id column: {item_id_column}.",
        "",
        (
            "You cannot access hidden evaluation ratings. Do not report a "
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


class LLMRecommendationGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        user_id_column: str = "user_id",
        item_id_column: str = "item_id",
        row_order: str = "key",
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        if row_order not in ("input", "key"):
            raise ValueError("row_order must be 'input' or 'key'")
        self._user_id_column = user_id_column
        self._item_id_column = item_id_column
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
                user_id_column=self._user_id_column,
                item_id_column=self._item_id_column,
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
                evidence["ndcg@5"],
                user_id_column=self._user_id_column,
                item_id_column=self._item_id_column,
                row_order=self._row_order,
            )
        )
