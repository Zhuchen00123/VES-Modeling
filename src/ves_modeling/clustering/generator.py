"""Candidate generators for the clustering slice.

- MockClusteringGenerator: returns trusted fixture code (no LLM).
- LLMClusteringGenerator: provider-neutral adapter over an OpenAI-compatible
  client with draft/improve prompts for clustering of the active contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "kmeans_fit.py",
    "spectral_fallback.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockClusteringGenerator:
    """Returns preset trusted candidates for draft/improve steps.

    draft(0) -> kmeans_fit, draft(1) -> spectral_fallback,
    improve -> spectral_fallback (cycled for later improves).
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


def _artifact_format_prompt(id_column: str | None, row_order: str) -> str:
    if row_order == "id":
        return (
            "{{\n"
            '  "predictions": [\n'
            '    {{"id": <test row id>, "label": <cluster label>}},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Every test row must appear exactly once, using the id from "
            "test_features.csv. The id is a row identifier, not a feature."
        )
    return (
        "{{\n"
        '  "labels": ["<cluster label>", ...]\n'
        "}}\n\n"
        "Labels must be aligned to test_features.csv row order."
    )


def _draft_prompt(
    *,
    id_column: str | None,
    row_order: str,
    n_features: int,
) -> str:
    lines = [
        "You are solving an unsupervised clustering task.",
        "",
        "Available files:",
        "/data/train.csv",
        "/data/test_features.csv",
        "",
        (
            "train.csv and test_features.csv contain numeric feature "
            "matrices with no labels."
        ),
        f"Number of feature columns: {n_features}.",
        "",
    ]
    if id_column:
        lines += [
            "test_features.csv contains the row id column:",
            id_column,
            "",
            "The id column must NOT be used as a feature.",
            "",
        ]
    lines += [
        "You must create one complete Python program. The program must:",
        "1. Load /data/train.csv.",
        "2. Load /data/test_features.csv.",
        (
            "3. Choose a number of clusters (at least 2) and assign every "
            "test row to a cluster (e.g. sklearn KMeans on train with a "
            "simple elbow heuristic, or a spectral/threshold fallback)."
        ),
        "4. Write exactly one artifact:",
        "   /output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(id_column, row_order),
        "",
        (
            "Cluster labels must be non-empty strings or finite numbers, "
            "and the output must contain at least two distinct cluster "
            "labels."
        ),
        "",
        "You cannot access hidden evaluation labels.",
        "Do not report or rely on a self-computed score.",
        "The host verifier will independently evaluate the partition.",
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
    id_column: str | None,
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
        lines.append(f"{name}: {value:.6f}")
    lines += [
        "",
        (
            "Improve the executable solution (better partition; ARI/NMI "
            "are permutation-invariant so cluster names do not need to "
            "match the reference)."
        ),
        "",
        (
            "You must still load /data/train.csv and /data/test_features.csv, "
            "assign every test row, and write exactly one artifact:"
        ),
        "/output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(id_column, row_order),
        "",
        f"Number of feature columns: {n_features}.",
        "",
        (
            "Cluster labels must be non-empty strings or finite numbers "
            "with at least two distinct labels. You cannot access hidden "
            "evaluation labels; do not report a self-computed score."
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


class LLMClusteringGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        id_column: str | None = None,
        row_order: str = "input",
        n_features: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        if row_order not in ("input", "id"):
            raise ValueError("row_order must be 'input' or 'id'")
        if row_order == "id" and not id_column:
            raise ValueError("row_order='id' requires id_column")
        self._id_column = id_column
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
                id_column=self._id_column,
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
                id_column=self._id_column,
                row_order=self._row_order,
                n_features=self._n_features,
            )
        )
