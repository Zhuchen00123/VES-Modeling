"""Candidate generators for the classification slice.

- MockClassificationGenerator: returns trusted fixture code (no LLM).
- LLMClassificationGenerator: provider-neutral adapter over an
  OpenAI-compatible client, with draft/improve prompts that follow the active
  classification data contract (host-fixed class order, label + probabilities
  artifact, argmax tie-first).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "logistic_balanced.py",
    "random_forest_balanced.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockClassificationGenerator:
    """Returns preset trusted candidates for draft/improve steps.

    draft(0) -> logistic_balanced, draft(1) -> random_forest_balanced,
    improve -> random_forest_balanced (cycled for later improves).
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
            '    {{"id": <test row id>, "label": <class label>, '
            '"probabilities": [<p0>, <p1>, ...]}},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Every test row must appear exactly once, using the id from "
            "test_features.csv. The id is a row identifier, not a model "
            "feature. 'label' must equal the index of the largest "
            "probability in the host class order; ties choose the first "
            "class."
        )
    return (
        "{{\n"
        '  "predictions": [\n'
        '    {{"label": <class label>, '
        '"probabilities": [<p0>, <p1>, ...]}},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Predictions must be aligned to test_features.csv row order. "
        "'label' must equal the index of the largest probability in the "
        "host class order; ties choose the first class."
    )


def _classes_prompt(classes: tuple[Any, ...]) -> str:
    return (
        "Host-fixed class order (labels and probability columns must follow "
        "this exact order):\n"
        + ", ".join(repr(value) for value in classes)
    )


def _draft_prompt(
    *,
    label_column: str,
    id_column: str | None,
    row_order: str,
    classes: tuple[Any, ...],
    n_features: int,
) -> str:
    lines = [
        "You are solving a classification task.",
        "",
        "Available files:",
        "/data/train.csv",
        "/data/test_features.csv",
        "",
        "train.csv contains the label column:",
        label_column,
        "",
        _classes_prompt(classes),
        "",
        (
            f"train.csv has {n_features} feature columns; test_features.csv "
            "has the same features without the label."
        ),
        "",
    ]
    if id_column:
        lines += [
            "test_features.csv contains the row id column:",
            id_column,
            "",
            "The id column must NOT be used as a model feature.",
            "",
        ]
    lines += [
        "You must create one complete Python program. The program must:",
        "1. Load /data/train.csv.",
        "2. Load /data/test_features.csv.",
        (
            "3. Train a classifier (use class_weight='balanced' or equivalent "
            "when classes are imbalanced)."
        ),
        "4. Predict every row in test_features.csv.",
        "5. Write exactly one artifact:",
        "   /output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(id_column, row_order),
        "",
        (
            "Probabilities must be non-negative, finite, each within [0, 1] "
            "and sum to 1 per row."
        ),
        (
            "Do not tune any decision threshold on held-out labels; the label "
            "must always be the host-order argmax with ties choosing the "
            "first class."
        ),
        "",
        "You cannot access hidden evaluation labels.",
        "Do not report or rely on a self-computed test score.",
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
    label_column: str,
    id_column: str | None,
    row_order: str,
    classes: tuple[Any, ...],
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
            "Improve the executable solution. You may change preprocessing, "
            "feature engineering, estimator, hyperparameters, class "
            "balancing, or ensembling."
        ),
        "",
        (
            "You must still load /data/train.csv and /data/test_features.csv, "
            "predict every row, and write exactly one artifact:"
        ),
        "/output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(id_column, row_order),
        "",
        "The label column is " + label_column + ".",
        _classes_prompt(classes),
    ]
    if id_column:
        lines += [
            "The id column " + id_column + " must NOT be used as a model "
            "feature.",
        ]
    lines += [
        "",
        (
            "Probabilities must be non-negative, finite, each within [0, 1] "
            "and sum to 1 per row. Do not tune any decision threshold; the "
            "label must always be the host-order argmax with ties choosing "
            "the first class."
        ),
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


class LLMClassificationGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence.

    ``fallback_code`` (optional) is used when the LLM client fails after its
    internal retries, so an infrastructure outage degrades to a real
    executable candidate instead of crashing the search.  The fallback is
    still verified by the host; no score is ever fabricated.
    """

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        label_column: str = "target",
        id_column: str | None = None,
        row_order: str = "input",
        classes: tuple[Any, ...] = (),
        n_features: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        if row_order not in ("input", "id"):
            raise ValueError("row_order must be 'input' or 'id'")
        if row_order == "id" and not id_column:
            raise ValueError("row_order='id' requires id_column")
        self._label_column = label_column
        self._id_column = id_column
        self._row_order = row_order
        self._classes = tuple(classes)
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
                label_column=self._label_column,
                id_column=self._id_column,
                row_order=self._row_order,
                classes=self._classes,
                n_features=self._n_features,
            )
        )

    def improve(
        self, problem: VerifiedProblem, anchor: VerifiedCandidate
    ) -> str:
        evidence = {
            observation.name: observation.value
            for observation in anchor.record.evidence
            if not observation.name.startswith("confusion_")
        }
        return self._complete(
            _improve_prompt(
                anchor.program,
                evidence,
                label_column=self._label_column,
                id_column=self._id_column,
                row_order=self._row_order,
                classes=self._classes,
            )
        )
