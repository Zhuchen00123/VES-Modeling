"""Candidate generators for the regression slice.

- MockRegressionGenerator: returns trusted fixture code (no LLM).
- LLMRegressionGenerator: provider-neutral adapter over an OpenAI-compatible
  client (``complete(prompt) -> str``), with draft/improve prompts.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

MOCK_FIXTURES = (
    "linear_regression.py",
    "random_forest.py",
    "gradient_boosting.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockRegressionGenerator:
    """Returns preset trusted candidates for draft/improve steps.

    draft(0) -> linear_regression, draft(1) -> random_forest,
    improve -> gradient_boosting (cycled for later improves).
    """

    def __init__(
        self,
        fixture_dir: Path,
        *,
        drafts: tuple[str, ...] = MOCK_FIXTURES[:2],
        improves: tuple[str, ...] = MOCK_FIXTURES[2:],
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


_DRAFT_PROMPT = """\
You are solving a tabular regression task.

Available files:
/data/train.csv
/data/test_features.csv

train.csv contains the target column:
target

You must create one complete Python program. The program must:
1. Load /data/train.csv.
2. Load /data/test_features.csv.
3. Train a regression model.
4. Predict every row in test_features.csv.
5. Write exactly one artifact:
   /output/predictions.json

Artifact format:
{{
  "predictions": [...]
}}

You cannot access hidden evaluation labels.
Do not report or rely on a self-computed test score.
The host verifier will independently evaluate the predictions.

Allowed libraries: numpy, pandas, scikit-learn.
Do not use pip install, curl, wget, or network access.

Your response must contain the complete solution.py only.
Do not wrap the code in markdown fences.
"""

_IMPROVE_PROMPT = """\
Previous candidate:

{code}

Host-verified evidence (independently recomputed by the host):

RMSE: {rmse:.6f}
MAE: {mae:.6f}

Improve the executable solution. You may change preprocessing, feature
engineering, estimator, hyperparameters, or ensembling.

You must still load /data/train.csv and /data/test_features.csv, predict every
row, and write exactly one artifact:
/output/predictions.json
(format: {{"predictions": [...]}})

You cannot access hidden evaluation labels. Do not report a self-computed
score; the host verifier independently evaluates the predictions.

Allowed libraries: numpy, pandas, scikit-learn.
Do not use pip install, curl, wget, or network access.

Return the complete solution.py only. Do not wrap the code in markdown fences.
"""


class LLMRegressionGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence.

    ``fallback_code`` (optional) is used when the LLM client fails after its
    internal retries, so an infrastructure outage degrades to a real executable
    candidate instead of crashing the search.  The fallback is still verified
    by the host; no score is ever fabricated.
    """

    def __init__(
        self, llm: LlmClient, fallback_code: str | None = None
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code

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
        return self._complete(_DRAFT_PROMPT)

    def improve(
        self, problem: VerifiedProblem, anchor: VerifiedCandidate
    ) -> str:
        evidence = anchor.record.evidence
        rmse = _observation(evidence, "rmse")
        mae = _observation(evidence, "mae")
        return self._complete(
            _IMPROVE_PROMPT.format(
                code=anchor.program,
                rmse=rmse,
                mae=mae,
            )
        )


def _observation(evidence, name: str) -> float:
    for observation in evidence:
        if observation.name == name:
            if not math.isfinite(observation.value):
                raise ValueError(
                    f"non-finite anchor observation {name!r} cannot be used"
                )
            return observation.value
    raise ValueError(f"missing observation {name!r} in anchor evidence")
