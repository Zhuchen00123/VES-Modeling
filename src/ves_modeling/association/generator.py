"""Candidate generators for the association slice."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "apriori_rules.py",
    "random_rules.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockAssociationGenerator:
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
        '  "rules": [\n'
        '    {{"antecedent": ["<item>", ...], '
        '"consequent": ["<item>", ...]}},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Provide at least one rule; antecedent and consequent must be "
        "non-empty, disjoint, use items from train.csv, and rules must be "
        "unique."
    )


def _draft_prompt(*, n_items: int, n_transactions: int) -> str:
    lines = [
        "You are solving an association rule mining task.",
        "",
        "Available files:",
        "/data/train.csv",
        "",
        (
            "train.csv is a transaction long table (transaction_id, "
            "item)."
        ),
        (
            f"train item count: {n_items}. train transaction count: "
            f"{n_transactions}."
        ),
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/train.csv.",
        (
            "2. Mine association rules (e.g. a simplified Apriori on "
            "frequent itemsets)."
        ),
        "3. Write exactly one artifact:",
        "   /output/rules.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        "You cannot access hidden evaluation transactions.",
        "Do not report or rely on a self-computed score.",
        "The host verifier will independently evaluate the rules.",
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
    n_items: int,
    n_transactions: int,
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
            "Improve the executable solution (higher mean lift/confidence "
            "on the hidden transactions)."
        ),
        "",
        (
            "You must still load /data/train.csv and write exactly one "
            "artifact:"
        ),
        "/output/rules.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        (
            f"train item count: {n_items}. train transaction count: "
            f"{n_transactions}."
        ),
        "",
        (
            "You cannot access hidden evaluation transactions. Do not "
            "report a self-computed score; the host verifier independently "
            "evaluates the rules."
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


class LLMAssociationGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        n_items: int = 0,
        n_transactions: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._n_items = n_items
        self._n_transactions = n_transactions

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
                n_items=self._n_items, n_transactions=self._n_transactions
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
                n_items=self._n_items,
                n_transactions=self._n_transactions,
            )
        )
