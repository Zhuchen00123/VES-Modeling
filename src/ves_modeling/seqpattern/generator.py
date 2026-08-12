"""Candidate generators for the sequential-pattern slice."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "seq_prefixspan.py",
    "seq_random.py",
)


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockSeqPatternGenerator:
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
        '  "patterns": [\n'
        '    {{"prefix": [<event>, ...], "suffix": [<event>, ...]}},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "At least one pattern; prefix and suffix are non-empty, disjoint "
        "and use events from train.csv only."
    )


def _draft_prompt(*, event_set_size: int) -> str:
    lines = [
        "You are mining sequential patterns from event sequences.",
        "",
        "Available files:",
        "/data/train.csv",
        "",
        f"distinct train events: {event_set_size}.",
        "",
        "You must create one complete Python program. The program must:",
        "1. Load /data/train.csv (sequence_id, step, event).",
        (
            "2. Mine sequential patterns (e.g. contiguous prefix extension "
            "from train sequences)."
        ),
        "3. Write exactly one artifact:",
        "   /output/patterns.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        (
            "Do not report or rely on self-computed confidence/lift; the "
            "host verifier independently evaluates your patterns on its own "
            "hidden sequences."
        ),
        "Never claim pattern optimality.",
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
    event_set_size: int,
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
            "Improve the executable solution (higher mean_lift, then higher "
            "mean_confidence)."
        ),
        "",
        (
            "You must still load /data/train.csv and write exactly one "
            "artifact:"
        ),
        "/output/patterns.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(),
        "",
        f"distinct train events: {event_set_size}.",
        "",
        (
            "Do not report or rely on self-computed metrics; the host "
            "independently evaluates your patterns. Never claim pattern "
            "optimality."
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


class LLMSeqPatternGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence."""

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        event_set_size: int = 0,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        self._event_set_size = event_set_size

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
            _draft_prompt(event_set_size=self._event_set_size)
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
                event_set_size=self._event_set_size,
            )
        )
