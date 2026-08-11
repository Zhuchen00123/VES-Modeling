"""LLM generator fallback: infrastructure failure degrades to a real candidate."""

from __future__ import annotations

import pytest
from ves.artifact import RawArtifact
from ves.evidence import Evidence, Observation
from ves.record import Candidate, VerificationRecord, VerifiedCandidate

from ves_modeling.regression.generator import LLMRegressionGenerator


class FailingLlm:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("reasoning-only: boom")


def make_anchor() -> VerifiedCandidate:
    record = VerificationRecord(
        verifier_version="0.1.0",
        context_id="regression:unit",
        context_fingerprint="f" * 64,
        evidence=Evidence(
            observations=(
                Observation(value=42.5, provenance="host", name="rmse"),
                Observation(value=33.0, provenance="host", name="mae"),
            )
        ),
    )
    return VerifiedCandidate(
        candidate=Candidate.draft("CODE"),
        artifact=RawArtifact(name="predictions.json", content="{}", producer="test"),
        record=record,
    )


def test_fallback_used_on_llm_failure() -> None:
    generator = LLMRegressionGenerator(FailingLlm(), fallback_code="FALLBACK")
    assert generator.draft(None, 0) == "FALLBACK"
    assert generator.improve(None, make_anchor()) == "FALLBACK"


def test_no_fallback_reraises() -> None:
    generator = LLMRegressionGenerator(FailingLlm())
    with pytest.raises(RuntimeError, match="reasoning-only"):
        generator.draft(None, 0)
