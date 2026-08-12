"""Host-computed sequential-pattern metrics; self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.seqpattern.context import SeqPatternVerificationContext
from ves_modeling.seqpattern.data_contract import (
    evaluate_patterns,
    validate_patterns,
)


class SeqPatternVerifier:
    """EvidenceVerifier for sequential-pattern artifacts."""

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, SeqPatternVerificationContext):
            raise TypeError(
                "SeqPatternVerifier requires SeqPatternVerificationContext"
            )
        payload = self._parse(raw_artifact)
        patterns = validate_patterns(
            payload, event_set=context.event_set
        )
        metrics = evaluate_patterns(
            patterns, context.hidden_sequences()
        )
        observations = (
            Observation(
                value=metrics["mean_lift"],
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="mean_lift",
            ),
            Observation(
                value=metrics["mean_confidence"],
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="mean_confidence",
            ),
            Observation(
                value=metrics["evaluable_pattern_count"],
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="evaluable_pattern_count",
            ),
            Observation(
                value=metrics["pattern_count"],
                uncertainty=0.0,
                provenance="host:hidden-test",
                name="pattern_count",
            ),
        )
        for observation in observations:
            if not np.isfinite(observation.value):
                raise ValueError("sequential-pattern metrics must be finite")
        return Evidence(observations=observations)

    @staticmethod
    def _parse(raw_artifact: RawArtifact) -> dict[str, Any]:
        text = (
            raw_artifact.content.decode("utf-8")
            if isinstance(raw_artifact.content, bytes)
            else raw_artifact.content
        )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise ValueError("patterns.json root must be an object")
        return data
