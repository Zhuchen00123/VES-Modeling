"""Host-computed Markov metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.markov.context import MarkovVerificationContext
from ves_modeling.markov.data_contract import validate_solution


class MarkovVerifier:
    """EvidenceVerifier for Markov estimation solution artifacts."""

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, MarkovVerificationContext):
            raise TypeError("MarkovVerifier requires MarkovVerificationContext")
        payload = self._parse(raw_artifact)
        estimate, confidence_interval = validate_solution(
            payload, context.contract
        )
        reference = context.reference
        absolute_error = abs(estimate - reference)
        relative_error = (
            absolute_error / abs(reference)
            if reference != 0.0
            else absolute_error
        )
        ci_coverage = 0.0
        if confidence_interval is not None:
            lo, hi = confidence_interval
            ci_coverage = 1.0 if lo <= reference <= hi else 0.0
        metrics = (absolute_error, relative_error, ci_coverage)
        for value in metrics:
            if not np.isfinite(value):
                raise ValueError("markov metrics must be finite")
        return Evidence(
            observations=(
                Observation(
                    value=float(absolute_error),
                    uncertainty=0.0,
                    provenance="host:hidden-parameters",
                    name="absolute_error",
                ),
                Observation(
                    value=float(relative_error),
                    uncertainty=0.0,
                    provenance="host:hidden-parameters",
                    name="relative_error",
                ),
                Observation(
                    value=float(ci_coverage),
                    uncertainty=0.0,
                    provenance="host:hidden-parameters",
                    name="ci_coverage",
                ),
            )
        )

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
            raise ValueError("solution.json root must be an object")
        return data
