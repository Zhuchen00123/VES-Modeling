"""Host-computed bin packing facts; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.binpacking.context import BinVerificationContext
from ves_modeling.binpacking.data_contract import validate_solution


class BinVerifier:
    """EvidenceVerifier for bin packing solution artifacts."""

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, BinVerificationContext):
            raise TypeError("BinVerifier requires BinVerificationContext")
        payload = self._parse(raw_artifact)
        assignment = validate_solution(payload, context.contract)
        bin_totals: dict[int, float] = {}
        for bin_index, size in zip(
            assignment, context.contract.items
        ):
            bin_totals[bin_index] = bin_totals.get(bin_index, 0.0) + size
        bin_count = float(max(bin_totals) + 1)
        capacity_violation = max(
            max(0.0, total - context.contract.capacity)
            for total in bin_totals.values()
        )
        metrics = (bin_count, capacity_violation)
        for value in metrics:
            if not np.isfinite(value):
                raise ValueError("bin packing metrics must be finite")
        return Evidence(
            observations=(
                Observation(
                    value=bin_count,
                    uncertainty=0.0,
                    provenance="host:problem",
                    name="bin_count",
                ),
                Observation(
                    value=float(capacity_violation),
                    uncertainty=0.0,
                    provenance="host:problem",
                    name="capacity_violation",
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
