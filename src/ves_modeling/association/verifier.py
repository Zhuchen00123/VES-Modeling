"""Host-computed association metrics; candidate self-reports are ignored."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from ves.artifact import RawArtifact
from ves.context import VerificationContext
from ves.evidence import Evidence, Observation

from ves_modeling.association.context import AssociationVerificationContext
from ves_modeling.association.data_contract import validate_rules


class AssociationVerifier:
    """EvidenceVerifier for association rule artifacts.

    The host recomputes confidence and lift on the hidden test transactions;
    rules whose antecedent never appears are skipped, and lift is clipped to
    a finite cap.  Candidate self-reported metrics are never read.
    """

    version = "0.1.0"

    def verify(
        self, raw_artifact: RawArtifact, context: VerificationContext
    ) -> Evidence:
        if not isinstance(context, AssociationVerificationContext):
            raise TypeError(
                "AssociationVerifier requires "
                "AssociationVerificationContext"
            )
        payload = self._parse(raw_artifact)
        rules = validate_rules(payload, context.contract)
        transactions = context.hidden_transactions()
        n_transactions = len(transactions)
        lift_cap = context.lift_cap
        lifts: list[float] = []
        confidences: list[float] = []
        for antecedent, consequent in rules:
            ante_set = set(antecedent)
            full_set = set(antecedent) | set(consequent)
            ante_support = 0
            both_support = 0
            consequent_support = 0
            for transaction in transactions:
                if ante_set <= transaction:
                    ante_support += 1
                    if full_set <= transaction:
                        both_support += 1
                if set(consequent) <= transaction:
                    consequent_support += 1
            if ante_support == 0:
                continue  # not evaluable
            confidence = both_support / ante_support
            support_consequent = consequent_support / n_transactions
            if support_consequent == 0.0:
                lift = lift_cap
            else:
                lift = min(confidence / support_consequent, lift_cap)
            lifts.append(float(lift))
            confidences.append(float(confidence))
        mean_lift = float(np.mean(lifts)) if lifts else 0.0
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0
        metrics = (
            mean_lift,
            mean_confidence,
            float(len(lifts)),
            float(len(rules)),
        )
        for value in metrics:
            if not np.isfinite(value):
                raise ValueError("association metrics must be finite")
        return Evidence(
            observations=(
                Observation(
                    value=mean_lift,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="mean_lift",
                ),
                Observation(
                    value=mean_confidence,
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="mean_confidence",
                ),
                Observation(
                    value=float(len(lifts)),
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="evaluable_rule_count",
                ),
                Observation(
                    value=float(len(rules)),
                    uncertainty=0.0,
                    provenance="host:hidden-test",
                    name="rule_count",
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
            raise ValueError("rules.json root must be an object")
        return data
