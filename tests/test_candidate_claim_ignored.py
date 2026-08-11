"""R3: candidate self-reported metrics must never affect judgment."""

from __future__ import annotations

import numpy as np
import pytest


def test_host_rmse_differs_from_claimed(verifier, context, labels, make_artifact):
    artifact = make_artifact(
        {
            "predictions": np.zeros(5).tolist(),
            "claimed_rmse": 0.000001,
            "claimed_mae": 0.000001,
        }
    )
    evidence = verifier.verify(artifact, context)
    values = {o.name: o.value for o in evidence}
    assert values["rmse"] == pytest.approx(float(np.sqrt(np.mean(labels**2))))
    assert values["rmse"] > 1.0
    assert values["rmse"] != pytest.approx(0.000001)


def test_cheating_candidate_via_pipeline_falls_to_host_truth(
    verifier, context, labels, make_artifact
):
    """End-to-end: pipeline verdict uses host-computed evidence only."""
    from ves.artifact import ArtifactContract
    from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
    from ves.problem import VerificationPipeline, VerifiedProblem

    problem = VerifiedProblem(
        contract=ArtifactContract(
            filename="predictions.json",
            media_type="application/json",
            required_fields=("predictions",),
        ),
        context_factory=lambda: context,
        verifier=verifier,
        judge_spec=JudgeSpec(
            objectives=(ObjectiveSpec(observation="rmse", direction=Direction.MINIMIZE),),
            gates=(Gate(name="finite", observation="rmse"),),
        ),
    )
    pipeline = VerificationPipeline(problem)
    result = pipeline.verify(
        make_artifact(
            {
                "predictions": np.zeros(5).tolist(),
                "claimed_rmse": 0.000001,
            }
        )
    )
    assert result.status.value == "verified"
    values = {o.name: o.value for o in result.evidence}
    assert values["rmse"] == pytest.approx(float(np.sqrt(np.mean(labels**2))))
    assert result.record.context_fingerprint == context.fingerprint()
