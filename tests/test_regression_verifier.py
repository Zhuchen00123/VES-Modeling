"""R1 acceptance: RegressionVerifier + ArtifactContract behaviour."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ves.artifact import ArtifactContract, RawArtifact
from ves.evidence import Evidence


def observation_map(evidence: Evidence) -> dict[str, float]:
    return {o.name: o.value for o in evidence}


def test_correct_predictions_verified(verifier, context, labels, make_artifact):
    artifact = make_artifact({"predictions": labels.tolist()})
    evidence = verifier.verify(artifact, context)
    values = observation_map(evidence)
    assert set(values) == {"rmse", "mae"}
    assert values["rmse"] == pytest.approx(0.0)
    assert values["mae"] == pytest.approx(0.0)
    assert all(o.provenance == "host:hidden-test" for o in evidence)


def test_missing_predictions_fails(verifier, context, make_artifact):
    artifact = make_artifact({})
    with pytest.raises(ValueError, match="predictions"):
        verifier.verify(artifact, context)


def test_wrong_prediction_count_rejected(verifier, context, make_artifact):
    artifact = make_artifact({"predictions": [1.0, 2.0]})
    with pytest.raises(ValueError, match="count"):
        verifier.verify(artifact, context)


@pytest.mark.parametrize(
    "bad",
    [
        [1.0, float("nan"), 3.0, 4.0, 5.0],
        [1.0, float("inf"), 3.0, 4.0, 5.0],
        [1.0, -float("inf"), 3.0, 4.0, 5.0],
    ],
)
def test_non_finite_rejected(verifier, context, bad, make_artifact):
    artifact = make_artifact({"predictions": bad})
    with pytest.raises(ValueError, match="finite"):
        verifier.verify(artifact, context)


def test_bool_and_non_numeric_rejected(verifier, context, make_artifact):
    for bad in ([True, 2.0, 3.0, 4.0, 5.0], [1.0, "x", 3.0, 4.0, 5.0]):
        artifact = make_artifact({"predictions": bad})
        with pytest.raises(ValueError):
            verifier.verify(artifact, context)


def test_contract_checks_generic_shape(verifier, context):
    contract = ArtifactContract(
        filename="predictions.json",
        media_type="application/json",
        required_fields=("predictions",),
    )
    artifact = RawArtifact(
        name="predictions.json", content=json.dumps({"predictions": [1.0]}), producer="test"
    )
    assert contract.validate(artifact) == []
    missing = RawArtifact(name="predictions.json", content="{}", producer="test")
    assert contract.validate(missing) == ["missing required field 'predictions'"]


def test_verifier_reproducible(verifier, context, labels, make_artifact):
    e1 = verifier.verify(make_artifact({"predictions": labels.tolist()}), context)
    e2 = verifier.verify(make_artifact({"predictions": labels.tolist()}), context)
    assert e1 == e2
    assert context.fingerprint() == context.fingerprint()


def test_rmse_mae_values(verifier, context, labels, make_artifact):
    shifted = labels + np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    evidence = verifier.verify(make_artifact({"predictions": shifted.tolist()}), context)
    values = observation_map(evidence)
    assert values["rmse"] == pytest.approx(1.0)
    assert values["mae"] == pytest.approx(1.0)
