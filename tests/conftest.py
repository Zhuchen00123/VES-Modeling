"""Shared fixtures for regression tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ves.artifact import RawArtifact

from ves_modeling.regression.context import RegressionVerificationContext
from ves_modeling.regression.verifier import RegressionVerifier


@pytest.fixture
def labels() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def context(labels: np.ndarray) -> RegressionVerificationContext:
    return RegressionVerificationContext(
        labels, dataset_name="unit", expected_count=int(labels.size)
    )


@pytest.fixture
def verifier() -> RegressionVerifier:
    return RegressionVerifier()


@pytest.fixture
def make_artifact():
    def _make(payload) -> RawArtifact:
        return RawArtifact(
            name="predictions.json",
            content=json.dumps(payload),
            producer="test",
        )

    return _make
