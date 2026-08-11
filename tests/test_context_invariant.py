"""C3: RegressionVerificationContext construction invariants."""

from __future__ import annotations

import numpy as np
import pytest

from ves_modeling.regression.context import RegressionVerificationContext


def test_empty_labels_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RegressionVerificationContext(np.array([]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_labels_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        RegressionVerificationContext(np.array([1.0, bad, 3.0]))


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_expected_count_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        RegressionVerificationContext(np.array([1.0, 2.0, 3.0]), expected_count=bad)


def test_expected_count_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="match"):
        RegressionVerificationContext(np.array([1.0, 2.0, 3.0]), expected_count=4)


def test_default_expected_count_matches_size() -> None:
    context = RegressionVerificationContext(np.array([1.0, 2.0, 3.0]))
    assert context.expected_count == 3
