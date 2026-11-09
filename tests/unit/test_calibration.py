"""Hand-ranked calibration examples protect finite-sample and tie boundaries."""

import math

import pytest

from backend.src.detection.calibration import (
    calculate_conformal_cutoff,
    cap_unsafe_cutoff,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "alpha,expected",
    [
        (0.19, math.inf),
        (0.2, 40.0),
        (0.4, 30.0),
        (0.8, 10.0),
    ],
)
def test_conformal_cutoff_uses_finite_sample_rank(alpha, expected):
    assert calculate_conformal_cutoff([40, 10, 30, 20], alpha) == expected


def test_conformal_cutoff_preserves_tied_scores():
    assert calculate_conformal_cutoff([1, 2, 2, 9], 0.4) == 2


@pytest.mark.parametrize(
    "scores,alpha",
    [
        ([], 0.2),
        ([[1, 2]], 0.2),
        ([float("nan")], 0.2),
        ([float("inf")], 0.2),
        ([1, 2], 0),
        ([1, 2], 1),
        ([1, 2], -0.01),
        ([1, 2], 1.01),
    ],
)
def test_conformal_cutoff_rejects_invalid_calibration(scores, alpha):
    with pytest.raises(ValueError, match="calibration scores"):
        calculate_conformal_cutoff(scores, alpha)


@pytest.mark.parametrize("cutoff,expected", [(50, 20), (15, 15)])
def test_unsafe_cap_can_only_tighten_the_fitted_cutoff(cutoff, expected):
    assert cap_unsafe_cutoff(cutoff, [40, 10, 30, 20], 0.4) == (expected, 2)


def test_unsafe_cap_rejects_an_unavailable_lower_order_statistic():
    with pytest.raises(ValueError, match="order-statistic rank"):
        cap_unsafe_cutoff(50, [10, 20, 30, 40], 0.19)
