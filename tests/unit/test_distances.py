"""Geometric examples and invariants independent of the Gram implementation."""

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from backend.src.detection.calibration import (
    aggregate_anchor_distances,
    fit_distance_rule,
    resolve_anchor_scoring_settings,
)
from backend.src.fields.distances import calculate_pairwise_euclidean_distances

pytestmark = pytest.mark.unit


def test_distances_match_a_three_four_five_triangle():
    actual = calculate_pairwise_euclidean_distances(
        torch.tensor([[0, 0], [3, 4]]), torch.tensor([[0, 0], [6, 8]])
    )
    torch.testing.assert_close(
        actual, torch.tensor([[0.0, 10.0], [5.0, 5.0]], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "queries,references",
    [
        (torch.zeros(2), torch.zeros(1, 2)),
        (torch.zeros(1, 2), torch.zeros(1, 3)),
        (torch.zeros(1, 2), torch.zeros(0, 2)),
    ],
)
def test_distances_reject_incompatible_feature_matrices(queries, references):
    with pytest.raises(ValueError, match="matching feature matrices"):
        calculate_pairwise_euclidean_distances(queries, references)


@given(
    st.lists(
        st.tuples(st.integers(-100, 100), st.integers(-100, 100)),
        min_size=1,
        max_size=8,
    )
)
def test_pairwise_distances_are_symmetric_nonnegative_and_zero_on_self(points):
    values = torch.tensor(points, dtype=torch.float64)
    distances = calculate_pairwise_euclidean_distances(values, values)

    torch.testing.assert_close(distances, distances.T)
    assert torch.all(distances >= 0)
    # The Gram formula takes a square root after cancellation. For coordinates
    # bounded by 100, 5e-6 covers FP64 reduction error in squared distances.
    torch.testing.assert_close(
        distances.diagonal(),
        torch.zeros(len(points), dtype=torch.float64),
        atol=5e-6,
        rtol=0,
    )


@pytest.mark.parametrize("method,expected", [("euclidean", 2), ("anchor_static", 4)])
def test_anchor_rule_distinguishes_nearest_from_uniform_mean(method, expected):
    assert aggregate_anchor_distances([2, 4, 6], {"method": method}) == expected


def test_adaptive_bandwidth_uses_only_distinct_anchor_pairs():
    spec = fit_distance_rule("anchor_adaptive", [[0, 2, 4], [2, 0, 6], [4, 6, 0]], 0.5)
    assert spec["bandwidth"] == 2
    assert spec["anchor_distance_median"] == 4


def test_adaptive_weighting_retains_nearest_anchor_when_other_weights_underflow():
    actual = aggregate_anchor_distances(
        [1000, 2000], {"method": "anchor_adaptive", "bandwidth": 1}
    )
    assert actual == pytest.approx(1000)


@pytest.mark.parametrize("values", [[], [-1, 2], [np.nan], [np.inf]])
def test_anchor_scores_reject_invalid_distances(values):
    with pytest.raises(ValueError, match="finite nonnegative"):
        aggregate_anchor_distances(values, {"method": "anchor_static"})


def test_conflicting_legacy_and_current_scoring_settings_are_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_anchor_scoring_settings(
            {"scoring": {"method": "euclidean", "anchor_weighting": "uniform"}}
        )
