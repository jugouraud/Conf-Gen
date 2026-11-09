import numpy as np
import pytest
import torch

from backend.src.repair.steering import (
    RepairConstraintScores,
    build_strategy_steering_schedules,
    build_window_steering_schedule,
    calculate_normalized_deviation_profiles,
    match_schedule_intervention_budget,
    optimize_schedule_knots,
)

pytestmark = pytest.mark.unit


def test_deviation_profiles_remove_safe_baseline_and_keep_the_largest_probe():
    query = torch.tensor([0.0, 2.0, 4.0, 8.0]).reshape(4, 1, 1, 1)
    profiles = calculate_normalized_deviation_profiles(
        query,
        torch.zeros_like(query),
        {"time_points": 2, "trajectories": 2},
        {"location": [1, 2], "scale": [1, 2]},
    )
    np.testing.assert_allclose(profiles["max"], [1, 3])
    np.testing.assert_allclose(profiles["mean"], [0.5, 2])
    np.testing.assert_allclose(profiles["top10"], [1, 3])
    np.testing.assert_allclose(profiles["top25"], [1, 3])


def test_preregistered_strategies_include_fixed_peak_and_contiguous_windows():
    schedules = build_strategy_steering_schedules(
        ["R1", "R2", "R3", "R4"],
        {"midpoints": [0.25, 0.5, 0.75]},
        {
            "max": np.array([1, 5, 1]),
            "mean": np.array([1, 5, 1]),
            "top10": np.array([1, 5, 1]),
            "top25": np.array([1, 5, 1]),
        },
        {"window_widths": [0.25], "ramp_fraction": 0.2, "top_k": 1},
    )
    by_name = {schedule.name: schedule for schedule in schedules}
    assert set(by_name) == {
        "R1",
        "R2_early_w0.25",
        "R2_middle_w0.25",
        "R2_late_w0.25",
        "R3_max_w0.25",
        "R4_mean_w0.25",
        "R4_top10_w0.25",
        "R4_top25_w0.25",
    }
    assert by_name["R2_early_w0.25"](0) == 1
    assert by_name["R2_late_w0.25"](1) == 1
    assert by_name["R3_max_w0.25"](0.5) == 1
    assert by_name["R3_max_w0.25"](0.25) == 0
    assert by_name["R4_mean_w0.25"](0.5) == 1


def test_budget_matching_preserves_the_window_and_matches_total_intervention():
    shape = build_window_steering_schedule("middle", [0.5], 0.5)
    matched = match_schedule_intervention_budget(shape, 0.1)
    assert matched.integrate_strength() == pytest.approx(0.1)
    assert matched(0) == 0
    assert matched(1) == 0
    assert match_schedule_intervention_budget(shape, 1) is None


@pytest.mark.parametrize(
    "feasible,expected_status", [(True, "verified"), (False, "optimization_unresolved")]
)
def test_knot_optimizer_requires_actual_admission_before_claiming_success(
    feasible, expected_status
):
    def score(schedule):
        safe = 1 - schedule(0.5) if feasible else 2
        return RepairConstraintScores(safe, 10, 0.5, None)

    choice = optimize_schedule_knots(
        score,
        {
            "optimization_knots": 3,
            "coordinate_steps": [0.5],
            "optimization_sweeps": 2,
            "smoothness_weight": 0.1,
            "constraint_penalty": 10,
            "unsafe_margin": 0.01,
        },
    )

    assert choice.status == expected_status
    assert choice.scores.satisfies_constraints is feasible
