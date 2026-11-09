import pytest
from hypothesis import given
from hypothesis import strategies as st

from backend.src.repair.steering import (
    RepairConstraintScores,
    SteeringSchedule,
    build_constant_steering_schedule,
    search_schedule_strength,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "safe,unsafe,unsafe_cutoff,admitted",
    [
        (0.99, 2.01, 2, True),
        (1, 2.01, 2, True),
        (1.01, 2.01, 2, False),
        (1, 2, 2, False),
        (1, 1.99, 2, False),
        (1, 0, None, True),
        (1.01, 0, None, False),
    ],
)
def test_admission_includes_safe_boundary_and_excludes_unsafe_boundary(
    safe, unsafe, unsafe_cutoff, admitted
):
    scores = RepairConstraintScores(safe, unsafe, 1, unsafe_cutoff)
    assert scores.satisfies_constraints is admitted


@pytest.mark.parametrize("time,expected", [(0, 0), (0.25, 0.15625), (0.5, 0.5), (1, 1)])
def test_schedule_interpolates_smoothly_between_bounded_knots(time, expected):
    schedule = SteeringSchedule("ramp", (0, 1), (0, 1))
    assert schedule(time) == pytest.approx(expected)


@pytest.mark.parametrize(
    "times,strengths",
    [
        ((0, 0.5), (0, 1)),
        ((0, 0, 1), (0, 1, 0)),
        ((0, 1), (-0.01, 1)),
        ((0, 1), (0, 1.01)),
        ((0, 1), (0, float("nan"))),
        ((0, 1), (0,)),
    ],
)
def test_schedule_rejects_invalid_knots(times, strengths):
    with pytest.raises(ValueError, match="Schedules require"):
        SteeringSchedule("invalid", times, strengths)


@pytest.mark.parametrize("time", [-0.001, 1.001, float("nan")])
def test_schedule_rejects_times_outside_the_flow_domain(time):
    with pytest.raises(ValueError, match="Schedule time"):
        build_constant_steering_schedule()(time)


@given(start=st.floats(0, 1), end=st.floats(0, 1), time=st.floats(0, 1))
def test_schedule_never_overshoots_its_endpoint_strengths(start, end, time):
    schedule = SteeringSchedule("bounded", (0, 1), (start, end))
    assert min(start, end) - 1e-15 <= schedule(time) <= max(start, end) + 1e-15


def test_ramp_has_half_a_unit_of_intervention():
    schedule = SteeringSchedule("ramp", (0, 1), (0, 1))
    assert schedule.integrate_strength() == pytest.approx(0.5)


def test_strength_search_returns_an_admitted_schedule_near_the_known_boundary():
    def score(schedule):
        return RepairConstraintScores(1 - schedule(0.5), 10, 0.75, None)

    choice = search_schedule_strength(
        build_constant_steering_schedule(1),
        score,
        {"grid_points": 9, "refinement_steps": 12},
    )

    assert choice.status == "verified"
    assert choice.scores.satisfies_constraints
    assert choice.schedule(0.5) == pytest.approx(0.25, abs=0.0001)


def test_infeasible_search_returns_an_unchanged_unverified_choice():
    def score(schedule):
        return RepairConstraintScores(2, 0, 1, None)

    choice = search_schedule_strength(
        build_constant_steering_schedule(1),
        score,
        {"grid_points": 5, "refinement_steps": 4},
    )

    assert choice.status == "no_feasible_grid_interval"
    assert not choice.has_intervention
    assert not choice.scores.satisfies_constraints
