"""Real sampler, geometry and SciPy MPC integration, including release failures."""

from dataclasses import replace

import numpy as np
import pytest

from backend.experiments.conformal_mpc.synthetic import SyntheticTrajectoryModel
from backend.src.control.contracts import (
    Anchor,
    Context,
    ControlState,
    Detector,
    PreviousReference,
    ProbeBank,
    Sampling,
)
from backend.src.control.pipeline import run_control_pipeline
from backend.src.control.release import run_control_release
from backend.src.control.signatures import evaluate_signature
from backend.src.control.track import run_tracking_mpc
from backend.src.mpc.dynamics import LinearDynamics, TrackingConfig
from backend.src.mpc.geometry import Obstacle
from backend.src.mpc.solver import TrackingResult
from backend.src.repair.steering import build_constant_steering_schedule

pytestmark = pytest.mark.integration


@pytest.fixture
def setup_control():
    model = SyntheticTrajectoryModel()
    obstacle = Obstacle(tuple((2.0, 0.0) for _ in range(9)), 0.2)
    context = Context("query", 0, (0, 0), (4, 0), (obstacle,), 0.4, 8)
    probes = ProbeBank(
        np.zeros((2, 9, 2)), np.array([0.25, 0.75]), ("train",), model.revision
    )
    anchors = [
        Anchor(
            "train",
            guidance,
            "default",
            evaluate_signature(model, probes, context, guidance),
        )
        for guidance in ("upper", "direct")
    ]
    detector = Detector(probes, (anchors[0],), (anchors[1],), 0.1, 0.1)
    dynamics = LinearDynamics(np.eye(2), 0.4 * np.eye(2), np.eye(2), 0.4)
    tracking = TrackingConfig(
        dynamics,
        20 * np.eye(2),
        0.05 * np.eye(2),
        np.array([-1, -4]),
        np.array([6, 4]),
        -2 * np.ones(2),
        2 * np.ones(2),
    )
    return context, model, detector, tracking


def invoke(setup_control, **kwargs):
    context, model, detector, tracking = setup_control
    return run_control_pipeline(
        context,
        model,
        detector,
        [build_constant_steering_schedule(v) for v in (0, 0.5, 1)],
        tracking,
        guidance="direct",
        sampling=Sampling(32),
        deadline_seconds=10,
        **kwargs,
    )


def test_repaired_pipeline_releases_feasible_mpc_and_nominal_is_geometrically_rejected(
    setup_control,
):
    repaired = invoke(setup_control)
    assert repaired.released, repaired.failure
    assert repaired.repair.schedule.integrate_strength() == 1
    assert repaired.reference_clearance >= 0.15
    assert repaired.mpc.minimum_clearance >= 0.15 - 1e-7
    assert np.max(np.abs(repaired.control)) <= 2 + 1e-7
    context, _, _, tracking = setup_control
    np.testing.assert_allclose(
        repaired.mpc.states[1],
        tracking.dynamics.a @ context.state + tracking.dynamics.b @ repaired.control,
    )
    nominal = invoke(setup_control, mode="nominal")
    assert nominal.failure == "reference_collision_margin" and nominal.control is None
    joint = invoke(setup_control, mode="joint")
    assert joint.failure == "detector_abstention" and not joint.released


def test_linear_tracking_cost_and_input_bounds_match_analytic_one_step_solution():
    dynamics = LinearDynamics(np.eye(2), np.eye(2), np.eye(2), 1)
    config = TrackingConfig(
        dynamics,
        2 * np.eye(2),
        np.eye(2),
        -10 * np.ones(2),
        10 * np.ones(2),
        -np.ones(2),
        np.ones(2),
    )
    context = Context("e", 0, (0, 0), (2, 0.6), (), 1, 1)
    state = ControlState(
        context, 0, 10, lambda: 0, reference=np.array([[0, 0], [2, 0.6]])
    )
    solved = run_tracking_mpc(state, config)
    assert solved.failure is None
    np.testing.assert_allclose(solved.mpc.controls[0], [1, 0.4], atol=2e-6)


def test_infeasible_initial_state_never_releases_solver_result(setup_control):
    context, model, detector, config = setup_control
    state = ControlState(
        context, 0, 10, lambda: 0, reference=model.target(context, "upper")
    )
    impossible = replace(config, state_lower=np.array([0.5, -4]))
    result = run_tracking_mpc(state, impossible)
    assert result.failure in {"solver_failure", "solver_constraint_failure"}
    assert result.mpc is None


def test_timeout_rejects_even_when_fallback_is_available(setup_control):
    context, model, _, _ = setup_control
    calls = iter((0.0, 11.0))

    def clock():
        return next(calls, 11.0)

    previous = PreviousReference(replace(context, k=0), model.target(context, "upper"))
    result = invoke(
        (replace(context, k=1), *setup_control[1:]), clock=clock, previous=previous
    )
    assert (
        result.failure == "timeout" and result.control is None and not result.released
    )


def test_deadline_crossed_at_final_release_withholds_previously_feasible_control(
    setup_control,
):
    calls = iter((0.9, 1.1))
    state = ControlState(
        setup_control[0],
        0,
        1,
        lambda: next(calls),
        reference_clearance=1.0,
        mpc=TrackingResult(np.zeros((2, 2)), np.ones((1, 2)), 0, 0, 1, 1),
    )
    result = run_control_release(state)
    assert (
        result.failure == "timeout" and result.control is None and not result.released
    )


def test_missing_anchor_and_model_failure_are_explicit(setup_control):
    context, model, detector, config = setup_control
    incompatible = replace(
        detector, safe=(replace(detector.safe[0], compatibility="other"),)
    )
    result = invoke((context, model, incompatible, config))
    assert result.failure == "missing_anchor" and not result.released

    class BrokenModel(SyntheticTrajectoryModel):
        def drift(self, x, t, c, a):
            raise RuntimeError("simulated inference failure")

    result = invoke((context, BrokenModel(), detector, config))
    assert result.failure == "detection_failure" and result.control is None


def test_old_reference_is_revalidated_against_current_moving_obstacle(setup_control):
    context, model, detector, config = setup_control
    old = model.target(context, "upper")
    previous = PreviousReference(context, old)
    position = tuple(old[1])
    crossing = Obstacle(tuple(position for _ in range(9)), 0.2)
    current = replace(context, k=1, state=position, obstacles=(crossing,))
    result = invoke((current, model, detector, config), mode="joint", previous=previous)
    assert result.fallback_reason == "detector_abstention"
    assert result.failure == "reference_collision_margin" and not result.released


def test_rejected_encoder_cannot_supply_obstacles_for_fallback_validation(
    setup_control,
):
    context, model, detector, config = setup_control
    old = model.target(context, "upper")
    current = replace(
        context,
        k=1,
        state=tuple(old[1]),
        obstacles=(Obstacle(tuple(tuple(old[1]) for _ in range(9)), 0.2),),
    )

    class InvalidEncoder(SyntheticTrajectoryModel):
        def encode(self, obs):
            return replace(obs, obstacles=())

    result = invoke(
        (current, InvalidEncoder(), detector, config),
        mode="joint",
        previous=PreviousReference(context, old),
    )
    assert result.failure == "encoding_failure"
    assert not result.released and result.control is None


def test_valid_old_reference_can_be_released_only_after_fresh_solve(setup_control):
    context, model, detector, config = setup_control
    old = model.target(context, "upper")
    current = replace(context, k=1, state=tuple(old[1]))
    result = invoke(
        (current, model, detector, config),
        mode="joint",
        previous=PreviousReference(context, old),
    )
    assert result.released, result.failure
    assert result.fallback_reason == "detector_abstention"
    np.testing.assert_array_equal(result.mpc.states[0], current.state)
