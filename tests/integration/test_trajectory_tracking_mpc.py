"""Frozen y_ref -> actual constrained velocity-disc MPC -> realized state and UI trace."""

import numpy as np
import pytest

from backend.experiments.trajectory_flow.track import run_tracking_rollout
from backend.src.expert.config import PlannerSettings, VehicleSettings
from backend.src.expert.scenes import Scene, TimedObstacle
from backend.src.expert.track import TrackSpec, track_for
from backend.src.mpc.dynamics import LinearDynamics, TrackingConfig
from backend.src.mpc.solver import solve_tracking
from frontend.mpc_browser import render_tracking_svg

pytestmark = pytest.mark.integration


def test_racing_reference_is_tracked_with_receding_horizon_and_physical_limits():
    track_spec = TrackSpec()
    track = track_for(track_spec)
    reference = track.point(np.linspace(0.0, 0.6, 9))
    far_obstacle = np.asarray([0.0, 0.0])
    obstacle = TimedObstacle("obstacle", np.array([0.0, 3.2]),
                             np.stack((far_obstacle, far_obstacle)), 0.15)
    scene = Scene(
        "s", "g", "track", (-10, 10, -10, 10), 0.0,
        tuple(reference[0]), (0.0, 0.0), tuple(reference[-1]), (obstacle,), track=track_spec,
    )
    result = run_tracking_rollout(
        scene, reference, VehicleSettings(), PlannerSettings(horizon=8, dt=0.4),
        window=4, seed=4,
    )
    assert result.status == "complete"
    assert result.states.shape == (9, 2)
    assert result.controls.shape == (8, 2)
    assert len(result.predictions) == 8
    assert result.predictions[0].shape == (5, 2)
    assert result.predictions[-1].shape == (2, 2)
    np.testing.assert_allclose(np.diff(result.states, axis=0), result.controls * 0.4,
                               atol=1e-9)
    assert np.max(np.linalg.norm(result.controls, axis=1)) <= 1.5 + 1e-6
    change = np.diff(np.vstack((scene.initial_velocity, result.controls)), axis=0)
    assert np.max(np.linalg.norm(change, axis=1)) <= 1.5 * 0.4 + 1e-6
    assert result.minimum_track_clearance > 0
    assert np.linalg.norm(result.states[-1] - reference[-1]) < 0.2


def test_mpc_speed_and_slew_limits_reject_invalid_previous_control():
    dynamics = LinearDynamics(np.eye(2), np.eye(2), np.eye(2), 1.0)
    tracking = TrackingConfig(
        dynamics, 20 * np.eye(2), 0.05 * np.eye(2),
        -10 * np.ones(2), 10 * np.ones(2),
        -10 * np.ones(2), 10 * np.ones(2),
        max_input_norm=0.5, max_input_delta=0.2,
    )
    with pytest.raises(ValueError, match="previous_control"):
        solve_tracking(np.zeros(2), np.array([[0.0, 0.0], [1.0, 0.0]]),
                       (), tracking, check_deadline=lambda: None)
    solution = solve_tracking(
        np.zeros(2), np.array([[0.0, 0.0], [1.0, 0.0]]),
        (), tracking, check_deadline=lambda: None, previous_control=np.zeros(2),
    )
    assert np.linalg.norm(solution.controls[0]) <= 0.2 + 1e-6
    np.testing.assert_allclose(solution.states[1], solution.controls[0], atol=1e-9)


def test_mpc_trajectory_shape_and_svg_reject_invalid_inputs():
    track = track_for(TrackSpec())
    reference = track.point(np.linspace(0.0, 0.6, 9))
    scene = Scene("s", "g", "track", (-10, 10, -10, 10), 0.0,
                  tuple(reference[0]), (0, 0), tuple(reference[-1]), (), track=TrackSpec())
    with pytest.raises(ValueError, match="endpoints"):
        changed = reference.copy()
        changed[0] += 0.5
        run_tracking_rollout(scene, changed, VehicleSettings(),
                             PlannerSettings(horizon=8, dt=0.4))
    svg = render_tracking_svg("<svg></svg>", reference[:3], (reference[:2],), 0, 0.15)
    assert 'data-kind="vehicle"' in svg
    assert 'data-kind="mpc-prediction"' in svg
    assert 'data-kind="executed"' in svg
    with pytest.raises(ValueError, match="visualization"):
        render_tracking_svg("<svg></svg>", reference[:2], (), 5, 0.15)
