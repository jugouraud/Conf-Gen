"""Receding-horizon velocity-disc tracking of one frozen flow-generated reference.

The reference is sampled once; at physical step k, MPC tracks y_ref[k:k+h+1].
This is an offline fixed-reference response experiment, not online flow replanning,
conformal detection/repair, bicycle dynamics, or a real-time safety controller.
"""

import time
from dataclasses import dataclass

import numpy as np

from backend.src.expert.config import PlannerSettings, VehicleSettings
from backend.src.expert.scenes import Scene
from backend.src.expert.track import track_for
from backend.src.mpc.dynamics import LinearDynamics, TrackingConfig
from backend.src.mpc.geometry import Obstacle, frozen_array, minimum_clearance
from backend.src.mpc.solver import TrackingFailure, solve_tracking


@dataclass(frozen=True)
class TrackingRollout:
    states: np.ndarray  # (K+1,2) realized positions in metres
    controls: np.ndarray  # (K,2) applied velocity commands in m/s
    predictions: tuple[np.ndarray, ...]  # each MPC's (h_k+1,2) predicted positions
    status: str  # complete, solver_failure, timeout, predicted_off_track, off_track, collision
    minimum_clearance: float  # executed swept disc/obstacle surface, metres (+inf if none)
    minimum_track_clearance: float  # executed disc/track boundary, metres


def run_tracking_rollout(
    scene: Scene,
    reference: np.ndarray,
    vehicle: VehicleSettings,
    planner: PlannerSettings,
    *,
    window: int = 8,
    seed: int = 0,
    process_noise: float = 0.0,
    deadline_seconds: float = 5.0,
) -> TrackingRollout:
    """Track physical y_ref (N+1,2), applying one MPC velocity at each dt.

    A short horizon controls SLSQP cost; its final window shrinks without retiming
    or resampling y_ref. Predicted track containment is a *release gate*, not an
    MPC track constraint. Disturbances are applied only after a released control.
    """
    y_ref = frozen_array(reference, ndim=2)
    if (scene.track is None or vehicle.kind != "velocity_disc" or vehicle.output != "xy"
            or y_ref.shape != (planner.horizon + 1, 2)
            or type(window) is not int or not 1 <= window <= planner.horizon
            or type(seed) is not int or not 0 <= seed < 2**32
            or not np.isfinite(process_noise) or process_noise < 0
            or not np.isfinite(deadline_seconds) or deadline_seconds <= 0):
        raise ValueError("Tracking requires a fixed-track velocity-disc reference and valid settings")
    if (np.linalg.norm(y_ref[0] - scene.initial_state[:2]) > 1e-4
            or np.linalg.norm(y_ref[-1] - scene.goal) > 1e-2):
        raise ValueError("Reference endpoints do not match scene context")
    track = track_for(scene.track)
    dynamics = LinearDynamics(np.eye(2), planner.dt * np.eye(2), np.eye(2), planner.dt)
    config = TrackingConfig(
        dynamics, 20.0 * np.eye(2), 0.05 * np.eye(2),
        np.asarray(vehicle.state_lower), np.asarray(vehicle.state_upper),
        np.asarray(vehicle.input_lower), np.asarray(vehicle.input_upper),
        footprint_radius=vehicle.footprint_radius,
        required_clearance=planner.required_clearance,
        tracking_uncertainty=0.0,  # No empirically calibrated tracking tube.
        max_iterations=80, max_input_norm=vehicle.max_speed,
        max_input_delta=vehicle.max_acceleration * planner.dt,
    )
    physical = np.asarray(scene.initial_state, dtype=float)
    previous = np.asarray(scene.initial_velocity, dtype=float)
    random = np.random.default_rng(seed)
    states, controls, predictions = [physical.copy()], [], []
    minimum_obstacle, minimum_track = np.inf, float(
        track.boundary_clearance(physical[None], vehicle.footprint_radius).min()
    )
    status = "complete"
    for k in range(planner.horizon):
        h = min(window, planner.horizon - k)
        times = scene.start_time + planner.dt * (k + np.arange(h + 1))
        obstacles = tuple(
            Obstacle(tuple(map(tuple, obstacle.at(times))), obstacle.radius)
            for obstacle in scene.obstacles
        )
        deadline = time.perf_counter() + deadline_seconds

        def check_deadline() -> None:
            if time.perf_counter() >= deadline:
                raise TimeoutError("MPC per-step deadline exceeded")

        try:
            solution = solve_tracking(
                physical, y_ref[k:k + h + 1], obstacles, config,
                previous_control=previous, check_deadline=check_deadline,
            )
        except TimeoutError:
            status = "timeout"
            break
        except TrackingFailure:
            status = "solver_failure"
            break
        # The generic linear MPC enforces obstacle and box constraints, not the
        # curved circuit: reject a proposed solution that exits its track tube.
        if np.min(track.boundary_clearance(solution.states, vehicle.footprint_radius)) < 0:
            status = "predicted_off_track"
            break
        control = np.asarray(solution.controls[0])
        next_state = dynamics.rollout(physical, control[None])[1]
        next_state = next_state + random.normal(0.0, process_noise, size=2)
        segment = np.stack((physical, next_state))
        current_obstacles = tuple(
            Obstacle((obstacle.centers[0], obstacle.centers[1]), obstacle.radius)
            for obstacle in obstacles
        )
        minimum_obstacle = min(minimum_obstacle, minimum_clearance(
            segment, current_obstacles, vehicle.footprint_radius
        ))
        minimum_track = min(minimum_track, float(
            np.min(track.boundary_clearance(segment, vehicle.footprint_radius))
        ))
        predictions.append(solution.states)
        controls.append(control.copy())
        states.append(next_state.copy())
        previous, physical = control, next_state
        if minimum_obstacle < 0:
            status = "collision"
            break
        if minimum_track < 0:
            status = "off_track"
            break
    return TrackingRollout(
        frozen_array(np.stack(states)),
        frozen_array(np.asarray(controls, dtype=float).reshape(-1, 2)),
        tuple(predictions), status, minimum_obstacle, minimum_track,
    )
