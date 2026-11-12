"""Independent final-export checks; never trusts the optimizer's success flag."""

import numpy as np

from backend.src.expert.config import ExpertConfig, Guidance
from backend.src.expert.geometry import (
    boundary_clearances,
    passage_laterals,
    timed_clearances,
)
from backend.src.expert.scenes import Scene
from backend.src.expert.vehicle import Vehicle


def validate_initial_scene(scene: Scene, config: ExpertConfig) -> list[str]:
    v, p = config.vehicle, config.planner
    vehicle = Vehicle(v, p.dt)
    state = np.asarray(scene.initial_state)
    if state.shape != (vehicle.state_dimension,):
        return ["initial_state_dimension"]
    reasons = []
    if np.any(state < v.state_lower) or np.any(state > v.state_upper):
        reasons.append("initial_state_bounds")
    if np.linalg.norm(np.asarray(scene.goal) - state[:2]) <= p.goal_tolerance:
        reasons.append("goal_already_reached")
    if (
        np.min(
            boundary_clearances(
                np.stack((state[:2], scene.goal)), scene.bounds, v.footprint_radius
            )
        )
        < 0
    ):
        reasons.append("initial_or_goal_outside_map")
    if scene.goal_heading is not None and v.kind != "bicycle":
        reasons.append("heading_goal_without_heading_state")
    if v.kind == "velocity_disc":
        velocity = np.asarray(scene.initial_velocity)
        if (
            np.linalg.norm(velocity) > v.max_speed
            or np.any(velocity < v.input_lower)
            or np.any(velocity > v.input_upper)
        ):
            reasons.append("initial_velocity_limits")
    else:
        if state[3] < 0 or state[3] > v.max_speed or abs(state[4]) > v.max_steering:
            reasons.append("initial_bicycle_limits")
        expected_velocity = state[3] * np.array([np.cos(state[2]), np.sin(state[2])])
        if not np.allclose(
            expected_velocity,
            scene.initial_velocity,
            atol=p.validation_tolerance,
            rtol=0,
        ):
            reasons.append("initial_velocity_inconsistent")
    for obstacle in scene.obstacles:
        try:
            centers = obstacle.at(
                np.array([scene.start_time, scene.start_time + p.horizon * p.dt])
            )
        except ValueError:
            reasons.append("obstacle_time_coverage")
            continue
        if (
            np.linalg.norm(state[:2] - centers[0])
            < v.footprint_radius + obstacle.radius + p.required_clearance
        ):
            reasons.append("initial_obstacle_clearance")
    return sorted(set(reasons))


def _positive_max(values: np.ndarray) -> float:
    return max(0.0, float(np.max(values, initial=0)))


def run_expert_validation(
    scene: Scene,
    guidance: Guidance,
    config: ExpertConfig,
    states: object,
    controls: object,
    timestamps: object,
    reference: object,
) -> dict:
    """Validate serialized target semantics independently of planning_constraints.

    Returns finite diagnostics (None for obstacle-free clearance) and reasons.
    No optimizer status or previously stored validation can grant acceptance.
    """
    p, v = config.planner, config.vehicle
    vehicle = Vehicle(v, p.dt)
    try:
        states, controls, times, reference = [
            np.asarray(x, dtype=float)
            for x in (states, controls, timestamps, reference)
        ]
        shapes = (
            states.shape == (p.horizon + 1, vehicle.state_dimension)
            and controls.shape == (p.horizon, 2)
            and times.shape == (p.horizon + 1,)
            and reference.shape == (p.horizon + 1, vehicle.output_dimension)
        )
        if not shapes or not all(
            np.isfinite(x).all() for x in (states, controls, times, reference)
        ):
            return {
                "valid": False,
                "reasons": ["reference_dimensions_or_nonfinite_values"],
            }
        expected_times = scene.start_time + np.arange(p.horizon + 1) * p.dt
        time_error = float(np.max(np.abs(times - expected_times)))
        if time_error > 1e-10 or np.any(np.diff(times) <= 0):
            return {
                "valid": False,
                "reasons": ["physical_timestamps_mismatch"],
                "timestamp_residual": time_error,
            }
        reasons = validate_initial_scene(scene, config)
        if reasons:
            return {"valid": False, "reasons": reasons}
        start_error = float(np.max(np.abs(states[0] - scene.initial_state)))
        dynamics_error = max(
            float(np.max(np.abs(states[i + 1] - vehicle.step(states[i], controls[i]))))
            for i in range(p.horizon)
        )
        reference_error = float(np.max(np.abs(reference - vehicle.output(states))))
        state_violation = max(
            _positive_max(np.asarray(v.state_lower) - states),
            _positive_max(states - v.state_upper),
        )
        input_violation = max(
            _positive_max(np.asarray(v.input_lower) - controls),
            _positive_max(controls - v.input_upper),
        )
        if v.kind == "velocity_disc":
            speeds = np.linalg.norm(controls, axis=1)
            accelerations = (
                np.linalg.norm(
                    np.diff(np.vstack((scene.initial_velocity, controls)), axis=0),
                    axis=1,
                )
                / p.dt
            )
            steering_violation = 0.0
            heading_error = None
        else:
            speeds = np.abs(states[:, 3])
            accelerations = np.abs(controls[:, 0])
            steering_violation = max(
                _positive_max(np.abs(states[:, 4]) - v.max_steering),
                _positive_max(np.abs(controls[:, 1]) - v.max_steering_rate),
                _positive_max(-states[:, 3]),
            )
            heading_error = (
                abs(
                    float(
                        np.arctan2(
                            np.sin(states[-1, 2] - scene.goal_heading),
                            np.cos(states[-1, 2] - scene.goal_heading),
                        )
                    )
                )
                if scene.goal_heading is not None
                else None
            )
        speed_violation = _positive_max(speeds - v.max_speed * guidance.speed_scale)
        acceleration_violation = _positive_max(accelerations - v.max_acceleration)
        terminal_speed_violation = max(0.0, float(speeds[-1]) - p.terminal_speed)
        goal_error = float(np.linalg.norm(states[-1, :2] - scene.goal))
        progress = float(
            np.linalg.norm(np.asarray(scene.initial_state[:2]) - scene.goal)
            - goal_error
        )
        goal_violation = max(
            0.0,
            goal_error - p.goal_tolerance
            if p.terminal_mode == "goal"
            else p.minimum_progress - progress,
        )
        dense_times, dense = vehicle.dense_path(states, controls, times)
        margin = vehicle.curve_margin()
        reference_clearances = timed_clearances(
            times, reference[:, :2], scene.obstacles, v.footprint_radius
        )
        physical_clearances = timed_clearances(
            dense_times, dense, scene.obstacles, v.footprint_radius
        )
        certified = np.concatenate((reference_clearances, physical_clearances - margin))
        clearance = float(certified.min()) if len(certified) else None
        clearance_violation = (
            max(0.0, p.required_clearance - clearance) if clearance is not None else 0.0
        )
        boundary_violation = _positive_max(
            -boundary_clearances(dense, scene.bounds, v.footprint_radius + margin)
        )
        passage_violation, offsets = 0.0, []
        if guidance.side:
            if guidance.obstacle_index >= len(scene.obstacles):
                return {"valid": False, "reasons": ["guidance_obstacle_missing"]}
            offsets = passage_laterals(
                scene, times, reference[:, :2], guidance
            ).tolist()
            threshold = (
                scene.obstacles[guidance.obstacle_index].radius
                + v.footprint_radius
                + p.required_clearance
                + p.passage_extra
            )
            passage_violation = (
                max(0.0, threshold - min(guidance.side * x for x in offsets))
                if offsets
                else threshold
            )
        violations = {
            "start": start_error,
            "dynamics": dynamics_error,
            "reference_output": reference_error,
            "state_limits": state_violation,
            "input_limits": input_violation,
            "speed_limit": speed_violation,
            "acceleration_limit": acceleration_violation,
            "steering_limits": steering_violation,
            "terminal_speed": terminal_speed_violation,
            "goal_or_progress": goal_violation,
            "clearance": clearance_violation,
            "map_boundary": boundary_violation,
            "guidance": passage_violation,
            "terminal_heading": max(0.0, heading_error - p.heading_tolerance)
            if heading_error is not None
            else 0.0,
        }
        reasons.extend(
            name for name, value in violations.items() if value > p.validation_tolerance
        )
        return {
            "valid": not reasons,
            "reasons": reasons,
            "violations": violations,
            "maximum_constraint_residual": max(violations.values()),
            "goal_error": goal_error,
            "progress": progress,
            "minimum_clearance": clearance,
            "reference_clearance": float(reference_clearances.min())
            if len(reference_clearances)
            else None,
            "physical_clearance": float(physical_clearances.min())
            if len(physical_clearances)
            else None,
            "interpolation_margin": margin,
            "max_speed": float(speeds.max()),
            "max_acceleration": float(accelerations.max()),
            "max_steering": float(np.abs(states[:, 4]).max())
            if v.kind == "bicycle"
            else None,
            "max_steering_rate": float(np.abs(controls[:, 1]).max())
            if v.kind == "bicycle"
            else None,
            "passage_lateral_offsets": offsets,
            "heading_error": heading_error,
        }
    except (ValueError, TypeError, IndexError, FloatingPointError, OverflowError):
        return {"valid": False, "reasons": ["invalid_numeric_or_scene_contract"]}
