"""Goal-driven direct-shooting objective, constraints and numerical initial guesses."""

import numpy as np

from backend.src.expert.config import ExpertConfig, Guidance
from backend.src.expert.geometry import (
    scene_boundary_clearances,
    passage_laterals,
    timed_clearances,
)
from backend.src.expert.scenes import Scene
from backend.src.expert.vehicle import Vehicle


def physical_times(scene: Scene, config: ExpertConfig) -> np.ndarray:
    return scene.start_time + np.arange(config.planner.horizon + 1) * config.planner.dt


def expert_cost(
    scene: Scene, controls: np.ndarray, states: np.ndarray, config: ExpertConfig
) -> float:
    p = config.planner
    goal_error = states[:, :2] - np.asarray(scene.goal)
    prior = (
        np.asarray(scene.initial_velocity)
        if config.vehicle.kind == "velocity_disc"
        else np.zeros(2)
    )
    differences = np.diff(np.vstack((prior, controls)), axis=0)
    return float(
        p.goal_weight * p.dt * np.sum(goal_error[1:] ** 2)
        + p.terminal_weight * np.sum(goal_error[-1] ** 2)
        + p.effort_weight * p.dt * np.sum(controls**2)
        + p.smoothness_weight / p.dt * np.sum(differences**2)
    )


def planning_constraints(
    scene: Scene,
    guidance: Guidance,
    states: np.ndarray,
    controls: np.ndarray,
    config: ExpertConfig,
    vehicle: Vehicle,
) -> np.ndarray:
    p, v = config.planner, config.vehicle
    times = physical_times(scene, config)
    dense_times, dense = vehicle.dense_path(states, controls, times)
    margin = vehicle.curve_margin()
    boundary = scene_boundary_clearances(scene, dense, v.footprint_radius + margin)
    constraints = [
        (states - np.asarray(v.state_lower)).ravel(),
        (np.asarray(v.state_upper) - states).ravel(),
        boundary if scene.track is None else np.array([np.min(boundary)]),
        timed_clearances(times, states[:, :2], scene.obstacles, v.footprint_radius)
        - p.required_clearance,
        timed_clearances(
            dense_times, dense, scene.obstacles, v.footprint_radius + margin
        )
        - p.required_clearance,
    ]
    if v.kind == "velocity_disc":
        speeds = np.linalg.norm(controls, axis=1)
        acceleration = (
            np.linalg.norm(
                np.diff(np.vstack((scene.initial_velocity, controls)), axis=0), axis=1
            )
            / p.dt
        )
        terminal_speed = speeds[-1]
    else:
        speeds = states[:, 3]
        acceleration = np.abs(controls[:, 0])
        terminal_speed = abs(speeds[-1])
        constraints.extend(
            (
                speeds,
                v.max_steering - np.abs(states[:, 4]),
                v.max_steering_rate - np.abs(controls[:, 1]),
            )
        )
        if scene.goal_heading is not None:
            heading_error = np.arctan2(
                np.sin(states[-1, 2] - scene.goal_heading),
                np.cos(states[-1, 2] - scene.goal_heading),
            )
            constraints.append(np.array([p.heading_tolerance - abs(heading_error)]))
    constraints.extend(
        (
            v.max_speed * guidance.speed_scale - speeds,
            v.max_acceleration - acceleration,
            np.array([p.terminal_speed - terminal_speed]),
        )
    )
    goal_error = np.linalg.norm(states[-1, :2] - scene.goal)
    initial_distance = np.linalg.norm(np.asarray(scene.initial_state[:2]) - scene.goal)
    constraints.append(
        np.array(
            [
                p.goal_tolerance - goal_error
                if p.terminal_mode == "goal"
                else initial_distance - goal_error - p.minimum_progress
            ]
        )
    )
    if guidance.side:
        offsets = passage_laterals(scene, times, states[:, :2], guidance)
        threshold = (
            scene.obstacles[guidance.obstacle_index].radius
            + v.footprint_radius
            + p.required_clearance
            + p.passage_extra
        )
        # A scalar keeps the SLSQP constraint dimension fixed when crossings vary.
        constraints.append(
            np.array(
                [
                    float(np.min(guidance.side * offsets)) - threshold
                    if len(offsets)
                    else -threshold
                ]
            )
        )
    return np.concatenate(constraints)


def initial_controls(
    scene: Scene,
    guidance: Guidance,
    config: ExpertConfig,
    vehicle: Vehicle,
    guess: int,
    seed: int,
    retry: int,
) -> np.ndarray:
    """Waypoints seed optimization only; no tracking term uses this initial path."""
    p, v = config.planner, config.vehicle
    phase = np.linspace(0, 1, p.horizon + 1)
    # Smooth endpoints give both models a braking/acceleration-aware guess.
    progress = phase**2 * (3 - 2 * phase)
    start, goal = np.asarray(scene.initial_state[:2]), np.asarray(scene.goal)
    axis = goal - start
    normal = np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
    sign = guidance.side if guidance.side else (1 if guess % 2 == 0 else -1)
    path = (
        start
        + progress[:, None] * axis
        + sign
        * p.initial_guess_offsets[guess]
        * np.sin(np.pi * progress)[:, None]
        * normal
    )
    if v.kind == "velocity_disc":
        controls = np.diff(path, axis=0) / p.dt
    else:
        controls, current = [], np.asarray(scene.initial_state)
        for i in range(p.horizon):
            direction = path[min(i + 2, p.horizon)] - current[:2]
            desired_yaw = np.arctan2(direction[1], direction[0])
            angle = np.arctan2(
                np.sin(desired_yaw - current[2]), np.cos(desired_yaw - current[2])
            )
            desired_speed = min(
                np.linalg.norm(path[i + 1] - path[i]) / p.dt,
                v.max_speed * guidance.speed_scale,
            )
            desired_steering = np.clip(
                np.arctan2(v.wheelbase * angle, max(desired_speed * p.dt, 0.1)),
                -v.max_steering,
                v.max_steering,
            )
            control = np.array(
                [
                    (desired_speed - current[3]) / p.dt,
                    (desired_steering - current[4]) / p.dt,
                ]
            )
            control = np.clip(control, v.input_lower, v.input_upper)
            controls.append(control)
            current = vehicle.step(current, control)
        controls = np.stack(controls)
    if retry:
        controls += np.random.default_rng(seed).normal(
            0, p.retry_jitter, controls.shape
        )
    return np.clip(controls, v.input_lower, v.input_upper)
