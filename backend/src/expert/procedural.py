"""Bounded, independently validated reference sampling without a nonlinear solver."""

import time

import numpy as np

from backend.src.expert.config import ExpertConfig, Guidance
from backend.src.expert.problem import expert_cost, physical_times
from backend.src.expert.scenes import Scene
from backend.src.expert.track import track_for
from backend.src.expert.validate import run_expert_validation, validate_initial_scene
from backend.src.expert.vehicle import Vehicle


def _candidate(
    scene: Scene, guidance: Guidance, config: ExpertConfig, *, seed: int, guess: int
) -> tuple[np.ndarray, np.ndarray]:
    if scene.track is None:
        raise ValueError("Procedural generation requires a fixed circuit")
    if config.vehicle.kind != "velocity_disc":
        raise ValueError("Procedural generation currently supports velocity_disc only")
    track = track_for(scene.track)
    start_s, start_d = track.project(np.asarray(scene.initial_state[:2]))
    goal_s, goal_d = track.project(np.asarray(scene.goal))
    if abs(float(start_d)) > 1e-5 or abs(float(goal_d)) > 1e-5:
        raise ValueError("Procedural start and goal must lie on the centerline")
    distance = float((goal_s - start_s) % track.length)
    if not 0.1 < distance < track.length / 2:
        raise ValueError("Start/goal arc distance is invalid for one horizon")
    if guidance.side and guidance.obstacle_index >= len(scene.obstacles):
        raise ValueError("Selected obstacle is absent")
    rng = np.random.default_rng(seed)
    planner = config.planner
    amplitude = rng.uniform(*planner.procedural_amplitude_range)
    width = rng.uniform(*planner.procedural_width_range)
    phase = np.linspace(0, 1, planner.horizon + 1)
    progress = phase**3 * (10 + phase * (-15 + 6 * phase))
    arc = float(start_s) + distance * progress
    positions = track.point(arc)
    if guidance.side:
        obstacle = scene.obstacles[guidance.obstacle_index]
        obstacle_s, _ = track.project(obstacle.at(np.array([scene.start_time +
            planner.horizon * planner.dt / 2]))[0])
        relative = (float(obstacle_s) - float(start_s)) % track.length
        if not 0 < relative < distance:
            raise ValueError("Obstacle lies outside the forward track segment")
        # Endpoint-corrected Gaussian guarantees exact initial and goal states.
        gauss = np.exp(-0.5 * ((distance * progress - relative) / width) ** 2)
        left = np.exp(-0.5 * (relative / width) ** 2)
        right = np.exp(-0.5 * ((distance - relative) / width) ** 2)
        offset = guidance.side * amplitude * (gauss - (1 - progress) * left - progress * right)
        positions += offset[:, None] * track.normal(arc)
    # Exact discrete dynamics for the velocity-disc; no hidden path smoothing.
    positions[0] = scene.initial_state[:2]
    positions[-1] = scene.goal
    controls = np.diff(positions, axis=0) / planner.dt
    states = Vehicle(config.vehicle, planner.dt).rollout(
        np.asarray(scene.initial_state), controls
    )
    return states, controls


def run_expert_procedural(
    scene: Scene,
    guidance: Guidance,
    config: ExpertConfig,
    *,
    guess: int,
    seed: int,
    retry: int = 0,
) -> dict:
    """One deterministic attempt; only the independent validator grants eligibility."""
    started = time.perf_counter()
    result = dict(status="pending", solver_success=False, feasible=False, eligible=False,
                  guess=guess, retry=retry, seed=seed, states=None, controls=None,
                  timestamps=None, y_ref_star=None, validation=None, cost=None,
                  method="procedural")
    reasons = validate_initial_scene(scene, config)
    if reasons:
        result.update(status="invalid_scene_or_guidance", validation={"valid": False, "reasons": reasons})
    else:
        try:
            states, controls = _candidate(scene, guidance, config, seed=seed, guess=guess)
            times = physical_times(scene, config)
            reference = Vehicle(config.vehicle, config.planner.dt).output(states)
            validation = run_expert_validation(
                scene, guidance, config, states, controls, times, reference
            )
            valid = bool(validation["valid"])
            result.update(
                status="validated" if valid else "validation_failed",
                solver_success=valid, feasible=valid, eligible=valid,
                states=states.tolist(), controls=controls.tolist(),
                timestamps=times.tolist(), y_ref_star=reference.tolist(),
                validation=validation,
                cost=expert_cost(scene, controls, states, config),
            )
        except (ValueError, FloatingPointError, OverflowError) as exc:
            result.update(status="candidate_rejected", validation={
                "valid": False, "reasons": [str(exc)[:120]]
            })
    result["runtime_seconds"] = time.perf_counter() - started
    return result
