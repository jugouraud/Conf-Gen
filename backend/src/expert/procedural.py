"""Fast deterministic, length-ranked references for fixed circular obstacles."""

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
    if any(not np.allclose(ob.centers, ob.centers[0], atol=1e-12, rtol=0)
           for ob in scene.obstacles):
        raise ValueError("Fast shortest-route sampling requires fixed obstacles")
    if guidance.side and guidance.obstacle_index >= len(scene.obstacles):
        raise ValueError("Selected obstacle is absent")

    track = track_for(scene.track)
    start_s, start_d = track.project(np.asarray(scene.initial_state[:2]))
    goal_s, goal_d = track.project(np.asarray(scene.goal))
    if abs(float(start_d)) > 1e-5 or abs(float(goal_d)) > 1e-5:
        raise ValueError("Procedural start and goal must lie on the centerline")
    distance = float((goal_s - start_s) % track.length)
    if not 0.1 < distance < track.length:
        raise ValueError("Start/goal forward arc must be shorter than one lap")

    planner = config.planner
    phase = np.linspace(0.0, 1.0, planner.horizon + 1)
    progress = phase**3 * (10 + phase * (-15 + 6 * phase))
    arc = float(start_s) + distance * progress
    centerline = track.point(arc)
    normals = track.normal(arc)
    times = physical_times(scene, config)
    sides = (guidance.side,) if guidance.side else (-1, 1)
    obstacles = []
    for obstacle in scene.obstacles:
        obstacle_s, _ = track.project(obstacle.centers[0])
        relative = (float(obstacle_s) - float(start_s)) % track.length
        if not 0 < relative < distance:
            continue  # A fixed obstacle outside the current forward segment.
        obstacles.append((relative, obstacle))

    candidates = []
    widths = np.linspace(*planner.procedural_width_range, 3)
    amplitudes = np.r_[0.0, np.linspace(*planner.procedural_amplitude_range, 44)]
    for width in widths:
        for side in sides:
            for amplitude in amplitudes if obstacles else (0.0,):
                offset = np.zeros_like(progress)
                # A detour may be needed at more than one fixed obstacle.
                for relative, obstacle in obstacles:
                    gauss = np.exp(-0.5 * ((distance * progress - relative) / width)**2)
                    left = np.exp(-0.5 * (relative / width)**2)
                    right = np.exp(-0.5 * ((distance - relative) / width)**2)
                    offset += side * amplitude * (gauss - (1 - progress)*left - progress*right)
                positions = centerline + offset[:, None] * normals
                positions[0] = scene.initial_state[:2]
                positions[-1] = scene.goal
                controls = np.diff(positions, axis=0) / planner.dt
                speeds = np.linalg.norm(controls, axis=1)
                accelerations = np.linalg.norm(
                    np.diff(np.vstack((scene.initial_velocity, controls)), axis=0), axis=1
                ) / planner.dt
                if (speeds.max() > config.vehicle.max_speed * guidance.speed_scale
                        or speeds[-1] > planner.terminal_speed
                        or accelerations.max() > config.vehicle.max_acceleration
                        or np.any(controls < config.vehicle.input_lower)
                        or np.any(controls > config.vehicle.input_upper)):
                    continue
                # Exact distance from each straight physical segment to each
                # stationary disk. The independent validator checks everything else.
                segment = np.diff(positions, axis=0)
                length_squared = np.sum(segment * segment, axis=1)
                safe = True
                for obstacle in scene.obstacles:
                    relative = obstacle.centers[0] - positions[:-1]
                    fraction = np.clip(
                        np.sum(relative * segment, axis=1) / np.maximum(length_squared, 1e-12),
                        0.0, 1.0,
                    )
                    closest = positions[:-1] + fraction[:, None] * segment
                    clearance = np.min(np.linalg.norm(closest - obstacle.centers[0], axis=1))
                    if clearance < (obstacle.radius + config.vehicle.footprint_radius
                                    + planner.required_clearance + 1e-6):
                        safe = False
                        break
                if safe:
                    candidates.append((float(np.linalg.norm(segment, axis=1).sum()),
                                       width, side, amplitude, positions.copy(), controls))
    # Minimise travel length, not the old control-effort surrogate. Ties are stable.
    vehicle = Vehicle(config.vehicle, planner.dt)
    for _, _, _, _, positions, controls in sorted(candidates, key=lambda entry: entry[:4]):
        states = vehicle.rollout(np.asarray(scene.initial_state), controls)
        validation = run_expert_validation(
            scene, guidance, config, states, controls, times, vehicle.output(states)
        )
        if validation["valid"]:
            return states, controls
    raise ValueError("No collision-free, dynamically feasible route in candidate family")


def run_expert_procedural(
    scene: Scene,
    guidance: Guidance,
    config: ExpertConfig,
    *,
    guess: int,
    seed: int,
    retry: int = 0,
) -> dict:
    """Deterministic approximate shortest route; independently validated on export."""
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
                path_length=float(np.linalg.norm(np.diff(reference[:, :2], axis=0), axis=1).sum()),
            )
        except (ValueError, FloatingPointError, OverflowError) as exc:
            result.update(status="candidate_rejected", validation={
                "valid": False, "reasons": [str(exc)[:120]]
            })
    result["runtime_seconds"] = time.perf_counter() - started
    return result
