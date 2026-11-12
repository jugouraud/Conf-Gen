"""Experiment-specific map distribution, separate from general planning stages."""

from pathlib import Path

import numpy as np

from backend.src.common.storage import calculate_file_sha256
from backend.src.expert.config import ExpertConfig
from backend.src.expert.storage import read_record


def _sample_legacy_scenes(
    settings: dict, config: ExpertConfig, supplied: Path | None = None
) -> tuple[list[dict], dict]:
    if supplied is not None:
        record = read_record(supplied)
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "scenes"}
            or record["schema"] != "expert-scenes-v1"
        ):
            raise ValueError("Unknown user scene schema")
        return record["scenes"], {
            "kind": "user_provided",
            "sha256": calculate_file_sha256(supplied),
        }
    ranges = (
        "start_time_range",
        "map_width_range",
        "map_height_range",
        "goal_distance_range",
        "initial_y_range",
        "obstacle_radius_range",
        "obstacle_speed_range",
    )
    if set(settings) != {
        *ranges,
        "count",
        "obstacle_count",
        "moving_fraction",
        "invalid_initial_fraction",
    }:
        raise ValueError("Unknown procedural scene setting")
    for name in ranges:
        value = np.asarray(settings[name])
        if value.shape != (2,) or not np.isfinite(value).all() or value[0] > value[1]:
            raise ValueError("Invalid procedural sampling range")
    if type(settings["count"]) is not int or not 4 <= settings["count"] <= 2000:
        raise ValueError("Procedural scene count must be 4..2000")
    if (
        type(settings["obstacle_count"]) is not int
        or not 0 <= settings["obstacle_count"] <= 8
    ):
        raise ValueError("Invalid obstacle count")
    if any(
        not 0 <= settings[name] <= 1
        for name in ("moving_fraction", "invalid_initial_fraction")
    ):
        raise ValueError("Invalid procedural probability")
    scenes = []
    # Independent deterministic child streams; worker order never samples scenes.
    children = np.random.SeedSequence(config.dataset.seed).spawn(settings["count"])
    for index, sequence in enumerate(children):
        rng = np.random.default_rng(sequence)
        values = {name: float(rng.uniform(*settings[name])) for name in ranges}
        start_time = values["start_time_range"]
        duration = config.planner.horizon * config.planner.dt
        times = start_time + np.array([0, 0.35, 0.7, 1.0]) * duration
        initial = np.array([0.3, values["initial_y_range"]])
        goal = initial + np.array([values["goal_distance_range"], 0.0])
        obstacles = []
        for obstacle_index in range(settings["obstacle_count"]):
            position = initial + np.array(
                [
                    values["goal_distance_range"]
                    * (obstacle_index + 1)
                    / (settings["obstacle_count"] + 1),
                    rng.uniform(-0.12, 0.12),
                ]
            )
            centers = [position]
            moving = rng.random() < settings["moving_fraction"]
            for dt in np.diff(times):
                velocity = (
                    rng.uniform(*settings["obstacle_speed_range"], 2)
                    if moving
                    else np.zeros(2)
                )
                centers.append(centers[-1] + dt * velocity)
            if (
                obstacle_index == 0
                and rng.random() < settings["invalid_initial_fraction"]
            ):
                # Intentional invalid cases exercise rejection records; never resample them away.
                centers[0] = initial.copy()
            obstacles.append(
                {
                    "obstacle_id": f"obstacle-{obstacle_index}",
                    "times": times.tolist(),
                    "centers": np.asarray(centers).tolist(),
                    "radius": float(rng.uniform(*settings["obstacle_radius_range"])),
                }
            )
        state = (
            initial.tolist()
            if config.vehicle.kind == "velocity_disc"
            else [*initial.tolist(), 0.0, 0.0, 0.0]
        )
        scenes.append(
            {
                "scene_id": f"scene-{index:04d}",
                "group_id": f"episode-{index:04d}",
                "map_id": f"map-{index:04d}",
                "bounds": [
                    -1.0,
                    -1.0 + values["map_width_range"],
                    -values["map_height_range"] / 2,
                    values["map_height_range"] / 2,
                ],
                "start_time": start_time,
                "initial_state": state,
                "initial_velocity": [0.0, 0.0],
                "goal": goal.tolist(),
                "goal_heading": 0.0 if config.vehicle.kind == "bicycle" else None,
                "obstacles": obstacles,
            }
        )
    return scenes, {
        "kind": "procedural",
        "settings": settings,
        "seed": config.dataset.seed,
        "rng": "numpy.SeedSequence child streams / PCG64",
    }

def run_scene_sampling(
    settings: dict, config: ExpertConfig, supplied: Path | None = None
) -> tuple[list[dict], dict]:
    """A single immutable ellipse; each episode gets its own RNG stream."""
    if supplied is not None or "track" not in settings:
        return _sample_legacy_scenes(settings, config, supplied)
    from dataclasses import asdict

    from backend.src.expert.track import TrackSpec, track_for

    ranges = (
        "start_time_range", "goal_arc_range", "obstacle_radius_range",
        "obstacle_lateral_range", "obstacle_speed_range",
        "obstacle_lateral_speed_range",
    )
    if set(settings) != {
        *ranges, "count", "obstacle_count", "moving_fraction",
        "invalid_initial_fraction", "track",
    }:
        raise ValueError("Unknown fixed-circuit sampling settings")
    for name in ranges:
        values = np.asarray(settings[name], dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
            raise ValueError(f"Invalid {name}")
    if type(settings["count"]) is not int or not 4 <= settings["count"] <= 2000:
        raise ValueError("Scene count must be 4..2000")
    if type(settings["obstacle_count"]) is not int or not 1 <= settings["obstacle_count"] <= 8:
        raise ValueError("Fixed-track guidance requires 1..8 obstacles")
    if any(
        isinstance(settings[name], bool) or not isinstance(settings[name], (int, float))
        or not np.isfinite(settings[name]) or not 0 <= settings[name] <= 1
        for name in ("moving_fraction", "invalid_initial_fraction")
    ):
        raise ValueError("Invalid scenario probability")
    spec = TrackSpec(**settings["track"])
    track = track_for(spec)
    if max(spec.left_width, spec.right_width) <= config.vehicle.footprint_radius + config.planner.required_clearance:
        raise ValueError("Track is too narrow for vehicle and clearance")
    if settings["goal_arc_range"][0] <= 0 or settings["goal_arc_range"][1] >= track.length / 2:
        raise ValueError("Goal arc must stay forward and below half a lap")
    time_range = settings["start_time_range"]
    duration = config.planner.dt * config.planner.horizon
    scenes = []
    children = np.random.SeedSequence(config.dataset.seed).spawn(settings["count"])
    for index, sequence in enumerate(children):
        rng = np.random.default_rng(sequence)
        t0 = float(rng.uniform(*time_range))
        start_arc = rng.uniform(0.0, track.length)
        distance = float(rng.uniform(*settings["goal_arc_range"]))
        start, goal = track.point(start_arc), track.point(start_arc + distance)
        initial_tangent = track.tangent(start_arc)
        goal_tangent = track.tangent(start_arc + distance)
        times = t0 + np.linspace(0.0, duration, config.planner.horizon + 1)
        obstacles = []
        for j in range(settings["obstacle_count"]):
            arc = start_arc + distance * (j + 1) / (settings["obstacle_count"] + 1)
            offset = rng.uniform(*settings["obstacle_lateral_range"])
            speed, lateral_speed = 0.0, 0.0
            if rng.random() < settings["moving_fraction"]:
                speed = rng.uniform(*settings["obstacle_speed_range"])
                lateral_speed = rng.uniform(*settings["obstacle_lateral_speed_range"])
            longitudinal = arc + speed * (times - t0)
            lateral = offset + lateral_speed * (times - t0)
            centers = track.point(longitudinal) + lateral[:, None] * track.normal(longitudinal)
            if j == 0 and rng.random() < settings["invalid_initial_fraction"]:
                centers[0] = start  # Rejected, never silently resampled.
            obstacles.append({
                "obstacle_id": f"obstacle-{j}", "times": times.tolist(),
                "centers": centers.tolist(),
                "radius": float(rng.uniform(*settings["obstacle_radius_range"])),
            })
        bicycle = config.vehicle.kind == "bicycle"
        state = [*start.tolist(), np.arctan2(initial_tangent[1], initial_tangent[0]), 0.0, 0.0] if bicycle else start.tolist()
        scenes.append({
            "scene_id": f"scene-{index:04d}", "group_id": f"episode-{index:04d}",
            "map_id": "fixed-ellipse-v1", "bounds": [
                spec.center[0] - spec.radii[0] - max(spec.left_width, spec.right_width),
                spec.center[0] + spec.radii[0] + max(spec.left_width, spec.right_width),
                spec.center[1] - spec.radii[1] - max(spec.left_width, spec.right_width),
                spec.center[1] + spec.radii[1] + max(spec.left_width, spec.right_width),
            ],
            "track": asdict(spec), "start_time": t0,
            "initial_state": state, "initial_velocity": [0.0, 0.0],
            "goal": goal.tolist(),
            "goal_heading": float(np.arctan2(goal_tangent[1], goal_tangent[0])) if bicycle else None,
            "obstacles": obstacles,
        })
    return scenes, {
        "kind": "fixed_circuit", "settings": settings,
        "seed": config.dataset.seed,
        "rng": "numpy.SeedSequence child streams / PCG64",
    }
