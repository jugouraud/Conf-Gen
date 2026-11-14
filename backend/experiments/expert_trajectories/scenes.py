"""Experiment-specific map distribution, separate from general planning stages."""

from pathlib import Path

import numpy as np

from backend.src.common.storage import calculate_file_sha256
from backend.src.expert.config import ExpertConfig
from backend.src.expert.storage import MAX_EXPERT_TRAJECTORIES, read_record


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
    if type(settings["count"]) is not int or not 4 <= settings["count"] <= MAX_EXPERT_TRAJECTORIES:
        raise ValueError(f"Procedural scene count must be 4..{MAX_EXPERT_TRAJECTORIES}")
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
    """Sample forward arcs with one fixed obstacle on a fixed circuit."""
    if supplied is not None or "track" not in settings:
        return _sample_legacy_scenes(settings, config, supplied)
    from dataclasses import asdict

    from backend.src.expert.track import TrackSpec, track_for

    ranges = (
        "goal_arc_range", "obstacle_fraction_range", "obstacle_lateral_range",
        "obstacle_radius_range",
    )
    allowed = {*ranges, "count", "track", "centerline_endpoints"}
    if not set(settings) <= allowed or not {*ranges, "count", "track"} <= set(settings):
        raise ValueError("Unknown fixed-circuit sampling settings")
    for name in ranges:
        values = np.asarray(settings[name], dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
            raise ValueError(f"Invalid {name}")
    if type(settings["count"]) is not int or not 4 <= settings["count"] <= MAX_EXPERT_TRAJECTORIES:
        raise ValueError(f"Scene count must be 4..{MAX_EXPERT_TRAJECTORIES}")
    centerline_endpoints = settings.get("centerline_endpoints", False)
    if type(centerline_endpoints) is not bool:
        raise ValueError("centerline_endpoints must be boolean")
    spec = TrackSpec(**settings["track"])
    track = track_for(spec)
    margin = config.vehicle.footprint_radius + config.planner.required_clearance
    left_limit, right_limit = spec.left_width - margin, spec.right_width - margin
    if min(left_limit, right_limit) <= 0:
        raise ValueError("Track is too narrow for vehicle and clearance")
    goal_min, goal_max = settings["goal_arc_range"]
    frac_min, frac_max = settings["obstacle_fraction_range"]
    lateral_min, lateral_max = settings["obstacle_lateral_range"]
    radius_min, radius_max = settings["obstacle_radius_range"]
    if not 0 < goal_min <= goal_max < track.length:
        raise ValueError("Forward goal arc must be shorter than one full lap")
    if not 0 < frac_min <= frac_max < 1:
        raise ValueError("Obstacle must lie strictly between start and goal")
    if (radius_min <= 0 or max(abs(lateral_min), abs(lateral_max))
            + radius_max + config.vehicle.footprint_radius
            + config.planner.required_clearance >= min(spec.left_width, spec.right_width)):
        raise ValueError("Obstacle and clearance margin do not fit on the track")
    if max(abs(lateral_min), abs(lateral_max)) >= (
        radius_min + config.vehicle.footprint_radius + config.planner.required_clearance
    ):
        raise ValueError("Obstacle lateral range must stay near the centerline")

    duration = config.planner.dt * config.planner.horizon
    times = np.linspace(0.0, duration, config.planner.horizon + 1).tolist()
    bounds = [
        spec.center[0] - spec.radii[0] - max(spec.left_width, spec.right_width),
        spec.center[0] + spec.radii[0] + max(spec.left_width, spec.right_width),
        spec.center[1] - spec.radii[1] - max(spec.left_width, spec.right_width),
        spec.center[1] + spec.radii[1] + max(spec.left_width, spec.right_width),
    ]
    track_record = asdict(spec)
    scenes = []
    children = np.random.SeedSequence(config.dataset.seed).spawn(settings["count"])
    for index, sequence in enumerate(children):
        rng = np.random.default_rng(sequence)
        # Offset normals near polyline corners can project onto a neighboring
        # segment; resample rather than changing the requested forward distance.
        for _ in range(16):
            start_arc = float(rng.uniform(0.0, track.length))
            distance = float(rng.uniform(goal_min, goal_max))
            goal_arc = start_arc + distance
            if centerline_endpoints:
                start, goal = track.point(start_arc), track.point(goal_arc)
            else:
                start_d, goal_d = rng.uniform(-right_limit, left_limit, size=2)
                start = track.point(start_arc) + start_d * track.normal(start_arc)
                goal = track.point(goal_arc) + goal_d * track.normal(goal_arc)
            projected, _ = track.project(np.stack((start, goal)))
            expected = np.mod((start_arc, goal_arc), track.length)
            residual = (projected - expected + track.length / 2) % track.length
            residual -= track.length / 2
            if np.max(np.abs(residual)) < 1e-6:
                break
        else:
            raise ValueError("Unable to sample unambiguous track endpoints")
        obstacle_arc = start_arc + distance * rng.uniform(frac_min, frac_max)
        lateral = rng.uniform(lateral_min, lateral_max)
        obstacle_center = track.point(obstacle_arc) + lateral * track.normal(obstacle_arc)
        obstacle = {
            "obstacle_id": "obstacle-0", "times": times,
            "centers": np.repeat(obstacle_center[None, :], len(times), axis=0).tolist(),
            "radius": float(rng.uniform(radius_min, radius_max)),
        }
        initial_tangent = track.tangent(start_arc)
        goal_tangent = track.tangent(goal_arc)
        bicycle = config.vehicle.kind == "bicycle"
        state = ([*start.tolist(), np.arctan2(initial_tangent[1], initial_tangent[0]),
                  0.0, 0.0] if bicycle else start.tolist())
        scenes.append({
            "scene_id": f"scene-{index:04d}", "group_id": f"episode-{index:04d}",
            "map_id": "fixed-silverstone-v1" if spec.layout == "silverstone" else "fixed-ellipse-v3",
            "bounds": bounds, "track": track_record, "start_time": 0.0,
            "initial_state": state, "initial_velocity": [0.0, 0.0],
            "goal": goal.tolist(),
            "goal_heading": float(np.arctan2(goal_tangent[1], goal_tangent[0])) if bicycle else None,
            "obstacles": [obstacle],
        })
    return scenes, {
        "kind": "fixed_circuit", "settings": settings,
        "seed": config.dataset.seed,
        "rng": "numpy.SeedSequence child streams / PCG64",
    }
