"""Experiment-specific map distribution, separate from general planning stages."""

from pathlib import Path

import numpy as np

from backend.src.common.storage import calculate_file_sha256
from backend.src.expert.config import ExpertConfig
from backend.src.expert.storage import read_record


def run_scene_sampling(
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
