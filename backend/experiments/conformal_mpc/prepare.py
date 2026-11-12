"""Freeze synthetic scenes, catalogue rules and runtime configuration."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from backend.experiments.conformal_mpc.synthetic import (
    Scene,
    SyntheticTrajectoryModel,
    make_scene,
)
from backend.src.common.storage import read_toml_config
from backend.src.control.contracts import ControlState, Example, RepairChoice, Sampling
from backend.src.control.partition import validate_partitions
from backend.src.control.sample import integrate_field
from backend.src.mpc.dynamics import LinearDynamics, TrackingConfig
from backend.src.mpc.geometry import minimum_clearance
from backend.src.repair.steering import (
    SteeringSchedule,
    build_constant_steering_schedule,
    build_window_steering_schedule,
)


@dataclass(frozen=True)
class PreparedExperiment:
    config: dict
    model: SyntheticTrajectoryModel
    groups: dict[str, tuple[Example, ...]]
    scenes: dict[str, tuple[Scene, ...]]
    tracking: TrackingConfig
    sampling: Sampling
    schedules: tuple[SteeringSchedule, ...]
    config_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "config",
            MappingProxyType(
                {
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in self.config.items()
                }
            ),
        )
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))
        object.__setattr__(self, "scenes", MappingProxyType(dict(self.scenes)))


def run_simulation_preparation(path: Path) -> PreparedExperiment:
    cfg = read_toml_config(path)
    required = {
        "seed",
        "reference_episodes",
        "detector_episodes",
        "output_episodes",
        "test_episodes",
        "physical_steps",
        "horizon",
        "physical_dt",
        "probe_count",
        "probe_steps",
        "sampling_steps",
        "sampling_method",
        "warm_start",
        "alpha_safe",
        "alpha_unsafe",
        "alpha_output",
        "aggregation",
        "bandwidth",
        "strengths",
        "window_widths",
        "deadline_seconds",
        "footprint_radius",
        "required_clearance",
        "tracking_uncertainty",
        "input_limit",
        "tracking_weight",
        "input_weight",
        "solver_iterations",
        "solver_tolerance",
        "process_noise",
        "goal_tolerance",
        "threads",
    }
    if set(cfg) != required:
        raise ValueError("Config fields differ from schema v1")
    for key in (
        "reference_episodes",
        "detector_episodes",
        "output_episodes",
        "test_episodes",
        "physical_steps",
        "horizon",
        "probe_count",
        "probe_steps",
        "threads",
    ):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 10000:
            raise ValueError("Invalid bounded integer setting")
    for key in ("alpha_safe", "alpha_unsafe", "alpha_output"):
        if not 0 < cfg[key] < 1:
            raise ValueError("Invalid alpha")
    for key in ("deadline_seconds", "process_noise", "goal_tolerance", "bandwidth"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("Invalid positive finite setting")
    if cfg["aggregation"] not in {"nearest", "mean", "exponential"}:
        raise ValueError("Unknown aggregation")
    if (
        4 * (cfg["output_episodes"] + cfg["test_episodes"]) * cfg["physical_steps"]
        > 100000
    ):
        raise ValueError("Experiment exceeds 100,000 physical policy steps")
    model = SyntheticTrajectoryModel()
    dt = cfg["physical_dt"]
    dynamics = LinearDynamics(np.eye(2), dt * np.eye(2), np.eye(2), dt)
    tracking = TrackingConfig(
        dynamics,
        cfg["tracking_weight"] * np.eye(2),
        cfg["input_weight"] * np.eye(2),
        np.array([-1.0, -4.0]),
        np.array([6.0, 4.0]),
        np.full(2, -cfg["input_limit"]),
        np.full(2, cfg["input_limit"]),
        cfg["footprint_radius"],
        cfg["required_clearance"],
        cfg["tracking_uncertainty"],
        cfg["solver_iterations"],
        cfg["solver_tolerance"],
    )
    sampling = Sampling(
        cfg["sampling_steps"], cfg["sampling_method"], cfg["warm_start"]
    )
    profiles = [build_constant_steering_schedule(1.0, "R1")]
    for width in cfg["window_widths"]:
        for name, center in (
            ("early", width / 2),
            ("middle", 0.5),
            ("late", 1 - width / 2),
        ):
            profiles.append(
                build_window_steering_schedule("R2_" + name, [center], width)
            )
    schedules = tuple(
        shape.scale_strength(strength)
        for shape in profiles
        for strength in cfg["strengths"]
    )
    if not 1 <= len(schedules) <= 10000:
        raise ValueError("Invalid candidate grid size")
    scenes, groups = {}, {}
    rng = np.random.default_rng(cfg["seed"])
    for split, count_key in (
        ("reference", "reference_episodes"),
        ("detector_calibration", "detector_episodes"),
        ("output_calibration", "output_episodes"),
        ("test", "test_episodes"),
    ):
        rows, scene_rows = [], []
        for index in range(cfg[count_key]):
            scene = make_scene(f"{split}-{index:04d}", int(rng.integers(0, 2**31)))
            context = scene.observe(np.asarray(scene.initial), 0, cfg["horizon"], dt)
            state = ControlState(
                context,
                0,
                np.inf,
                lambda: 0.0,
                repair=RepairChoice("unsteered", build_constant_steering_schedule()),
            )
            reference = integrate_field(
                model, state, scene.guidance, sampling, scene.seed
            )
            # Independent geometry determines labels; no detector score or guidance
            # name is used to label the reference.
            safe = minimum_clearance(
                reference, context.obstacles, tracking.footprint_radius
            ) >= (tracking.required_clearance + tracking.tracking_uncertainty)
            rows.append(Example(context, scene.guidance, safe))
            scene_rows.append(scene)
        scenes[split], groups[split] = tuple(scene_rows), tuple(rows)
    validate_partitions(groups)
    config_id = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    return PreparedExperiment(
        cfg, model, groups, scenes, tracking, sampling, schedules, config_id
    )
