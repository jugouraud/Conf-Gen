"""Explicit, bounded expert-planning configuration; defaults are frozen in artifacts."""

from dataclasses import asdict, dataclass, fields

import numpy as np

SPLITS = ("train", "reference", "future_calibration", "test")


def construct(cls: type, record: dict):
    if not isinstance(record, dict) or set(record) - {f.name for f in fields(cls)}:
        raise ValueError(f"Unknown fields for {cls.__name__}")
    return cls(**record)


def positive(value: float, name: str, *, zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not np.isfinite(value)
        or (value < 0 if zero else value <= 0)
    ):
        raise ValueError(f"Invalid {name}")


@dataclass(frozen=True)
class VehicleSettings:
    kind: str = "velocity_disc"
    footprint_radius: float = 0.15
    wheelbase: float = 0.5
    max_speed: float = 1.5
    max_acceleration: float = 1.5
    max_steering: float = 0.5
    max_steering_rate: float = 0.8
    state_lower: tuple[float, ...] = (-10.0, -10.0)
    state_upper: tuple[float, ...] = (10.0, 10.0)
    input_lower: tuple[float, ...] = (-2.0, -2.0)
    input_upper: tuple[float, ...] = (2.0, 2.0)
    output: str = "xy"
    substeps: int = 4
    integration_margin: float = 0.002

    def __post_init__(self) -> None:
        if self.kind not in {"velocity_disc", "bicycle"} or self.output not in {
            "xy",
            "xy_sincos",
        }:
            raise ValueError("Unknown vehicle or output map")
        if self.output == "xy_sincos" and self.kind != "bicycle":
            raise ValueError("Heading output requires a bicycle state")
        for name in (
            "footprint_radius",
            "wheelbase",
            "max_speed",
            "max_acceleration",
            "max_steering",
            "max_steering_rate",
        ):
            positive(getattr(self, name), name)
        positive(self.integration_margin, "integration_margin", zero=True)
        if (
            self.max_steering >= np.pi / 2
            or type(self.substeps) is not int
            or not 1 <= self.substeps <= 32
        ):
            raise ValueError("Invalid steering range or integration substeps")
        n = 2 if self.kind == "velocity_disc" else 5
        for prefix, size in (("state", n), ("input", 2)):
            low, high = (
                np.asarray(getattr(self, prefix + "_lower")),
                np.asarray(getattr(self, prefix + "_upper")),
            )
            if (
                low.shape != (size,)
                or high.shape != (size,)
                or not np.isfinite([low, high]).all()
                or np.any(low >= high)
            ):
                raise ValueError("Invalid state/input bounds")
            object.__setattr__(self, prefix + "_lower", tuple(float(x) for x in low))
            object.__setattr__(self, prefix + "_upper", tuple(float(x) for x in high))


@dataclass(frozen=True)
class PlannerSettings:
    horizon: int = 16
    dt: float = 0.4
    terminal_mode: str = "goal"
    minimum_progress: float = 1.0
    goal_tolerance: float = 0.08
    terminal_speed: float = 0.08
    heading_tolerance: float = 0.1
    required_clearance: float = 0.12
    goal_weight: float = 0.2
    terminal_weight: float = 30.0
    effort_weight: float = 0.1
    smoothness_weight: float = 0.2
    passage_extra: float = 0.05
    initial_guess_offsets: tuple[float, ...] = (0.9, 1.5)
    retries: int = 0
    retry_jitter: float = 0.03
    max_iterations: int = 160
    ftol: float = 1e-8
    validation_tolerance: float = 1e-6
    solver_timeout: float = 10.0
    require_convergence: bool = True

    def __post_init__(self) -> None:
        for name, maximum in (("horizon", 128), ("max_iterations", 5000)):
            if (
                type(getattr(self, name)) is not int
                or not 1 <= getattr(self, name) <= maximum
            ):
                raise ValueError(f"Invalid {name}")
        if (
            type(self.retries) is not int
            or not 0 <= self.retries <= 4
            or type(self.require_convergence) is not bool
        ):
            raise ValueError("Invalid retry or convergence policy")
        if self.terminal_mode not in {"goal", "progress"}:
            raise ValueError("Unknown terminal mode")
        for name in (
            "dt",
            "goal_tolerance",
            "heading_tolerance",
            "ftol",
            "validation_tolerance",
            "solver_timeout",
        ):
            positive(getattr(self, name), name)
        for name in (
            "minimum_progress",
            "terminal_speed",
            "required_clearance",
            "goal_weight",
            "terminal_weight",
            "effort_weight",
            "smoothness_weight",
            "passage_extra",
            "retry_jitter",
        ):
            positive(getattr(self, name), name, zero=True)
        if (
            self.solver_timeout > 3600
            or self.dt > 60
            or self.validation_tolerance > 0.01
        ):
            raise ValueError("Unreasonable time or validation tolerance")
        if not 1 <= len(self.initial_guess_offsets) <= 8:
            raise ValueError("Need 1..8 initial guesses")
        for offset in self.initial_guess_offsets:
            positive(offset, "initial_guess_offset", zero=True)
        object.__setattr__(
            self, "initial_guess_offsets", tuple(self.initial_guess_offsets)
        )


@dataclass(frozen=True)
class Guidance:
    name: str
    side: int = 0
    speed_scale: float = 1.0
    obstacle_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not 1 <= len(self.name) <= 100:
            raise ValueError("Invalid guidance name")
        if type(self.side) is not int or self.side not in {-1, 0, 1}:
            raise ValueError("Passage side is -1/right, 0/unconstrained, +1/left")
        if type(self.obstacle_index) is not int or not 0 <= self.obstacle_index < 32:
            raise ValueError("Invalid guidance obstacle selector")
        positive(self.speed_scale, "speed_scale")
        if self.speed_scale > 1:
            raise ValueError("Guidance cannot relax the vehicle speed limit")


@dataclass(frozen=True)
class DatasetSettings:
    seed: int = 20261109
    split_weights: tuple[float, ...] = (0.5, 1 / 6, 1 / 6, 1 / 6)
    minimum_diversity: float = 0.08
    normalization_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid dataset seed")
        weights = np.asarray(self.split_weights)
        if (
            weights.shape != (4,)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1)
        ):
            raise ValueError("Four positive split weights must sum to one")
        object.__setattr__(self, "split_weights", tuple(float(x) for x in weights))
        positive(self.minimum_diversity, "minimum_diversity", zero=True)
        positive(self.normalization_epsilon, "normalization_epsilon")


@dataclass(frozen=True)
class ExecutionSettings:
    workers: int = 2
    threads_per_worker: int = 1
    startup_timeout: float = 15.0

    def __post_init__(self) -> None:
        if any(
            type(x) is not int or not 1 <= x <= 16
            for x in (self.workers, self.threads_per_worker)
        ):
            raise ValueError("Invalid worker/thread count")
        positive(self.startup_timeout, "startup_timeout")
        if self.startup_timeout > 300:
            raise ValueError("Startup timeout exceeds five minutes")


@dataclass(frozen=True)
class ExpertConfig:
    vehicle: VehicleSettings
    planner: PlannerSettings
    dataset: DatasetSettings
    execution: ExecutionSettings
    guidance: tuple[Guidance, ...]

    @classmethod
    def from_record(cls, record: dict) -> "ExpertConfig":
        if set(record) != {"vehicle", "planner", "dataset", "execution", "guidance"}:
            raise ValueError("Expert config sections differ from schema")
        guidance = tuple(construct(Guidance, row) for row in record["guidance"])
        if not 1 <= len(guidance) <= 8 or len({g.name for g in guidance}) != len(
            guidance
        ):
            raise ValueError("Need 1..8 uniquely named guidance choices")
        return cls(
            construct(VehicleSettings, record["vehicle"]),
            construct(PlannerSettings, record["planner"]),
            construct(DatasetSettings, record["dataset"]),
            construct(ExecutionSettings, record["execution"]),
            guidance,
        )

    def to_record(self) -> dict:
        return asdict(self)
