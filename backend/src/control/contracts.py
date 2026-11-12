"""Immutable scientific inputs and per-call owned stage results.

Latents/signatures are FP64 arrays; probes have shape (J, *latent_shape).
Decoded references have shape (H+1, 2), in metres at physical offsets i*dt.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from backend.src.mpc.geometry import Obstacle, frozen_array
from backend.src.mpc.solver import TrackingResult
from backend.src.repair.steering import SteeringSchedule

Mode = Literal["nominal", "safe_only", "joint", "repaired"]
SPLITS = ("reference", "detector_calibration", "output_calibration", "test")


@dataclass(frozen=True)
class Context:
    episode: str
    k: int
    state: tuple[float, ...]
    goal: tuple[float, float]
    obstacles: tuple[Obstacle, ...]
    dt: float
    horizon: int
    compatibility: str = "default"
    features: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.episode or type(self.k) is not int or self.k < 0:
            raise ValueError("Episode and nonnegative physical step required")
        if type(self.horizon) is not int or not 1 <= self.horizon <= 128:
            raise ValueError("Horizon must be in [1,128]")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Physical dt must be positive")
        for name in ("state", "goal", "features"):
            object.__setattr__(
                self, name, tuple(frozen_array(getattr(self, name), ndim=1))
            )
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        if len(self.goal) != 2 or not self.state or len(self.obstacles) > 128:
            raise ValueError("Invalid state, goal or obstacle count")
        if any(len(o.centers) != self.horizon + 1 for o in self.obstacles):
            raise ValueError("Predictions must cover i=0..H")


@dataclass(frozen=True)
class Example:
    context: Context
    guidance: str
    safe: bool


@dataclass(frozen=True)
class ProbeBank:
    states: np.ndarray
    times: np.ndarray
    episodes: tuple[str, ...]
    model_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", frozen_array(self.states))
        object.__setattr__(self, "times", frozen_array(self.times, ndim=1))
        object.__setattr__(self, "episodes", tuple(self.episodes))
        if (
            self.states.ndim < 2
            or not 1 <= len(self.states) <= 4096
            or self.times.shape != (len(self.states),)
            or np.any((self.times < 0) | (self.times >= 1))
            or not self.episodes
            or not self.model_revision
        ):
            raise ValueError("Invalid fixed probes or provenance")


@dataclass(frozen=True)
class Anchor:
    """Frozen catalogue guidance; signature belongs to its reference context."""

    episode: str
    guidance: str
    compatibility: str
    signature: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "signature", frozen_array(self.signature))


@dataclass(frozen=True)
class Detector:
    probes: ProbeBank
    safe: tuple[Anchor, ...]
    unsafe: tuple[Anchor, ...]
    r_safe: float
    r_unsafe: float
    aggregation: str = "nearest"
    bandwidth: float = 1.0

    def __post_init__(self) -> None:
        for name in ("safe", "unsafe"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            bank = getattr(self, name)
            if not bank or len(bank) > 4096:
                raise ValueError("Both reference classes must be nonempty and bounded")
            if any(a.signature.shape != self.probes.states.shape for a in bank):
                raise ValueError("Reference signature shape differs from fixed probes")
        if any(np.isnan(r) or r < 0 for r in (self.r_safe, self.r_unsafe)):
            raise ValueError("Detector radii must be nonnegative, possibly +inf")
        if self.aggregation not in {"nearest", "mean", "exponential"}:
            raise ValueError("Unknown aggregation")
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("Positive frozen exponential bandwidth required")


@dataclass(frozen=True)
class Decision:
    d_safe: float
    d_unsafe: float
    safe_flag: bool
    unsafe_flag: bool

    @property
    def flagged(self) -> bool:
        return self.safe_flag or self.unsafe_flag


@dataclass(frozen=True)
class RepairChoice:
    status: str
    schedule: SteeringSchedule
    anchor: Anchor | None = None
    decision: Decision | None = None
    evaluated: int = 0


@dataclass(frozen=True)
class Sampling:
    steps: int = 64
    method: Literal["ode", "sde"] = "ode"
    warm_start: Literal["cold", "shifted"] = "cold"

    def __post_init__(self) -> None:
        if type(self.steps) is not int or not 1 <= self.steps <= 10000:
            raise ValueError("Sampling steps must be in [1,10000]")
        if self.method not in {"ode", "sde"} or self.warm_start not in {
            "cold",
            "shifted",
        }:
            raise ValueError("Unknown sampling or warm-start protocol")


@dataclass(frozen=True)
class PreviousReference:
    context: Context
    positions: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", frozen_array(self.positions, ndim=2))
        if self.positions.shape != (self.context.horizon + 1, 2):
            raise ValueError("Invalid old reference shape")

    def shifted(self, current: Context) -> np.ndarray:
        offset = current.k - self.context.k
        if (
            self.context.episode != current.episode
            or not 0 < offset <= self.context.horizon
            or self.context.dt != current.dt
            or self.context.horizon != current.horizon
            or self.context.compatibility != current.compatibility
            or self.context.goal != current.goal
        ):
            raise ValueError("Old reference has incompatible episode, time or horizon")
        remaining = self.positions[offset:]
        return np.concatenate((remaining, np.repeat(remaining[-1:], offset, axis=0)))


@dataclass
class ControlState:
    """Owned by one pipeline invocation; never persisted as a frozen policy."""

    context: Context | None
    started: float
    deadline: float
    clock: Callable[[], float]
    failure: str | None = None
    signature: np.ndarray | None = None
    decision: Decision | None = None
    repair: RepairChoice | None = None
    reference: np.ndarray | None = None
    sampled_reference: np.ndarray | None = None
    reference_clearance: float | None = None
    mpc: TrackingResult | None = None
    control: np.ndarray | None = None
    released: bool = False
    fallback_reason: str | None = None
    latency: float = 0.0

    def check_deadline(self) -> None:
        if self.clock() >= self.deadline:
            raise TimeoutError("Control deadline exceeded")
