"""Analytic trajectory field and independent moving-disc scene generator.

This module contains no learned weights and must not be described as a trained
diffusion planner. It exercises the same drift/sampler contract as a real adapter.
"""

from dataclasses import dataclass

import numpy as np

from backend.src.control.contracts import Context
from backend.src.mpc.geometry import Obstacle


@dataclass(frozen=True)
class SyntheticTrajectoryModel:
    revision: str = "synthetic-disc-field-v1"

    def assert_frozen(self) -> None:
        return None  # Frozen dataclass, analytic model, no fitted parameters.

    def encode(self, obs: Context) -> Context:
        return obs

    def target(self, c: Context, a: str) -> np.ndarray:
        amplitudes = {"direct": 0.0, "upper": 1.5, "lower": -1.5}
        if a not in amplitudes:
            raise ValueError("Unknown frozen catalogue guidance")
        phase = np.linspace(0, 1, c.horizon + 1)
        start, goal = np.asarray(c.state), np.asarray(c.goal)
        target = (1 - phase[:, None]) * start + phase[:, None] * goal
        remaining = min(1.0, np.linalg.norm(goal - start) / 2.0)
        target[:, 1] += amplitudes[a] * remaining * np.sin(np.pi * phase)
        return target

    def baseline(self, x: np.ndarray, t: float) -> np.ndarray:
        return -0.2 * x

    def drift(self, x: np.ndarray, t: float, c: Context, a: str) -> np.ndarray:
        # The continuous ODE from zero ends at target; discretization is retained
        # and measured, rather than replaced by directly returning that target.
        return self.baseline(x, t) + self.target(c, a) * (0.2 / (1 - np.exp(-0.2)))

    def decode(self, x_final: np.ndarray) -> np.ndarray:
        return x_final.copy()

    def initialize(
        self, c: Context, rng: np.random.Generator, warm: np.ndarray | None
    ) -> np.ndarray:
        x = rng.normal(0, 0.005, (c.horizon + 1, 2))
        if warm is not None:
            x += 0.01 * (warm - self.target(c, "direct"))
        x[[0, -1]] = 0
        return x

    def noise(self, x: np.ndarray, t: float) -> np.ndarray:
        coefficient = np.full_like(x, 0.005)
        coefficient[[0, -1]] = 0
        return coefficient


@dataclass(frozen=True)
class Scene:
    episode: str
    initial: tuple[float, float]
    goal: tuple[float, float]
    obstacle: tuple[float, float]
    velocity: tuple[float, float]
    radius: float
    guidance: str
    seed: int

    def observe(self, state: np.ndarray, k: int, horizon: int, dt: float) -> Context:
        times = (k + np.arange(horizon + 1)) * dt
        centers = np.asarray(self.obstacle) + times[:, None] * np.asarray(self.velocity)
        obstacle = Obstacle(tuple(map(tuple, centers)), self.radius)
        return Context(
            self.episode,
            k,
            tuple(state),
            self.goal,
            (obstacle,),
            dt,
            horizon,
            "disc-v1",
        )


def make_scene(episode: str, seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    offset = float(rng.uniform(-0.12, 0.12))
    return Scene(
        episode,
        (0.0, offset),
        (4.0, offset),
        (float(rng.uniform(1.8, 2.2)), offset + float(rng.uniform(-0.2, 0.2))),
        (0.0, float(rng.uniform(-0.035, 0.035))),
        float(rng.uniform(0.18, 0.25)),
        ("direct", "upper", "lower")[int(rng.integers(3))],
        seed,
    )
