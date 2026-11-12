"""Paired Euler/Heun integration for the learned deterministic trajectory field."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from backend.src.control.contracts import Context
from backend.src.trajectory_flow.model import TrajectoryFlowModel


@dataclass(frozen=True)
class FlowSample:
    reference: np.ndarray
    seed: int
    solver: str
    steps: int
    latency_seconds: float


def run_flow_sampling(
    model: TrajectoryFlowModel,
    observation: Context,
    guidance: str,
    *,
    seed: int,
    steps: int = 32,
    solver: str = "euler",
    warm_start: np.ndarray | None = None,
    anchor: str | None = None,
    steering: Callable[[float], float] | None = None,
) -> FlowSample:
    """Start at t=0; reuse seed for paired nominal/repaired comparisons.

    A warm start shifts the standard Gaussian initializer distribution; its use
    requires a new downstream detector and output calibration. Never substitute
    a historical scene for the live context.
    """
    if solver not in {"euler", "heun"} or type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("Sampling requires Euler/Heun and a bounded positive step count")
    if (anchor is None) != (steering is None):
        raise ValueError("Repair requires both compatible anchor and steering schedule")
    model.assert_frozen()
    context = model.encode(observation)
    warm = None if warm_start is None else torch.as_tensor(warm_start, dtype=torch.float32, device=model.target_mean.device)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    with torch.inference_mode():
        x = model.initialize(context, rng, warm)

        def velocity(state: torch.Tensor, t: float) -> torch.Tensor:
            nominal = model.drift(state, t, context, guidance)
            if anchor is None:
                return nominal
            strength = float(steering(t))
            if not np.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError("Steering strengths must lie in [0,1]")
            return (1 - strength) * nominal + strength * model.drift(state, t, context, anchor)

        h = 1.0 / steps
        for i in range(steps):
            t = i * h
            first = velocity(x, t)
            if solver == "heun":
                forecast = x + h * first
                second = velocity(forecast, min(1.0, t + h))
                x = x + 0.5 * h * (first + second)
            else:
                x = x + h * first
            if not bool(torch.isfinite(x).all()):
                raise ValueError("Nonfinite learned trajectory sample")
        reference = model.decode(x).detach().cpu().numpy().astype(np.float64)
    return FlowSample(reference, seed, solver, steps, time.perf_counter() - started)
