"""Integrate the selected drift field, then decode a physical reference."""

import numpy as np

from backend.src.control.contracts import ControlState, PreviousReference, Sampling
from backend.src.control.model import TrajectoryModel


def integrate_field(
    model: TrajectoryModel,
    state: ControlState,
    guidance: str,
    sampling: Sampling,
    seed: int,
    previous: PreviousReference | None = None,
) -> np.ndarray:
    """Euler / Euler-Maruyama, common grid and Brownian draws for paired seeds.

    Warm initialization is model-owned and occurs at t=0. A shifted decoded path
    is only an initializer input; it is never a substitute anchor drift.
    """
    model.assert_frozen()
    rng = np.random.default_rng(seed)
    warm = (
        previous.shifted(state.context)
        if previous is not None and sampling.warm_start == "shifted"
        else None
    )
    x = np.asarray(model.initialize(state.context, rng, warm), dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Invalid initializer output")
    dt = 1.0 / sampling.steps
    for index in range(sampling.steps):
        state.check_deadline()
        t = index * dt
        strength = state.repair.schedule(t)
        nominal = np.asarray(model.drift(x.copy(), t, state.context, guidance))
        velocity = nominal
        if strength:
            anchor = np.asarray(
                model.drift(x.copy(), t, state.context, state.repair.anchor.guidance)
            )
            if anchor.shape != x.shape:
                raise ValueError("Anchor field shape mismatch")
            velocity = (1 - strength) * nominal + strength * anchor
        if velocity.shape != x.shape:
            raise ValueError("Nominal field shape mismatch")
        # Diffusion coefficient is model-owned and never repaired.
        noise = (
            np.asarray(model.noise(x.copy(), t))
            if sampling.method == "sde"
            else np.zeros_like(x)
        )
        if noise.shape != x.shape or not np.isfinite(noise).all():
            raise ValueError("Noise coefficient shape/value mismatch")
        increment = rng.standard_normal(x.shape)
        x = x + dt * velocity + np.sqrt(dt) * noise * increment
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite diffusion sample")
    model.assert_frozen()
    return np.asarray(model.decode(x), dtype=np.float64)


def run_reference_sampling(
    state: ControlState,
    model: TrajectoryModel,
    guidance: str,
    sampling: Sampling,
    seed: int,
    previous: PreviousReference | None = None,
) -> ControlState:
    if state.failure:
        return state
    try:
        state.reference = integrate_field(
            model, state, guidance, sampling, seed, previous
        )
        state.sampled_reference = state.reference.copy()
        state.check_deadline()
    except TimeoutError:
        state.failure = "timeout"
    except (ValueError, RuntimeError, ArithmeticError):
        state.failure = "sampling_failure"
    return state
