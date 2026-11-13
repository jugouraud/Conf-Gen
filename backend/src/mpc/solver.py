"""General linear-dynamics tracking MPC, independent of diffusion/detection."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from backend.src.mpc.dynamics import TrackingConfig
from backend.src.mpc.geometry import Obstacle, frozen_array, swept_clearances


class TrackingFailure(RuntimeError):
    """Solver failed or returned a point outside the checked feasible region."""


@dataclass(frozen=True)
class TrackingResult:
    states: np.ndarray
    controls: np.ndarray
    cost: float
    tracking_rmse: float
    minimum_clearance: float
    iterations: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", frozen_array(self.states, ndim=2))
        object.__setattr__(self, "controls", frozen_array(self.controls, ndim=2))


def solve_tracking(
    initial: np.ndarray,
    reference: np.ndarray,
    obstacles: tuple[Obstacle, ...],
    config: TrackingConfig,
    *,
    check_deadline: Callable[[], None],
    previous_control: np.ndarray | None = None,
) -> TrackingResult:
    """H controls; position stage/terminal cost plus input cost u'Ru.

    SLSQP is local: failure is not proof of infeasibility. All state/input bounds
    and swept collision constraints are rechecked before returning a result.
    The caller owns the physical deadline and handles TrackingFailure/TimeoutError.
    """
    check_deadline()
    dynamics = config.dynamics
    initial, reference = frozen_array(initial, ndim=1), frozen_array(reference, ndim=2)
    horizon = len(reference) - 1
    if (
        initial.shape != (len(dynamics.a),)
        or reference.shape[1:] != (2,)
        or not 1 <= horizon <= 128
    ):
        raise ValueError("Invalid MPC state/reference dimensions")
    if len(obstacles) > 128:
        raise ValueError("Too many predicted obstacles")
    shape = (horizon, dynamics.b.shape[1])
    if config.max_input_delta is not None:
        if previous_control is None:
            raise ValueError("Input slew constraint requires previous_control")
        previous_control = frozen_array(previous_control, ndim=1)
        if previous_control.shape != (shape[1],):
            raise ValueError("Previous control dimension differs from MPC input")

    def rollout(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        check_deadline()
        states = dynamics.rollout(initial, flat.reshape(shape))
        return states, states @ dynamics.c.T

    def objective(flat: np.ndarray) -> float:
        _, positions = rollout(flat)
        error, controls = positions - reference, flat.reshape(shape)
        return float(
            np.einsum("hi,ij,hj->", error, config.q, error)
            + np.einsum("hi,ij,hj->", controls, config.r, controls)
        )

    def constraints(flat: np.ndarray) -> np.ndarray:
        states, positions = rollout(flat)
        clearances = swept_clearances(positions, obstacles, config.footprint_radius)
        commands = flat.reshape(shape)
        residuals = [
            (states - config.state_lower).ravel(),
            (config.state_upper - states).ravel(),
            clearances - config.required_clearance - config.tracking_uncertainty,
        ]
        if config.max_input_norm is not None:
            residuals.append(config.max_input_norm**2 - np.sum(commands**2, axis=1))
        if config.max_input_delta is not None:
            changes = np.diff(np.vstack((previous_control, commands)), axis=0)
            residuals.append(config.max_input_delta**2 - np.sum(changes**2, axis=1))
        return np.concatenate(residuals)

    # The same reference-derived initialization is used across policies; no
    # hidden solver state carries between methods or independent episodes.
    guess, current = [], initial
    for position in reference[1:]:
        control = np.linalg.lstsq(
            dynamics.c @ dynamics.b,
            position - dynamics.c @ dynamics.a @ current,
            rcond=None,
        )[0]
        control = np.clip(control, config.input_lower, config.input_upper)
        guess.append(control)
        current = dynamics.a @ current + dynamics.b @ control
    bounds = list(
        zip(
            np.tile(config.input_lower, horizon),
            np.tile(config.input_upper, horizon),
            strict=True,
        )
    )
    result = minimize(
        objective,
        np.asarray(guess).ravel(),
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "ineq", "fun": constraints},
        options={"maxiter": config.max_iterations, "ftol": config.tolerance},
    )
    check_deadline()
    if not result.success or not np.isfinite(result.x).all():
        raise TrackingFailure("solver_failure")
    controls = result.x.reshape(shape)
    states, positions = rollout(result.x)
    residuals = constraints(result.x)
    if (
        not np.isfinite(residuals).all()
        or np.min(residuals) < -config.tolerance
        or np.any(controls < config.input_lower - config.tolerance)
        or np.any(controls > config.input_upper + config.tolerance)
    ):
        raise TrackingFailure("solver_constraint_failure")
    clearance = float(
        np.min(
            swept_clearances(positions, obstacles, config.footprint_radius),
            initial=np.inf,
        )
    )
    return TrackingResult(
        states,
        controls,
        objective(result.x),
        float(np.sqrt(np.mean((positions - reference) ** 2))),
        clearance,
        int(result.nit),
    )
