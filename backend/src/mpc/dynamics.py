"""Configured discrete linear dynamics and quadratic tracking/input costs."""

from dataclasses import dataclass

import numpy as np

from backend.src.mpc.geometry import frozen_array


@dataclass(frozen=True)
class LinearDynamics:
    """z[i+1]=A z[i]+B u[i]; C z[i] gives physical x,y in metres."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    dt: float

    def __post_init__(self) -> None:
        for name in ("a", "b", "c"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), ndim=2))
        n = len(self.a)
        if (
            not 1 <= n <= 32
            or self.a.shape != (n, n)
            or self.b.shape[0] != n
            or not 1 <= self.b.shape[1] <= 16
            or self.c.shape != (2, n)
            or not np.isfinite(self.dt)
            or self.dt <= 0
        ):
            raise ValueError("Invalid linear dynamics dimensions or physical dt")

    def rollout(self, initial: np.ndarray, controls: np.ndarray) -> np.ndarray:
        controls = np.asarray(controls, dtype=np.float64)
        initial = np.asarray(initial, dtype=np.float64)
        if (
            initial.shape != (len(self.a),)
            or controls.ndim != 2
            or controls.shape[1] != self.b.shape[1]
            or not np.isfinite(initial).all()
            or not np.isfinite(controls).all()
        ):
            raise ValueError("Invalid rollout dimensions or values")
        states = [initial]
        for control in controls:
            states.append(self.a @ states[-1] + self.b @ control)
        return np.stack(states)


@dataclass(frozen=True)
class TrackingConfig:
    dynamics: LinearDynamics
    q: np.ndarray
    r: np.ndarray
    state_lower: np.ndarray
    state_upper: np.ndarray
    input_lower: np.ndarray
    input_upper: np.ndarray
    footprint_radius: float = 0.15
    required_clearance: float = 0.1
    tracking_uncertainty: float = 0.05
    max_iterations: int = 100
    tolerance: float = 1e-7

    def __post_init__(self) -> None:
        for name in (
            "q",
            "r",
            "state_lower",
            "state_upper",
            "input_lower",
            "input_upper",
        ):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        n, m = self.dynamics.b.shape
        for name, size in (("state", n), ("input", m)):
            low, high = getattr(self, name + "_lower"), getattr(self, name + "_upper")
            if low.shape != (size,) or high.shape != (size,) or np.any(low >= high):
                raise ValueError("Invalid finite state/input bounds")
        for matrix, size in ((self.q, 2), (self.r, m)):
            if (
                matrix.shape != (size, size)
                or not np.allclose(matrix, matrix.T)
                or np.linalg.eigvalsh(matrix).min() <= 0
            ):
                raise ValueError(
                    "Tracking/input costs must be symmetric positive definite"
                )
        values = [
            self.footprint_radius,
            self.required_clearance,
            self.tracking_uncertainty,
            self.tolerance,
        ]
        if not np.isfinite(values).all() or min(values) < 0 or self.tolerance <= 0:
            raise ValueError("Invalid clearance, footprint, uncertainty or tolerance")
        if (
            type(self.max_iterations) is not int
            or not 1 <= self.max_iterations <= 10000
        ):
            raise ValueError("Invalid bounded solver iteration budget")
