"""Exact swept clearance for piecewise-linear centres and circular footprints."""

from dataclasses import dataclass

import numpy as np


def frozen_array(value: object, *, ndim: int | None = None) -> np.ndarray:
    """Own finite FP64 data in an immutable byte buffer (including array views)."""
    array = np.asarray(value, dtype=np.float64)
    if (ndim is not None and array.ndim != ndim) or not np.isfinite(array).all():
        raise ValueError("Expected finite array with matching dimensions")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True)
class Obstacle:
    """Predicted centre at i=0..H, metres; linear interpolation between samples."""

    centers: tuple[tuple[float, float], ...]
    radius: float

    def __post_init__(self) -> None:
        centers = frozen_array(self.centers, ndim=2)
        if centers.shape[1:] != (2,) or len(centers) < 2:
            raise ValueError("Obstacle needs at least two 2D predictions")
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("Invalid obstacle radius")
        object.__setattr__(self, "centers", tuple(map(tuple, centers.tolist())))


def swept_clearances(
    positions: np.ndarray, obstacles: tuple[Obstacle, ...], footprint_radius: float
) -> np.ndarray:
    """Physical surface clearance per obstacle/segment; negative means overlap.

    Both centres move on the same physical interval. Minimizing their relative
    line segment catches collisions missed by waypoint-only checks.
    """
    points = np.asarray(positions, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or not np.isfinite(points).all()
        or not np.isfinite(footprint_radius)
        or footprint_radius < 0
    ):
        raise ValueError("Invalid path or footprint")
    clearances = []
    for obstacle in obstacles:
        centers = np.asarray(obstacle.centers)
        if centers.shape != points.shape:
            raise ValueError("Obstacle and reference horizons differ")
        relative = points - centers
        start = relative[:-1]
        direction = np.diff(relative, axis=0)
        denominator = np.sum(direction * direction, axis=1)
        fraction = np.divide(
            -np.sum(start * direction, axis=1),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        ).clip(0, 1)
        closest = start + fraction[:, None] * direction
        clearances.extend(
            np.linalg.norm(closest, axis=1) - footprint_radius - obstacle.radius
        )
    return np.asarray(clearances, dtype=np.float64)


def minimum_clearance(
    positions: np.ndarray, obstacles: tuple[Obstacle, ...], footprint_radius: float
) -> float:
    return float(
        np.min(swept_clearances(positions, obstacles, footprint_radius), initial=np.inf)
    )
