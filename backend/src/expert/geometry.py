"""Timed footprint checks and passage descriptors shared by planner and validator."""

import numpy as np

from backend.src.expert.config import Guidance
from backend.src.expert.scenes import Scene, TimedObstacle
from backend.src.mpc.geometry import Obstacle, swept_clearances


def synchronize_path(
    times: np.ndarray, positions: np.ndarray, obstacle: TimedObstacle
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Include obstacle turns inside physical intervals; never merely sample knots."""
    interior = obstacle.times[
        (obstacle.times > times[0]) & (obstacle.times < times[-1])
    ]
    common = np.unique(np.concatenate((times, interior)))
    path = np.column_stack(
        [np.interp(common, times, positions[:, axis]) for axis in range(2)]
    )
    return common, path, obstacle.at(common)


def timed_clearances(
    times: np.ndarray,
    positions: np.ndarray,
    obstacles: tuple[TimedObstacle, ...],
    radius: float,
) -> np.ndarray:
    values = []
    for obstacle in obstacles:
        _, path, centers = synchronize_path(times, positions, obstacle)
        values.extend(
            swept_clearances(
                path, (Obstacle(tuple(map(tuple, centers)), obstacle.radius),), radius
            )
        )
    return np.asarray(values)


def boundary_clearances(
    positions: np.ndarray, bounds: tuple, radius: float
) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    return np.column_stack(
        (
            positions[:, 0] - xmin - radius,
            xmax - positions[:, 0] - radius,
            positions[:, 1] - ymin - radius,
            ymax - positions[:, 1] - radius,
        )
    ).ravel()


def passage_laterals(
    scene: Scene, times: np.ndarray, positions: np.ndarray, guidance: Guidance
) -> np.ndarray:
    """Signed lateral offset at crossings of the selected moving obstacle's plane.

    Positive is left relative to initial-to-goal direction. All relative forward
    crossings must satisfy the requested side; no crossing is an invalid passage.
    """
    obstacle = scene.obstacles[guidance.obstacle_index]
    _, path, centers = synchronize_path(times, positions, obstacle)
    axis = np.asarray(scene.goal) - np.asarray(scene.initial_state[:2])
    axis /= np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])
    relative = path - centers
    longitudinal = relative @ axis
    lateral = relative @ normal
    crossings = []
    for i in range(len(path) - 1):
        if (
            longitudinal[i] <= 0 <= longitudinal[i + 1]
            and longitudinal[i + 1] > longitudinal[i]
        ):
            fraction = -longitudinal[i] / (longitudinal[i + 1] - longitudinal[i])
            crossings.append((1 - fraction) * lateral[i] + fraction * lateral[i + 1])
    return np.asarray(crossings)
