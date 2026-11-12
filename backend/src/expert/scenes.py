"""Physical-time scene contracts and deterministic group ownership."""

from dataclasses import asdict, dataclass

import numpy as np

from backend.src.expert.config import SPLITS, DatasetSettings
from backend.src.expert.track import TrackSpec
from backend.src.mpc.geometry import frozen_array


@dataclass(frozen=True)
class TimedObstacle:
    obstacle_id: str
    times: np.ndarray
    centers: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", frozen_array(self.times, ndim=1))
        object.__setattr__(self, "centers", frozen_array(self.centers, ndim=2))
        if (
            not isinstance(self.obstacle_id, str)
            or not self.obstacle_id
            or not 2 <= len(self.times) <= 1024
            or np.any(np.diff(self.times) <= 0)
            or self.centers.shape != (len(self.times), 2)
            or not np.isfinite(self.radius)
            or self.radius < 0
        ):
            raise ValueError("Invalid timed obstacle")

    def at(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times)
        if (
            not np.isfinite(times).all()
            or times.min() < self.times[0] - 1e-10
            or times.max() > self.times[-1] + 1e-10
        ):
            raise ValueError("Obstacle prediction does not cover physical timestamps")
        return np.column_stack(
            [np.interp(times, self.times, self.centers[:, axis]) for axis in range(2)]
        )


@dataclass(frozen=True)
class Scene:
    scene_id: str
    group_id: str
    map_id: str
    bounds: tuple[float, float, float, float]
    start_time: float
    initial_state: tuple[float, ...]
    initial_velocity: tuple[float, float]
    goal: tuple[float, float]
    obstacles: tuple[TimedObstacle, ...]
    goal_heading: float | None = None
    track: TrackSpec | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not 1 <= len(v) <= 200
            for v in (self.scene_id, self.group_id, self.map_id)
        ):
            raise ValueError("Invalid scene/group/map identity")
        for name in ("bounds", "initial_state", "initial_velocity", "goal"):
            object.__setattr__(
                self, name, tuple(frozen_array(getattr(self, name), ndim=1))
            )
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        if isinstance(self.track, dict):
            object.__setattr__(self, "track", TrackSpec(**self.track))
        elif self.track is not None and not isinstance(self.track, TrackSpec):
            raise ValueError("Invalid fixed-track descriptor")
        if (
            len(self.bounds) != 4
            or self.bounds[0] >= self.bounds[1]
            or self.bounds[2] >= self.bounds[3]
            or len(self.initial_velocity) != 2
            or len(self.goal) != 2
            or not np.isfinite(self.start_time)
            or len(self.obstacles) > 32
            or len({o.obstacle_id for o in self.obstacles}) != len(self.obstacles)
            or (self.goal_heading is not None and not np.isfinite(self.goal_heading))
        ):
            raise ValueError("Invalid scene dimensions, time or obstacle IDs")

    @classmethod
    def from_record(cls, row: dict) -> "Scene":
        expected = {
            "scene_id",
            "group_id",
            "map_id",
            "bounds",
            "start_time",
            "initial_state",
            "initial_velocity",
            "goal",
            "obstacles",
            "goal_heading",
        }
        if not isinstance(row, dict) or set(row) not in (expected, expected | {"track"}):
            raise ValueError("Scene fields differ from supported schema")
        if not isinstance(row["obstacles"], list) or len(row["obstacles"]) > 32:
            raise ValueError("Invalid obstacle collection")
        obstacles = []
        for obstacle in row["obstacles"]:
            if set(obstacle) != {"obstacle_id", "times", "centers", "radius"}:
                raise ValueError("Obstacle fields differ from schema v1")
            obstacles.append(TimedObstacle(**obstacle))
        return cls(**dict(row, obstacles=tuple(obstacles)))

    def to_record(self) -> dict:
        return asdict(self)


def assign_groups(group_ids: list[str], settings: DatasetSettings) -> dict[str, str]:
    """Allocate independent groups before alternatives/windows, never individual rows."""
    groups = np.array(sorted(set(group_ids)), dtype=object)
    if len(groups) < len(SPLITS):
        raise ValueError("At least four independent groups are required")
    rng = np.random.default_rng(settings.seed)
    rng.shuffle(groups)
    sizes = np.asarray(settings.split_weights) * len(groups)
    counts = np.floor(sizes).astype(int)
    order = np.argsort(-(sizes - np.floor(sizes)), kind="stable")
    counts[order[: len(groups) - counts.sum()]] += 1
    # Tiny development datasets still reserve an independent group per split.
    for empty in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        counts[donor] -= 1
        counts[empty] = 1
    assignment, start = {}, 0
    for name, count in zip(SPLITS, counts, strict=True):
        assignment.update({group: name for group in groups[start : start + count]})
        start += count
    return assignment
