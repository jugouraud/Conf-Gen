"""Fixed closed elliptical circuit with arc-length and footprint geometry."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class TrackSpec:
    center: tuple[float, float] = (0.0, 0.0)
    radii: tuple[float, float] = (5.0, 3.5)
    left_width: float = 1.35
    right_width: float = 1.35
    resolution: int = 256

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        radii = np.asarray(self.radii, dtype=float)
        if (
            center.shape != (2,) or radii.shape != (2,)
            or not np.isfinite(center).all() or not np.isfinite(radii).all()
            or np.any(radii <= 0) or np.max(radii) > 100
            or type(self.resolution) is not int or not 64 <= self.resolution <= 1024
        ):
            raise ValueError("Invalid fixed-track geometry")
        for width in (self.left_width, self.right_width):
            if isinstance(width, bool) or not np.isfinite(width) or width <= 0:
                raise ValueError("Invalid track width")
        # An ellipse's minimum curvature radius is min(radii)^2/max(radii).
        # Restrict offset widths below this to avoid a self-folding track tube.
        if max(self.left_width, self.right_width) >= min(radii) ** 2 / max(radii):
            raise ValueError("Track width exceeds its minimum curvature radius")
        object.__setattr__(self, "center", tuple(center.tolist()))
        object.__setattr__(self, "radii", tuple(radii.tolist()))


class ClosedTrack:
    """Metre-valued s modulo length; positive lateral is left of travel."""

    def __init__(self, spec: TrackSpec):
        self.spec = spec
        theta = np.linspace(0, 2 * np.pi, spec.resolution + 1)
        points = np.column_stack((
            spec.center[0] + spec.radii[0] * np.cos(theta),
            spec.center[1] + spec.radii[1] * np.sin(theta),
        ))
        delta = np.diff(points, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        self.lengths = lengths
        self.arc = np.r_[0.0, np.cumsum(lengths)]
        self.length = float(self.arc[-1])
        self.points = points
        self.tangents = delta / lengths[:, None]

    def _segments(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.mod(np.asarray(s, dtype=float), self.length)
        index = np.searchsorted(self.arc[1:], s, side="right")
        fraction = (s - self.arc[index]) / self.lengths[index]
        return index, fraction

    def point(self, s: np.ndarray | float) -> np.ndarray:
        index, fraction = self._segments(np.asarray(s))
        return self.points[index] + np.expand_dims(fraction, -1) * (
            self.points[index + 1] - self.points[index]
        )

    def tangent(self, s: np.ndarray | float) -> np.ndarray:
        index, _ = self._segments(np.asarray(s))
        return self.tangents[index]

    def normal(self, s: np.ndarray | float) -> np.ndarray:
        tangent = self.tangent(s)
        return np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)

    def project(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest polyline projection, signed lateral offset; input (...,2)."""
        positions = np.asarray(positions, dtype=float)
        if positions.shape[-1] != 2 or not np.isfinite(positions).all():
            raise ValueError("Invalid Cartesian track coordinates")
        flat = positions.reshape(-1, 2)
        starts = self.points[:-1]
        displacement = self.points[1:] - starts
        relative = flat[:, None, :] - starts[None, :, :]
        fraction = np.clip(
            np.sum(relative * displacement[None, :, :], axis=-1)
            / self.lengths[None, :] ** 2, 0.0, 1.0
        )
        offsets = relative - fraction[..., None] * displacement[None, :, :]
        index = np.argmin(np.sum(offsets**2, axis=-1), axis=1)
        selection = np.arange(len(flat))
        near = offsets[selection, index]
        signed = np.sum(near * np.stack(
            (-self.tangents[index, 1], self.tangents[index, 0]), axis=-1
        ), axis=1)
        arc = self.arc[index] + fraction[selection, index] * self.lengths[index]
        return arc.reshape(positions.shape[:-1]), signed.reshape(positions.shape[:-1])

    def boundary_clearance(self, positions: np.ndarray, radius: float) -> np.ndarray:
        """Conservative continuous-segment clearance via <=0.08 m samples."""
        points = np.asarray(positions, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            raise ValueError("Invalid track path")
        if len(points) > 4096:
            raise ValueError("Path exceeds bounded boundary checking budget")
        dense = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            subdivisions = int(np.ceil(np.linalg.norm(end - start) / 0.08))
            if subdivisions > 512:
                raise ValueError("One segment is too long for certified checking")
            for k in range(1, subdivisions + 1):
                dense.append(start + (end - start) * (k / subdivisions))
        _, d = self.project(np.asarray(dense))
        # Distance-to-boundary is 1-Lipschitz; 0.04 m covers any missed
        # interior point between <=0.08 m adjacent checks.
        margin = 0.04 if len(points) > 1 else 0.0
        return np.minimum(self.spec.left_width - d, self.spec.right_width + d) - radius - margin


@lru_cache(maxsize=32)
def track_for(spec: TrackSpec) -> ClosedTrack:
    return ClosedTrack(spec)
