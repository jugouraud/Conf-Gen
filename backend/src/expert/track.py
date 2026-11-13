"""Closed ellipse and Silverstone-inspired circuits with metric track geometry."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline

# Stylized Silverstone GP driving order: Abbey/Farm, Village/Loop/Aintree,
# Brooklands/Luffield/Woodcote, Copse/Maggotts/Becketts/Chapel,
# Hangar/Stowe/Vale/Club. Dimensionless points; not survey/GPS coordinates.
_SILVERSTONE_KNOTS = np.array([
    (-6.2, -2.0), (-4.1, 0.5), (-2.7, 0.75), (-1.1, 1.45),
    (-0.2, 1.65), (0.10, 0.25), (0.65, 0.40), (1.0, 2.0),
    (-3.45, 4.70), (-4.6, 3.4), (-5.25, 3.5), (-5.0, 4.45),
    (-3.85, 6.55), (3.2, 6.9), (4.3, 5.3), (4.9, 2.15),
    (5.0, 1.0), (5.35, 0.5), (5.0, -0.75), (5.5, -1.5),
    (3.7, -2.7), (0.2, -7.1), (-0.7, -7.25), (-3.6, -4.95),
    (-4.6, -4.0), (-5.4, -4.4), (-6.7, -4.15), (-6.9, -3.5),
    (-6.45, -2.55),
], dtype=float) / (7.0, 8.0)


@dataclass(frozen=True)
class TrackSpec:
    center: tuple[float, float] = (0.0, 0.0)
    radii: tuple[float, float] = (5.0, 3.5)
    left_width: float = 1.35
    right_width: float = 1.35
    resolution: int = 256
    layout: str = "ellipse"  # Default preserves pre-existing serialized datasets.

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        radii = np.asarray(self.radii, dtype=float)
        if (
            center.shape != (2,) or radii.shape != (2,)
            or not np.isfinite(center).all() or not np.isfinite(radii).all()
            or np.any(radii <= 0) or np.max(radii) > 100
            or type(self.resolution) is not int or not 64 <= self.resolution <= 1024
            or not isinstance(self.layout, str) or self.layout not in {"ellipse", "silverstone"}
        ):
            raise ValueError("Invalid fixed-track geometry")
        for width in (self.left_width, self.right_width):
            if isinstance(width, bool) or not np.isfinite(width) or width <= 0:
                raise ValueError("Invalid track width")
        # An ellipse's minimum curvature radius is min(radii)^2/max(radii).
        # Restrict offset widths below this to avoid a self-folding track tube.
        if (self.layout == "ellipse"
                and max(self.left_width, self.right_width) >= min(radii) ** 2 / max(radii)):
            raise ValueError("Track width exceeds its minimum curvature radius")
        object.__setattr__(self, "center", tuple(center.tolist()))
        object.__setattr__(self, "radii", tuple(radii.tolist()))


class ClosedTrack:
    """Metre-valued s modulo length; positive lateral is left of travel."""

    def __init__(self, spec: TrackSpec):
        self.spec = spec
        if spec.layout == "ellipse":
            theta = np.linspace(0, 2 * np.pi, spec.resolution + 1)
            points = np.column_stack((
                spec.center[0] + spec.radii[0] * np.cos(theta),
                spec.center[1] + spec.radii[1] * np.sin(theta),
            ))
        else:
            knots = _SILVERSTONE_KNOTS * spec.radii + spec.center
            closed = np.vstack((knots, knots[0]))
            chord = np.linalg.norm(np.diff(closed, axis=0), axis=1)
            arc = np.r_[0.0, np.cumsum(chord)]
            spline = CubicSpline(arc, closed, axis=0, bc_type="periodic")
            parameters = np.linspace(0.0, arc[-1], spec.resolution + 1)
            points = spline(parameters)
            # The track tube must not fold into the interior of a tight turn.
            samples = np.linspace(0.0, arc[-1], 2049)
            velocity, acceleration = spline(samples, 1), spline(samples, 2)
            curvature = np.abs(
                velocity[:, 0] * acceleration[:, 1]
                - velocity[:, 1] * acceleration[:, 0]
            ) / np.linalg.norm(velocity, axis=1) ** 3
            if np.max(curvature) * max(spec.left_width, spec.right_width) >= 1:
                raise ValueError("Track width exceeds its minimum curvature radius")
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

    def reference(
        self, start: np.ndarray, goal: np.ndarray, progress: np.ndarray
    ) -> np.ndarray:
        """Minimum-jerk lateral reference between arbitrary points on the road.

        `progress` is the quintic time law; the optimal lateral offset is
        d*(q) = d0 + (d1 - d0) * progress(q), with zero endpoint derivatives.
        Small corrections anchor points whose polyline projection is ambiguous
        near corners without a sharp first or last physical step.
        """
        endpoints = np.stack((start, goal))
        arcs, laterals = self.project(endpoints)
        distance = (float(arcs[1]) - float(arcs[0])) % self.length
        s = float(arcs[0]) + distance * progress
        d_star = laterals[0] + (laterals[1] - laterals[0]) * progress
        path = self.point(s) + d_star[:, None] * self.normal(s)
        path += (1 - progress)[:, None] * (endpoints[0] - path[0])
        path += progress[:, None] * (endpoints[1] - path[-1])
        return path

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
