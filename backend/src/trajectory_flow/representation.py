"""Smooth endpoint-anchored trajectory coordinates for conditional flow matching."""

from collections.abc import Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from backend.src.control.contracts import Context
from backend.src.expert.track import TrackSpec, track_for


class SplineRepresentation:
    """Latent [start_xy, goal_xy, interior_xy_spline_coefficients].

    The first four coordinates are fixed context data, not random variables.
    Cubic-spline residuals are measured from a forward track centerline when a
    fixed track is available, otherwise from a straight endpoint interpolation.
    Output is always (H+1,2) in metres; the internal spline uses no extrapolation.
    """

    def __init__(self, horizon: int, knots: int, track: dict | None) -> None:
        self.horizon = horizon
        self.knots = min(knots, horizon - 1)
        self.track = None if track is None else track_for(TrackSpec(**track))
        phase = np.linspace(0.0, 1.0, horizon + 1)
        nodes = np.linspace(0.0, 1.0, self.knots + 2)
        identity = np.eye(self.knots + 2)
        self.basis = CubicSpline(nodes, identity, axis=0, bc_type="clamped")(phase)[:, 1:-1]
        self.basis[[0, -1]] = 0.0
        self.projector = np.linalg.pinv(self.basis[1:-1])
        self.progress = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
        self.dimension = 4 + 2 * self.knots
        for values in (self.basis, self.projector, self.progress):
            values.setflags(write=False)

    def _baseline(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if self.track is None:
            path = start + self.progress[:, None] * (goal - start)
        else:
            arclength, _ = self.track.project(np.stack((start, goal)))
            length = (float(arclength[1]) - float(arclength[0])) % self.track.length
            path = self.track.point(float(arclength[0]) + length * self.progress)
        path[0], path[-1] = start, goal
        return path

    def encode(
        self, targets: np.ndarray, contexts: Sequence[Context], *,
        check_endpoints: bool = True,
    ) -> np.ndarray:
        targets = np.asarray(targets, dtype=np.float64)
        if (targets.shape != (len(contexts), self.horizon + 1, 2)
                or not np.isfinite(targets).all()):
            raise ValueError("Invalid spline training target shape or values")
        result = np.empty((len(contexts), self.dimension), dtype=np.float64)
        for index, (target, context) in enumerate(zip(targets, contexts, strict=True)):
            start = np.asarray(context.state[:2], dtype=np.float64)
            goal = np.asarray(context.goal, dtype=np.float64)
            if check_endpoints and (np.linalg.norm(target[0] - start) > 1e-4
                                    or np.linalg.norm(target[-1] - goal) > 1e-2):
                raise ValueError("Training target disagrees with conditioned endpoints")
            base = self._baseline(start, goal)
            coefficients = self.projector @ (target - base)[1:-1]
            result[index] = np.r_[start, goal, coefficients.ravel()]
        return result

    def decode(self, latents: np.ndarray) -> np.ndarray:
        latents = np.asarray(latents, dtype=np.float64)
        if (latents.ndim != 2 or latents.shape[1] != self.dimension
                or not np.isfinite(latents).all()):
            raise ValueError("Invalid spline latent shape or values")
        paths = np.empty((len(latents), self.horizon + 1, 2), dtype=np.float64)
        for index, latent in enumerate(latents):
            start, goal = latent[:2], latent[2:4]
            residual = self.basis @ latent[4:].reshape(self.knots, 2)
            paths[index] = self._baseline(start, goal) + residual
            paths[index, 0], paths[index, -1] = start, goal
        return paths
