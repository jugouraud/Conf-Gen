"""Diagnostic generator quality on validation scenes; never a safety certificate."""

import numpy as np

from backend.src.mpc.geometry import minimum_clearance
from backend.src.trajectory_flow.data import PreparedTraining
from backend.src.trajectory_flow.model import TrajectoryFlowModel
from backend.src.trajectory_flow.sample import run_flow_sampling

# Surface distance on the expert trajectory; selected error is in physical metres.
OBSTACLE_LOCAL_WINDOW_M = 0.5


def _local_error_stats(paths: list[np.ndarray]) -> dict:
    values = [path for path in paths if len(path)]
    if not values:
        return {"obstacle_local_ade_m": None, "obstacle_local_max_error_m": None,
                "obstacle_local_points": 0}
    errors = np.concatenate(values)
    return {"obstacle_local_ade_m": float(errors.mean()),
            "obstacle_local_max_error_m": float(errors.max()),
            "obstacle_local_points": len(errors)}


def run_flow_evaluation(
    model: TrajectoryFlowModel,
    prepared: PreparedTraining,
    *,
    seed: int,
    steps: int,
    solver: str,
    footprint_radius: float,
) -> dict:
    """Validation-only ADE/FDE, obstacle-local accuracy, clearance and latency."""
    errors, finals, clearances, latencies = [], [], [], []
    local_errors: list[np.ndarray] = []
    regimes: dict[str, list[tuple[float, float, np.ndarray]]] = {}
    for index, row in enumerate(prepared.validation):
        sample = run_flow_sampling(
            model, row.context, row.guidance, seed=seed + index, steps=steps, solver=solver
        )
        displacement = np.linalg.norm(sample.reference - row.target, axis=1)
        errors.append(float(displacement.mean()))
        finals.append(float(displacement[-1]))
        clearances.append(minimum_clearance(sample.reference, row.context.obstacles, footprint_radius))
        latencies.append(sample.latency_seconds)
        nearby = np.zeros(len(row.target), dtype=bool)
        for obstacle in row.context.obstacles:
            centers = np.asarray(obstacle.centers)
            surface_distance = (np.linalg.norm(row.target - centers, axis=1)
                                - obstacle.radius - footprint_radius)
            nearby |= surface_distance <= OBSTACLE_LOCAL_WINDOW_M
        local_errors.append(displacement[nearby])
        if model.codec.track is not None:
            endpoints = np.stack((row.context.state[:2], row.context.goal))
            _, offsets = model.codec.track.project(endpoints)
            regime = "centerline" if np.max(np.abs(offsets)) < 1e-4 else "arbitrary_endpoints"
            regimes.setdefault(regime, []).append((errors[-1], clearances[-1], local_errors[-1]))
    finite_clearances = np.asarray(clearances)[np.isfinite(clearances)]
    # These metrics do not reuse detector or output-calibration episodes.
    return {
        "by_endpoint_regime": {
            name: {
                "examples": len(values),
                "ade_m": float(np.mean([value[0] for value in values])),
                "collision_fraction": float(np.mean([value[1] < 0 for value in values])),
                **_local_error_stats([value[2] for value in values]),
            }
            for name, values in regimes.items()
        },
        "split": "training_group_heldout_validation",
        "examples": len(errors),
        "ade_m": float(np.mean(errors)),
        "fde_m": float(np.mean(finals)),
        "minimum_surface_clearance_m": (min(clearances) if np.isfinite(min(clearances)) else None),
        "p05_surface_clearance_m": (float(np.quantile(finite_clearances, 0.05))
                                    if len(finite_clearances) else None),
        "median_surface_clearance_m": (float(np.median(finite_clearances))
                                       if len(finite_clearances) else None),
        "collision_fraction": float(np.mean(np.asarray(clearances) < 0)),
        "obstacle_local_window_m": OBSTACLE_LOCAL_WINDOW_M,
        **_local_error_stats(local_errors),
        "mean_sample_latency_seconds": float(np.mean(latencies)),
        "solver": solver,
        "steps": steps,
        "not_a_safety_certificate": True,
    }
