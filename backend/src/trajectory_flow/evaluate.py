"""Diagnostic generator quality on validation scenes; never a safety certificate."""

import numpy as np

from backend.src.mpc.geometry import minimum_clearance
from backend.src.trajectory_flow.data import PreparedTraining
from backend.src.trajectory_flow.model import TrajectoryFlowModel
from backend.src.trajectory_flow.sample import run_flow_sampling


def run_flow_evaluation(
    model: TrajectoryFlowModel,
    prepared: PreparedTraining,
    *,
    seed: int,
    steps: int,
    solver: str,
    footprint_radius: float,
) -> dict:
    """Validation-only ADE/FDE, clearance, collision and sampler latency."""
    errors, finals, clearances, latencies = [], [], [], []
    for index, row in enumerate(prepared.validation):
        sample = run_flow_sampling(
            model, row.context, row.guidance, seed=seed + index, steps=steps, solver=solver
        )
        displacement = np.linalg.norm(sample.reference - row.target, axis=1)
        errors.append(float(displacement.mean()))
        finals.append(float(displacement[-1]))
        clearances.append(minimum_clearance(sample.reference, row.context.obstacles, footprint_radius))
        latencies.append(sample.latency_seconds)
    # These metrics do not reuse detector or output-calibration episodes.
    return {
        "split": "training_group_heldout_validation",
        "examples": len(errors),
        "ade_m": float(np.mean(errors)),
        "fde_m": float(np.mean(finals)),
        "minimum_surface_clearance_m": (min(clearances) if np.isfinite(min(clearances)) else None),
        "collision_fraction": float(np.mean(np.asarray(clearances) < 0)),
        "mean_sample_latency_seconds": float(np.mean(latencies)),
        "solver": solver,
        "steps": steps,
        "not_a_safety_certificate": True,
    }
