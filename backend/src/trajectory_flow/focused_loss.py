"""Obstacle-localized geometric regularization for conditional flow matching.

The rectified-flow one-step clean estimate is x1_hat = xt + (1-t)*v_theta.
Spline residual differences decode linearly, so gradients never pass through the
NumPy inference decoder. This loss is a training heuristic, not a certificate.
"""

from dataclasses import dataclass

import numpy as np
import torch

from backend.src.trajectory_flow.data import FlowExample
from backend.src.trajectory_flow.model import TrajectoryFlowModel


@dataclass(frozen=True)
class LossGeometry:
    """Physical metres; indices are (batch, obstacle, physical step, xy)."""

    decoded_targets: torch.Tensor  # (B, H+1, 2)
    centers: torch.Tensor  # (B, O, H+1, 2)
    radii: torch.Tensor  # (B, O), obstacle + vehicle footprint in metres
    present: torch.Tensor  # (B, O), bool
    focus: torch.Tensor  # (B, H+1), in [0, 1], expert-target proximity

    def take(self, indices: list[int]) -> "LossGeometry":
        return LossGeometry(*(getattr(self, name)[indices] for name in (
            "decoded_targets", "centers", "radii", "present", "focus"
        )))


def prepare_loss_geometry(
    model: TrajectoryFlowModel,
    rows: tuple[FlowExample, ...],
    latents: torch.Tensor,
    *,
    footprint_radius: float,
    clearance_margin_m: float,
    focus_sigma_m: float,
) -> LossGeometry:
    """Precompute static geometries once; never use validation to fit statistics."""
    device, dtype = latents.device, latents.dtype
    with torch.no_grad():
        decoded = model.decode(latents).detach()
        count, steps = len(rows), model.architecture.horizon + 1
        maximum = max((len(row.context.obstacles) for row in rows), default=0)
        centers = torch.zeros((count, maximum, steps, 2), dtype=dtype, device=device)
        radii = torch.zeros((count, maximum), dtype=dtype, device=device)
        present = torch.zeros((count, maximum), dtype=torch.bool, device=device)
        focus = torch.zeros((count, steps), dtype=dtype, device=device)
        for i, row in enumerate(rows):
            for j, obstacle in enumerate(row.context.obstacles):
                positions = np.asarray(obstacle.centers, dtype=np.float64)
                if positions.shape != (steps, 2):
                    raise ValueError("Obstacle prediction does not match physical horizon")
                centers[i, j] = torch.as_tensor(positions, dtype=dtype, device=device)
                radii[i, j] = obstacle.radius + footprint_radius
                present[i, j] = True
            if row.context.obstacles:
                # Weight by expert trajectory proximity, not noisy predicted paths.
                target = torch.as_tensor(row.target, dtype=dtype, device=device)
                distance = torch.linalg.vector_norm(
                    target[None] - centers[i, :len(row.context.obstacles)], dim=-1
                ) - radii[i, :len(row.context.obstacles), None]
                excess = (distance.amin(dim=0) - clearance_margin_m).clamp_min(0)
                focus[i] = torch.exp(-0.5 * (excess / focus_sigma_m).square())
    return LossGeometry(decoded, centers, radii, present, focus)


def focused_losses(
    model: TrajectoryFlowModel,
    x1: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    geometry: LossGeometry,
    conditioned: torch.Tensor,
    *,
    focus_weight: float,
    clearance_weight: float,
    clearance_margin_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (objective, FM MSE, local path MSE [m²], swept hinge [m²])."""
    fm = (prediction[:, 4:] - target_velocity[:, 4:]).square().mean()
    zero = fm.new_zeros(())
    if (focus_weight == 0 and clearance_weight == 0) or geometry.centers.shape[1] == 0:
        return fm, fm, zero, zero

    # x1_hat - x1 contains only residual spline coordinates; fixed endpoints cancel.
    delta = ((xt[:, 4:] + (1 - t) * prediction[:, 4:] - x1[:, 4:])
             .reshape(-1, model.codec.knots, 2) * model.target_scale)
    basis = torch.as_tensor(model.codec.basis, dtype=delta.dtype, device=delta.device)
    path_delta = torch.einsum("hk,bkc->bhc", basis, delta)
    estimated = geometry.decoded_targets + path_delta

    local_weights = geometry.focus * conditioned[:, None]
    local_error = path_delta.square().sum(dim=-1)
    focus = (local_weights * local_error).sum() / local_weights.sum().clamp_min(1)

    # Relative linear motion across each physical interval; closest approach on
    # the swept segment, rather than only distance at sampled waypoints.
    relative = estimated[:, None] - geometry.centers
    start, step = relative[:, :, :-1], relative[:, :, 1:] - relative[:, :, :-1]
    fraction = (-torch.sum(start * step, dim=-1) /
                torch.sum(step.square(), dim=-1).clamp_min(1e-12)).clamp(0, 1)
    closest = start + fraction[..., None] * step
    clearance = torch.sqrt(closest.square().sum(dim=-1) + 1e-12) - geometry.radii[..., None]
    violation = (clearance_margin_m - clearance).clamp_min(0).square()
    valid = geometry.present[:, :, None] & conditioned[:, None, None]
    # Normalize across active obstacle/segment pairs, not padded obstacles.
    denominator = (valid.sum() * (estimated.shape[1] - 1)).clamp_min(1)
    barrier = (violation * valid).sum() / denominator
    return fm + focus_weight * focus + clearance_weight * barrier, fm, focus, barrier
