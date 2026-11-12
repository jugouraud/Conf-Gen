"""Validate decoded references before attempting tracking/release."""

import numpy as np

from backend.src.control.contracts import ControlState
from backend.src.mpc.dynamics import TrackingConfig
from backend.src.mpc.geometry import minimum_clearance


def run_reference_validation(
    state: ControlState, config: TrackingConfig
) -> ControlState:
    if state.failure:
        return state
    try:
        state.check_deadline()
        c, reference = state.context, state.reference
        if c.dt != config.dynamics.dt or len(c.state) != len(config.dynamics.a):
            raise ValueError("Scene and dynamics disagree")
        if reference.shape != (c.horizon + 1, 2) or not np.isfinite(reference).all():
            raise ValueError("Invalid decoded reference")
        position = config.dynamics.c @ np.asarray(c.state)
        if (
            np.linalg.norm(reference[0] - position)
            > config.tracking_uncertainty + config.tolerance
        ):
            state.failure = "reference_start_mismatch"
            return state
        # Validate the actual initial connection, including reused references.
        reference = reference.copy()
        reference[0] = position
        state.reference = reference
        state.reference_clearance = minimum_clearance(
            reference, c.obstacles, config.footprint_radius
        )
        if (
            state.reference_clearance
            < config.required_clearance + config.tracking_uncertainty
        ):
            state.failure = "reference_collision_margin"
        state.check_deadline()
    except TimeoutError:
        state.failure = "timeout"
    except (ValueError, RuntimeError, ArithmeticError):
        state.failure = "reference_validation_failure"
    return state
