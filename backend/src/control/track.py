"""Connect the decoded-reference workflow to the general tracking solver."""

import numpy as np

from backend.src.control.contracts import ControlState
from backend.src.mpc.dynamics import TrackingConfig
from backend.src.mpc.solver import TrackingFailure, solve_tracking


def run_tracking_mpc(state: ControlState, config: TrackingConfig) -> ControlState:
    if state.failure:
        return state
    try:
        if state.context.dt != config.dynamics.dt:
            raise ValueError("MPC and observation physical dt differ")
        state.mpc = solve_tracking(
            np.asarray(state.context.state),
            state.reference,
            state.context.obstacles,
            config,
            check_deadline=state.check_deadline,
        )
    except TimeoutError:
        state.failure = "timeout"
    except TrackingFailure as error:
        state.failure = str(error)
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError):
        state.failure = "solver_failure"
    return state
