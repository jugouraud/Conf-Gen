"""Only a validated, feasible, on-time result exposes a control input."""

from backend.src.control.contracts import ControlState


def run_control_release(state: ControlState) -> ControlState:
    try:
        state.check_deadline()
    except TimeoutError:
        state.failure = "timeout"
    if (
        not state.failure
        and state.mpc is not None
        and state.reference_clearance is not None
    ):
        state.control = state.mpc.controls[0].copy()
        state.released = True
    else:
        state.control = None
        state.released = False
        state.failure = state.failure or "incomplete_pipeline"
    state.latency = state.clock() - state.started
    # The final clock read itself may cross the budget boundary.
    if state.latency >= state.deadline - state.started:
        state.control, state.released, state.failure = None, False, "timeout"
    return state
