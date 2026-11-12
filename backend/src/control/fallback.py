"""Reuse only a temporally aligned old reference, subject to fresh validation."""

from backend.src.control.contracts import ControlState, PreviousReference


def run_reference_fallback(
    state: ControlState, previous: PreviousReference | None
) -> ControlState:
    if not state.failure or previous is None or state.context is None:
        return state
    state.fallback_reason = state.failure
    try:
        state.check_deadline()
        state.reference = previous.shifted(state.context)
        state.reference_clearance = None
        state.mpc = None
        state.failure = None
    except TimeoutError:
        state.failure = "timeout"
    except ValueError:
        state.failure = "fallback_incompatible"
    return state
