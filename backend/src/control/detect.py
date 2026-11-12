"""Evaluate the nominal conditioning at the frozen detector probes."""

from backend.src.control.contracts import ControlState, Detector, Mode
from backend.src.control.model import TrajectoryModel
from backend.src.control.signatures import evaluate_signature, score_signature


def run_control_detection(
    state: ControlState,
    model: TrajectoryModel,
    detector: Detector,
    guidance: str,
    mode: Mode,
) -> ControlState:
    if mode not in {"nominal", "safe_only", "joint", "repaired"}:
        raise ValueError("Unknown operation mode")
    if state.failure:
        return state
    try:
        state.check_deadline()
        state.signature = evaluate_signature(
            model, detector.probes, state.context, guidance
        )
        state.decision = score_signature(state.signature, detector)
        if (mode == "safe_only" and state.decision.safe_flag) or (
            mode == "joint" and state.decision.flagged
        ):
            state.failure = "detector_abstention"
        state.check_deadline()
    except TimeoutError:
        state.failure = "timeout"
    except (ValueError, RuntimeError, ArithmeticError):
        state.failure = "detection_failure"
    return state
