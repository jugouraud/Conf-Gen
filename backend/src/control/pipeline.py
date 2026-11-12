"""Online observation -> field detection/repair -> reference -> MPC -> control."""

import time
from collections.abc import Callable, Sequence

from backend.src.control.contracts import (
    Context,
    ControlState,
    Detector,
    Mode,
    PreviousReference,
    Sampling,
)
from backend.src.control.detect import run_control_detection
from backend.src.control.encode import run_context_encoding
from backend.src.control.fallback import run_reference_fallback
from backend.src.control.model import TrajectoryModel
from backend.src.control.release import run_control_release
from backend.src.control.repair import run_drift_repair
from backend.src.control.sample import run_reference_sampling
from backend.src.control.track import run_tracking_mpc
from backend.src.control.validate import run_reference_validation
from backend.src.mpc.dynamics import TrackingConfig
from backend.src.repair.steering import SteeringSchedule


def run_control_pipeline(
    obs: Context,
    model: TrajectoryModel,
    detector: Detector,
    schedules: Sequence[SteeringSchedule],
    tracking: TrackingConfig,
    *,
    guidance: str,
    mode: Mode = "repaired",
    seed: int = 0,
    sampling: Sampling = Sampling(),
    deadline_seconds: float = 1.0,
    previous: PreviousReference | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> ControlState:
    """Compose public stage H1s; no hidden state is shared between physical steps."""
    state = run_context_encoding(model, obs, deadline_seconds, clock)
    state = run_control_detection(state, model, detector, guidance, mode)
    state = run_drift_repair(state, model, detector, schedules, mode)
    state = run_reference_sampling(state, model, guidance, sampling, seed, previous)
    state = run_reference_validation(state, tracking)
    state = run_tracking_mpc(state, tracking)
    if state.failure and previous is not None:
        state = run_reference_fallback(state, previous)
        state = run_reference_validation(state, tracking)
        state = run_tracking_mpc(state, tracking)
    return run_control_release(state)
