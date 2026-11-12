"""Finite schedule search with unchanged context and joint admission criterion."""

from collections.abc import Callable, Sequence

import numpy as np

from backend.src.control.contracts import ControlState, Detector, Mode, RepairChoice
from backend.src.control.model import TrajectoryModel
from backend.src.control.signatures import (
    evaluate_signature,
    mix_signatures,
    score_signature,
    signature_distances,
)
from backend.src.repair.steering import (
    SteeringSchedule,
    build_constant_steering_schedule,
)


def choose_schedule(
    nominal: np.ndarray,
    anchor: np.ndarray,
    detector: Detector,
    schedules: Sequence[SteeringSchedule],
    check_deadline: Callable[[], None] = lambda: None,
) -> RepairChoice:
    """Evaluate every finite candidate: no monotonicity or interpolation of scores."""
    if not 1 <= len(schedules) <= 10000:
        raise ValueError("Schedule grid must contain 1..10000 candidates")
    feasible = []
    for index, schedule in enumerate(schedules):
        check_deadline()
        signature = mix_signatures(nominal, anchor, detector.probes.times, schedule)
        decision = score_signature(signature, detector)
        if not decision.flagged:
            feasible.append((schedule.integrate_strength(), index, schedule, decision))
    if not feasible:
        return RepairChoice(
            "repair_infeasible",
            build_constant_steering_schedule(),
            evaluated=len(schedules),
        )
    _, _, schedule, decision = min(feasible, key=lambda entry: entry[:2])
    return RepairChoice(
        "repaired", schedule, decision=decision, evaluated=len(schedules)
    )


def run_drift_repair(
    state: ControlState,
    model: TrajectoryModel,
    detector: Detector,
    schedules: Sequence[SteeringSchedule],
    mode: Mode,
) -> ControlState:
    if state.failure:
        return state
    zero = build_constant_steering_schedule()
    if mode != "repaired" or not state.decision.flagged:
        state.repair = RepairChoice("unsteered", zero, decision=state.decision)
        return state
    try:
        candidates = []
        for anchor in detector.safe:
            state.check_deadline()
            if anchor.compatibility == state.context.compatibility:
                # A catalogue entry is a field provider under CURRENT context.
                signature = evaluate_signature(
                    model, detector.probes, state.context, anchor.guidance
                )
                distance = float(
                    signature_distances(state.signature, signature[None])[0]
                )
                candidates.append((distance, anchor, signature))
        if not candidates:
            state.failure = "missing_anchor"
            state.repair = RepairChoice(state.failure, zero)
            return state
        _, anchor, signature = min(candidates, key=lambda item: item[0])
        choice = choose_schedule(
            state.signature, signature, detector, schedules, state.check_deadline
        )
        state.repair = RepairChoice(
            choice.status, choice.schedule, anchor, choice.decision, choice.evaluated
        )
        if choice.status != "repaired":
            state.failure = choice.status
        state.check_deadline()
    except TimeoutError:
        state.failure = "timeout"
    except (ValueError, RuntimeError, ArithmeticError):
        state.failure = "repair_failure"
    return state
