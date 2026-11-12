"""Observation boundary and start of the common physical-step deadline."""

import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from backend.src.control.contracts import Context, ControlState
from backend.src.control.model import TrajectoryModel


def run_context_encoding(
    model: TrajectoryModel,
    obs: Context,
    budget: float,
    clock: Callable[[], float] = time.perf_counter,
) -> ControlState:
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("Positive finite control budget required")
    started = clock()
    state = ControlState(None, started, started + budget, clock)
    try:
        model.assert_frozen()
        context = model.encode(obs)
        if (
            not isinstance(context, Context)
            or replace(context, features=obs.features) != obs
        ):
            raise ValueError("Encoding must preserve the physical scene")
        # Publish only a validated scene; fallback must never see rejected
        # encoder output (in particular altered obstacle predictions).
        state.context = context
        state.check_deadline()
    except TimeoutError:
        state.failure = "timeout"
    except (ValueError, RuntimeError, ArithmeticError):
        state.failure = "encoding_failure"
    return state
