"""Build shared probes only from frozen reference episodes."""

from collections.abc import Mapping, Sequence

import numpy as np

from backend.src.control.contracts import Example, ProbeBank
from backend.src.control.model import TrajectoryModel
from backend.src.control.partition import validate_partitions


def run_probe_construction(
    model: TrajectoryModel,
    groups: Mapping[str, Sequence[Example]],
    *,
    count: int,
    steps: int,
    seed: int,
) -> ProbeBank:
    """Own (x_j,t_j), using reference ODE trajectories and a fixed local RNG."""
    validate_partitions(groups)
    model.assert_frozen()
    if type(count) is not int or not 1 <= count <= 4096 or not 1 <= steps <= 10000:
        raise ValueError("Invalid bounded probe construction budget")
    rng = np.random.default_rng(seed)
    states, times, episodes = [], [], []
    for index in range(count):
        row = groups["reference"][index % len(groups["reference"])]
        context = model.encode(row.context)
        x = np.asarray(model.initialize(context, rng, None), dtype=np.float64)
        selected = int(rng.integers(0, steps))
        for j in range(selected):
            velocity = np.asarray(
                model.drift(x.copy(), j / steps, context, row.guidance)
            )
            if velocity.shape != x.shape:
                raise ValueError("Probe field shape mismatch")
            x = x + velocity / steps
        states.append(x)
        times.append(selected / steps)
        episodes.append(row.context.episode)
    model.assert_frozen()
    return ProbeBank(np.stack(states), np.array(times), tuple(episodes), model.revision)
