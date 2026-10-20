"""Numerical sampling of safe trajectories and paired source noises."""

import random
from fractions import Fraction

import torch


def time_grid(time_points: int, integration_steps: int):
    """Euler nodes include every exact quadrature midpoint and the final time."""
    if time_points < 1 or integration_steps < 1:
        raise ValueError("time_points and integration_steps must be positive")
    midpoints = [Fraction(2 * k + 1, 2 * time_points) for k in range(time_points)]
    nodes = sorted(set(Fraction(i, integration_steps)
                       for i in range(integration_steps + 1)) | set(midpoints))
    return nodes, midpoints


def _integrate_trajectory(adapter, row, seed, nodes, midpoints):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((1, *adapter.shape), generator=generator).to(adapter.device)
    embeddings = adapter.encode(row["prompt"])
    selected = {}
    midpoint_set = set(midpoints)
    for current, following in zip(nodes, nodes[1:]):
        if current in midpoint_set:
            selected[current] = x[0].detach().cpu().clone()
        velocity = adapter.guided_velocity(x, float(current), embeddings)
        x = x + float(following - current) * velocity
        if not torch.isfinite(x).all():
            raise ValueError(f"Non-finite flow trajectory for {row['id']}")
    return [selected[time] for time in midpoints]


def sample_trajectories(cfg, adapter, fit_rows):
    options = cfg["probes"]
    count = int(options["trajectories"])
    if count < 1:
        raise ValueError("trajectories must be positive")
    nodes, midpoints = time_grid(int(options["time_points"]),
                                  int(options["integration_steps"]))
    rng = random.Random(int(cfg["experiment"]["seed"]))
    states_by_time = [[] for _ in midpoints]
    provenance_by_time = [[] for _ in midpoints]
    for trajectory_id in range(count):
        row = fit_rows[rng.randrange(len(fit_rows))]
        seed = rng.getrandbits(63)
        states = _integrate_trajectory(adapter, row, seed, nodes, midpoints)
        for time_id, state in enumerate(states):
            states_by_time[time_id].append(state)
            provenance_by_time[time_id].append({
                "time_index": time_id, "trajectory_id": trajectory_id,
                "prompt_id": row["id"], "seed": seed,
            })
    provenance = [record for group in provenance_by_time for record in group]
    return states_by_time, provenance, [float(time) for time in midpoints]


def source_bank(current):
    """Reconstruct the exact noises used to launch the current safe-fit trajectories."""
    count = current["trajectories"]
    records = [dict(record, time_index=0) for record in current["provenance"][:count]]
    shape = current["states"].shape[1:]
    states = torch.stack([torch.randn((1, *shape), generator=torch.Generator("cpu").manual_seed(record["seed"]))[0]
                          for record in records])
    return dict(current, states=states, times=torch.zeros(count),
                midpoints=torch.zeros(1), weights=torch.full((count,), 1 / count),
                time_points=1, provenance=records, method="source_slice")

