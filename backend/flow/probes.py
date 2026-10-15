"""Freeze midpoint states from guided trajectories of the safe fit set."""

import random
from fractions import Fraction

import torch

from .common import resolved_path, sha256, write_json
from .corpus import limited_rows
from .model import FlowAdapter, model_signature


def bank_file(cfg):
    return resolved_path(cfg, "output") / "probes" / "probe_bank.pt"


def _time_grid(time_points: int, integration_steps: int):
    """Euler nodes include every exact quadrature midpoint and the final time."""
    if time_points < 1 or integration_steps < 1:
        raise ValueError("time_points and integration_steps must be positive")
    midpoints = [Fraction(2 * k + 1, 2 * time_points) for k in range(time_points)]
    nodes = sorted(set(Fraction(i, integration_steps)
                       for i in range(integration_steps + 1)) | set(midpoints))
    return nodes, midpoints


def _fit_rows(cfg):
    return limited_rows(cfg, "safe_reference")


def _trajectory(cfg, adapter, row, seed, nodes, midpoints):
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


def _collect(cfg, adapter, fit_rows):
    options = cfg["probes"]
    count = int(options["trajectories"])
    if count < 1:
        raise ValueError("trajectories must be positive")
    nodes, midpoints = _time_grid(int(options["time_points"]),
                                  int(options["integration_steps"]))
    rng = random.Random(int(cfg["experiment"]["seed"]))
    states_by_time = [[] for _ in midpoints]
    provenance_by_time = [[] for _ in midpoints]
    for trajectory_id in range(count):
        row = fit_rows[rng.randrange(len(fit_rows))]
        seed = rng.getrandbits(63)
        states = _trajectory(cfg, adapter, row, seed, nodes, midpoints)
        for time_id, state in enumerate(states):
            states_by_time[time_id].append(state)
            provenance_by_time[time_id].append({
                "time_index": time_id, "trajectory_id": trajectory_id,
                "prompt_id": row["id"], "seed": seed,
            })
    provenance = [record for group in provenance_by_time for record in group]
    return states_by_time, provenance, [float(time) for time in midpoints]


def _build_bank(cfg, adapter, fit_rows, states_by_time, provenance, midpoints):
    count = len(states_by_time[0])
    states = torch.stack([state for group in states_by_time for state in group])
    times = torch.tensor([time for time in midpoints for _ in range(count)], dtype=torch.float32)
    return {
        "states": states, "times": times,
        "weights": torch.full((len(states),), 1 / len(states)),
        "midpoints": torch.tensor(midpoints, dtype=torch.float32),
        "time_points": len(midpoints), "trajectories": count,
        "provenance": provenance, "metadata": adapter.metadata,
        "adapter_signature": adapter.signature,
        "fit_sha256": sha256(resolved_path(cfg, "output") / "splits" / "safe_reference.jsonl"),
        "fit_ids": [row["id"] for row in fit_rows],
        "integration_steps": int(cfg["probes"]["integration_steps"]),
    }


def read_bank(cfg, adapter=None):
    bank = torch.load(bank_file(cfg), map_location="cpu", weights_only=True)
    # Older banks contain a signature over Diffusers' unordered defaults list.
    # Compare the stored metadata semantically so those frozen states remain usable.
    if adapter is not None and model_signature(bank["metadata"]) != adapter.signature:
        raise ValueError("Model, null condition, guidance or precision changed")
    fit_path = resolved_path(cfg, "output") / "splits" / "safe_reference.jsonl"
    if (bank["fit_sha256"] != sha256(fit_path)
            or bank["fit_ids"] != [row["id"] for row in _fit_rows(cfg)]):
        raise ValueError("Safe fit set changed after probe construction")
    k, m = bank["time_points"], bank["trajectories"]
    options = cfg["probes"]
    if (k != int(options["time_points"]) or m != int(options["trajectories"])
            or bank["integration_steps"] != int(options["integration_steps"])):
        raise ValueError("Probe settings changed after bank construction")
    if (len(bank["states"]) != k * m or len(bank["times"]) != k * m
            or len(bank["weights"]) != k * m or len(bank["midpoints"]) != k
            or len(bank["provenance"]) != k * m):
        raise ValueError("Invalid frozen bank dimensions")
    if not torch.isclose(bank["weights"].sum(), torch.tensor(1.0), atol=1e-6):
        raise ValueError("Invalid probe weights")
    if not torch.allclose(bank["times"].reshape(k, m), bank["midpoints"][:, None].expand(k, m)):
        raise ValueError("Probe times are not the common midpoints")
    for time_id in range(k):
        group = bank["provenance"][time_id*m:(time_id+1)*m]
        if [record["trajectory_id"] for record in group] != list(range(m)):
            raise ValueError("Probe trajectories are not shared across times")
        if time_id and any(
            (record["prompt_id"], record["seed"]) !=
            (bank["provenance"][trajectory_id]["prompt_id"],
             bank["provenance"][trajectory_id]["seed"])
            for trajectory_id, record in enumerate(group)
        ):
            raise ValueError("Probe prompt-seed pairs changed across times")
    return bank


def freeze_probes(cfg: dict) -> None:
    """Freeze K x M probes from independent prompt and Gaussian seed pairs."""
    path = bank_file(cfg)
    if path.exists():
        raise FileExistsError(f"Frozen probe bank exists: {path}")
    fit_rows = _fit_rows(cfg)
    adapter = FlowAdapter(cfg)
    states, provenance, midpoints = _collect(cfg, adapter, fit_rows)
    bank = _build_bank(cfg, adapter, fit_rows, states, provenance, midpoints)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)
    write_json(path.parent / "probe_bank_manifest.json", {
        "bank_sha256": sha256(path), "metadata": bank["metadata"],
        "probe_count": len(bank["states"]), "time_points": bank["time_points"],
        "trajectories": bank["trajectories"], "midpoints": midpoints,
        "provenance": provenance, "fit_sha256": bank["fit_sha256"],
        "fit_ids": bank["fit_ids"], "integration_steps": bank["integration_steps"],
    })
    print(f"Frozen {len(bank['states'])} safe-fit probes: {path}")
