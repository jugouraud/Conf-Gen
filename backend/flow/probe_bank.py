"""Shared frozen probe-bank representation and provenance validation."""

import torch

from .common import resolved_path, sha256
from .model import model_signature
from .prompt_records import limited_rows


def bank_file(cfg):
    return resolved_path(cfg, "output") / "probes" / "probe_bank.pt"


def build_bank(cfg, adapter, fit_rows, states_by_time, provenance, midpoints):
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
            or bank["fit_ids"] != [row["id"] for row in limited_rows(cfg, "safe_reference")]):
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
