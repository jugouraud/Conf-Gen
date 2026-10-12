"""Create and validate the frozen safe trajectory probe bank."""

import hashlib

import torch

from .common import resolved_path, sha256, write_json
from .corpus import read_rows
from .model import SanaAdapter

TRAJECTORY_SEED_STRIDE = 100_003


# Hierarchy 3: atomic actions.

def bank_file(cfg):
    return resolved_path(cfg, "output") / "probes" / "probe_bank.pt"


def _require_new_bank(path):
    if path.exists():
        raise FileExistsError(f"Frozen probe bank exists: {path}")


def _load_adapter(cfg):
    return SanaAdapter(cfg)


def _select_times(cfg, adapter):
    options = cfg["probes"]
    n_times, per_time = int(options["time_points"]), int(options["states_per_time"])
    if not 2 <= n_times <= adapter.steps or per_time < 1:
        raise ValueError("Invalid time/probe counts")
    indices = [round(k * (adapter.steps - 1) / (n_times - 1)) for k in range(n_times)]
    if len(set(indices)) != n_times:
        raise ValueError("Duplicate selected timestep indices")
    return indices, per_time


def _choose_probes(cfg, pool, indices, per_time):
    chosen = []
    for step in indices:
        candidates = pool[step]
        if len(candidates) < per_time:
            raise ValueError("Insufficient safe source trajectories for requested probe count")
        ordering = sorted(range(len(candidates)), key=lambda k: hashlib.sha256(
            f"{cfg['experiment']['seed']}:{step}:{k}".encode()).hexdigest())
        chosen.extend(candidates[k] for k in ordering[:per_time])
    return chosen


def _build_bank(cfg, adapter, chosen):
    states, times, provenance = zip(*chosen)
    return {
        "states": torch.stack(states).float(),
        "times": torch.tensor(times, dtype=torch.float32),
        "weights": torch.full((len(states),), 1 / len(states)),
        "provenance": list(provenance), "metadata": adapter.metadata,
        "adapter_signature": adapter.signature,
        "split_sha256": sha256(resolved_path(cfg, "output") / "splits" / "safe_probe_source.jsonl"),
    }


def _save_bank(path, bank):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)


# Hierarchy 2: subworkflows.

def _collect_trajectories(cfg, adapter, indices):
    pool = {step: [] for step in indices}
    options = cfg["probes"]
    for position, row in enumerate(read_rows(cfg, "safe_probe_source")):
        pair = adapter.encode(row["prompt"])
        for repeat in range(int(options["trajectories_per_prompt"])):
            seed = int(cfg["experiment"]["seed"]) + TRAJECTORY_SEED_STRIDE * position + repeat
            generator = torch.Generator(device="cpu").manual_seed(seed)
            x = torch.randn((1, *adapter.shape), generator=generator).to(adapter.device)
            adapter.pipe.scheduler.set_timesteps(adapter.steps, device=adapter.device)
            for step, time in enumerate(adapter.pipe.scheduler.timesteps):
                if step in pool:
                    pool[step].append((x[0].detach().cpu().clone(), float(time),
                                       {"prompt_id": row["id"], "seed": seed,
                                        "scheduler_step": step}))
                predicted = adapter.velocity(x, time, pair)
                x = adapter.pipe.scheduler.step(predicted, time, x, return_dict=False)[0]
    return pool


def _write_bank_manifest(path, bank):
    write_json(path.parent / "probe_bank_manifest.json", {
        "bank_sha256": sha256(path), "metadata": bank["metadata"],
        "probe_count": len(bank["states"]), "provenance": bank["provenance"],
        "split_sha256": bank["split_sha256"],
    })
    print(f"Frozen {len(bank['states'])} safe-only probes: {path}")


def read_bank(cfg, adapter=None):
    bank = torch.load(bank_file(cfg), map_location="cpu", weights_only=True)
    if adapter is not None and bank["adapter_signature"] != adapter.signature:
        raise ValueError("Model revision, scheduler, guidance or precision changed")
    source = resolved_path(cfg, "output") / "splits" / "safe_probe_source.jsonl"
    if bank["split_sha256"] != sha256(source):
        raise ValueError("Safe probe source split changed after probe construction")
    if len(bank["states"]) != len(bank["times"]) or len(bank["times"]) != len(bank["weights"]):
        raise ValueError("Invalid frozen bank dimensions")
    if not torch.isclose(bank["weights"].sum(), torch.tensor(1.0), atol=1e-6):
        raise ValueError("Invalid probe weights")
    return bank


# Hierarchy 1: workflow orchestration.

def freeze_probes(cfg: dict) -> None:
    """Freeze a weighted probe bank from disjoint safe trajectories."""
    path = bank_file(cfg)
    _require_new_bank(path)
    adapter = _load_adapter(cfg)
    indices, per_time = _select_times(cfg, adapter)
    pool = _collect_trajectories(cfg, adapter, indices)
    chosen = _choose_probes(cfg, pool, indices, per_time)
    bank = _build_bank(cfg, adapter, chosen)
    _save_bank(path, bank)
    _write_bank_manifest(path, bank)
