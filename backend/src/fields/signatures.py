"""Frozen probe banks, signature storage, and extraction progress."""

import copy
import hashlib
import json
import os
import random
import time
import zipfile
from fractions import Fraction
from pathlib import Path

import torch

from backend.src.common.storage import (
    calculate_file_sha256,
    read_jsonl,
    resolve_config_path,
    write_json,
)
from backend.src.data.prompts import read_limited_prompt_split
from backend.src.fields.definitions import FEATURE_KIND, VARIANTS


def load_signature_space_config(output, space):
    if space not in VARIANTS:
        raise ValueError(f"Unknown signature space: {space}")
    manifest = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(manifest["config"])
    cfg["paths"]["output"] = str(output / space)
    cfg["scoring"] = {"method": "anchor_static"}
    if space == "source_slice":
        cfg["probes"]["time_points"] = 1
    return cfg


def resolve_signature_path(cfg, split, prompt_id):
    return resolve_config_path(cfg, "output") / "features" / split / f"{prompt_id}.pt"


def validate_cached_signature(path, bank_hash, adapter_signature):
    old = torch.load(path, map_location="cpu", weights_only=True)
    if (
        old["bank_sha256"] != bank_hash
        or old["adapter_signature"] != adapter_signature
        or old["kind"] != FEATURE_KIND
    ):
        raise ValueError(f"Stale cached feature: {path}")


def save_field_signature(path, prompt_id, values, bank_hash, adapter_signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "values": values.float(),
            "id": prompt_id,
            "bank_sha256": bank_hash,
            "adapter_signature": adapter_signature,
            "kind": FEATURE_KIND,
        },
        temporary,
    )
    os.replace(temporary, path)


def evaluate_probe_field(cfg, adapter, bank, embeddings):
    remote_evaluator = getattr(adapter, "evaluate_frozen_probes", None)
    if remote_evaluator is not None:
        return remote_evaluator(cfg, bank, embeddings)
    chunk_size = int(cfg["experiment"].get("probe_batch_size", 1))
    if chunk_size < 1:
        raise ValueError("probe_batch_size must be positive")
    outputs = []
    count = bank["trajectories"]
    for time_id, midpoint in enumerate(bank["midpoints"]):
        for offset in range(0, count, chunk_size):
            start = time_id * count + offset
            end = min(start + chunk_size, (time_id + 1) * count)
            states = bank["states"][start:end].to(adapter.device)
            effect = adapter.evaluate_prompt_effect(states, float(midpoint), embeddings)
            outputs.extend(effect.cpu().unbind(0))
    return torch.stack(outputs)


def calculate_model_signature(metadata: dict) -> str:
    """Hash model settings without Diffusers' unordered default-value list."""
    stable = dict(metadata)
    scheduler = dict(stable["scheduler"])
    if "_use_default_values" in scheduler:
        scheduler["_use_default_values"] = sorted(scheduler["_use_default_values"])
    stable["scheduler"] = scheduler
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str).encode()
    ).hexdigest()


def resolve_probe_bank_path(cfg):
    return resolve_config_path(cfg, "output") / "probes" / "probe_bank.pt"


def read_probe_fit_rows(cfg):
    """Resolve frozen probe-fitting rows; an expanded region may keep its original bank."""
    root = resolve_config_path(cfg, "output")
    split = cfg["probes"].get("fit_split", "safe_reference")
    if split not in ("safe_reference", "safe_probe_fit"):
        raise ValueError("Unknown probe fit split")
    path = root / "splits" / f"{split}.jsonl"
    rows = (
        read_limited_prompt_split(cfg, "safe_reference")
        if split == "safe_reference"
        else read_jsonl(path)
    )
    if split == "safe_probe_fit":
        training = {
            row["id"]: row for row in read_jsonl(root / "splits/safe_reference.jsonl")
        }
        if (
            not rows
            or len({row["id"] for row in rows}) != len(rows)
            or any(training.get(row["id"]) != row for row in rows)
        ):
            raise ValueError(
                "Reused probe fit rows must be unchanged legal training records"
            )
    return path, rows


def build_probe_bank(cfg, adapter, fit_rows, states_by_time, provenance, midpoints):
    fit_path, expected_rows = read_probe_fit_rows(cfg)
    if fit_rows != expected_rows:
        raise ValueError("Probe fitting rows differ from the frozen fit split")
    count = len(states_by_time[0])
    states = torch.stack([state for group in states_by_time for state in group])
    times = torch.tensor(
        [time for time in midpoints for _ in range(count)], dtype=torch.float32
    )
    return {
        "states": states,
        "times": times,
        "weights": torch.full((len(states),), 1 / len(states)),
        "midpoints": torch.tensor(midpoints, dtype=torch.float32),
        "time_points": len(midpoints),
        "trajectories": count,
        "provenance": provenance,
        "metadata": adapter.metadata,
        "adapter_signature": adapter.signature,
        "fit_sha256": calculate_file_sha256(fit_path),
        "fit_ids": [row["id"] for row in fit_rows],
        "integration_steps": int(cfg["probes"]["integration_steps"]),
    }


def build_source_slice_bank(trajectory_bank):
    """Pair source Gaussian probes with the frozen trajectory seeds at flow time zero."""
    count = trajectory_bank["trajectories"]
    provenance = copy.deepcopy(trajectory_bank["provenance"][:count])
    if [row["trajectory_id"] for row in provenance] != list(range(count)):
        raise ValueError("Invalid paired trajectory provenance")
    states = []
    for row in provenance:
        if row["prompt_id"] not in trajectory_bank["fit_ids"]:
            raise ValueError("Probe lineage is outside the legal training fit set")
        generator = torch.Generator(device="cpu").manual_seed(row["seed"])
        states.append(
            torch.randn((1, *trajectory_bank["states"].shape[1:]), generator=generator)[
                0
            ]
        )
        row["time_index"] = 0
    bank = copy.deepcopy(
        {key: value for key, value in trajectory_bank.items() if key != "states"}
    )
    bank.update(
        states=torch.stack(states),
        times=torch.zeros(count),
        midpoints=torch.zeros(1),
        weights=torch.full((count,), 1 / count),
        time_points=1,
        provenance=provenance,
        method="source_slice",
    )
    return bank


def read_probe_bank(cfg, adapter=None):
    bank = torch.load(
        resolve_probe_bank_path(cfg), map_location="cpu", weights_only=True
    )
    # Older banks contain a signature over Diffusers' unordered defaults list.
    # Compare the stored metadata semantically so those frozen states remain usable.
    if (
        adapter is not None
        and calculate_model_signature(bank["metadata"]) != adapter.signature
    ):
        raise ValueError("Model, null condition, guidance or precision changed")
    fit_path, fit_rows = read_probe_fit_rows(cfg)
    if bank["fit_sha256"] != calculate_file_sha256(fit_path) or bank["fit_ids"] != [
        row["id"] for row in fit_rows
    ]:
        raise ValueError("Safe fit set changed after probe construction")
    k, m = bank["time_points"], bank["trajectories"]
    options = cfg["probes"]
    if (
        k != int(options["time_points"])
        or m != int(options["trajectories"])
        or bank["integration_steps"] != int(options["integration_steps"])
    ):
        raise ValueError("Probe settings changed after bank construction")
    if (
        len(bank["states"]) != k * m
        or len(bank["times"]) != k * m
        or len(bank["weights"]) != k * m
        or len(bank["midpoints"]) != k
        or len(bank["provenance"]) != k * m
    ):
        raise ValueError("Invalid frozen bank dimensions")
    if not torch.isclose(bank["weights"].sum(), torch.tensor(1.0), atol=1e-6):
        raise ValueError("Invalid probe weights")
    if not torch.allclose(
        bank["times"].reshape(k, m), bank["midpoints"][:, None].expand(k, m)
    ):
        raise ValueError("Probe times are not the common midpoints")
    for time_id in range(k):
        group = bank["provenance"][time_id * m : (time_id + 1) * m]
        if [record["trajectory_id"] for record in group] != list(range(m)):
            raise ValueError("Probe trajectories are not shared across times")
        if time_id and any(
            (record["prompt_id"], record["seed"])
            != (
                bank["provenance"][trajectory_id]["prompt_id"],
                bank["provenance"][trajectory_id]["seed"],
            )
            for trajectory_id, record in enumerate(group)
        ):
            raise ValueError("Probe prompt-seed pairs changed across times")
    return bank


def load_weighted_signature_vector(cfg, split, row, bank, probe_hash):
    path = resolve_signature_path(cfg, split, row["id"])
    feature = torch.load(path, map_location="cpu", weights_only=True)
    if (
        feature["bank_sha256"] != probe_hash
        or feature["kind"] != FEATURE_KIND
        or feature["values"].shape != bank["states"].shape
    ):
        raise ValueError(f"Incompatible flow signature: {path}")
    return (
        feature["values"].float() * bank["weights"].sqrt()[:, None, None, None]
    ).flatten()


def build_integration_time_grid(time_points: int, integration_steps: int):
    """Euler nodes include every exact quadrature midpoint and the final time."""
    if time_points < 1 or integration_steps < 1:
        raise ValueError("time_points and integration_steps must be positive")
    midpoints = [Fraction(2 * k + 1, 2 * time_points) for k in range(time_points)]
    nodes = sorted(
        set(Fraction(i, integration_steps) for i in range(integration_steps + 1))
        | set(midpoints)
    )
    return nodes, midpoints


def _integrate_trajectory(adapter, row, seed, nodes, midpoints):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((1, *adapter.shape), generator=generator).to(adapter.device)
    embeddings = adapter.encode_prompt(row["prompt"])
    selected = {}
    midpoint_set = set(midpoints)
    for current, following in zip(nodes, nodes[1:]):
        if current in midpoint_set:
            selected[current] = x[0].detach().cpu().clone()
        velocity = adapter.evaluate_guided_velocity(x, float(current), embeddings)
        x = x + float(following - current) * velocity
        if not torch.isfinite(x).all():
            raise ValueError(f"Non-finite flow trajectory for {row['id']}")
    return [selected[time] for time in midpoints]


def sample_safe_reference_trajectories(cfg, adapter, fit_rows):
    options = cfg["probes"]
    count = int(options["trajectories"])
    if count < 1:
        raise ValueError("trajectories must be positive")
    nodes, midpoints = build_integration_time_grid(
        int(options["time_points"]), int(options["integration_steps"])
    )
    rng = random.Random(int(cfg["experiment"]["seed"]))
    states_by_time = [[] for _ in midpoints]
    provenance_by_time = [[] for _ in midpoints]
    for trajectory_id in range(count):
        row = fit_rows[rng.randrange(len(fit_rows))]
        seed = rng.getrandbits(63)
        states = _integrate_trajectory(adapter, row, seed, nodes, midpoints)
        for time_id, state in enumerate(states):
            states_by_time[time_id].append(state)
            provenance_by_time[time_id].append(
                {
                    "time_index": time_id,
                    "trajectory_id": trajectory_id,
                    "prompt_id": row["id"],
                    "seed": seed,
                }
            )
    provenance = [record for group in provenance_by_time for record in group]
    return states_by_time, provenance, [float(time) for time in midpoints]


class SignatureExtractionTracker:
    """Persist extraction progress, artifact hashes, and bounded transfer bundles."""

    def __init__(self, output, *, resume=False):
        self.output = output.resolve()
        self.status_path = self.output / "status.json"
        self.state = {
            "status": "running",
            "phase": "loading_model",
            "completed_features": 0,
            "counts": {},
            "bundles": {},
            "started_unix": time.time(),
        }
        manifest_path = self.output / "experiment.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.state["total_features"] = len(VARIANTS) * sum(
                manifest["counts"].values()
            )
        previous = (
            json.loads(self.status_path.read_text(encoding="utf-8"))
            if resume and self.status_path.exists()
            else {}
        )
        self.state["bundles"] = previous.get("bundles", {})
        self.transferred = set()
        if resume:
            for name, receipt in self.state["bundles"].items():
                path = self.output / "transfer" / name
                if (
                    not path.exists()
                    or calculate_file_sha256(path) != receipt["sha256"]
                ):
                    raise ValueError("Resume transfer bundle changed or missing")
                with zipfile.ZipFile(path) as bundle:
                    self.transferred.update(
                        json.loads(bundle.read("bundle_manifest.json"))
                    )
        self.pending = []
        self.hashes = {space: {} for space in VARIANTS}
        self.tables = {space: {} for space in VARIANTS}
        self.counter = (
            max(
                (int(Path(name).stem.split("-")[-1]) for name in self.state["bundles"]),
                default=-1,
            )
            + 1
        )
        self.write_status()

    def write_status(self, **values):
        """Input: changed progress fields. Output: atomically updated status.json."""
        self.state.update(
            values, elapsed_seconds=time.time() - self.state["started_unix"]
        )
        if self.state.get("total_features"):
            self.state["feature_progress_percent"] = (
                100 * self.state["completed_features"] / self.state["total_features"]
            )
        temporary = self.status_path.with_suffix(".tmp")
        write_json(temporary, self.state)
        temporary.replace(self.status_path)

    def flush_transfer_bundle(self, force=False):
        """Input: flush policy. Output: pending artifacts bundled when size limits or force require it."""
        if not self.pending or (
            not force
            and len(self.pending) < 16
            and sum(p.stat().st_size for p in self.pending) < 32 * 1024**2
        ):
            return
        folder = self.output / "transfer"
        folder.mkdir(exist_ok=True)
        name = f"part-{self.counter:05d}.zip"
        path = folder / name
        files = {
            p.relative_to(self.output).as_posix(): calculate_file_sha256(p)
            for p in self.pending
        }
        with zipfile.ZipFile(
            path.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for p in self.pending:
                archive.write(p, p.relative_to(self.output).as_posix())
            archive.writestr("bundle_manifest.json", json.dumps(files))
        path.with_suffix(".tmp").replace(path)
        self.state["bundles"][name] = {
            "sha256": calculate_file_sha256(path),
            "files": len(files),
            "bytes": path.stat().st_size,
        }
        self.counter += 1
        self.pending.clear()
        self.write_status()

    def register_artifact(self, path, space=None, split=None, prompt_id=None):
        """Input: artifact and optional signature identity. Output: tracked hash, count, and transfer entry."""
        relative = path.relative_to(self.output).as_posix()
        if relative not in self.transferred:
            self.pending.append(path)
        if space is not None:
            relative = f"{split}/{prompt_id}.pt"
            self.hashes[space][relative] = calculate_file_sha256(path)
            self.state["completed_features"] += 1
            key = f"{space}/{split}"
            self.state["counts"][key] = self.state["counts"].get(key, 0) + 1
        self.flush_transfer_bundle()


class FrozenSignatureStore:
    """Bound memory by loading individual reference signatures on demand."""

    def __init__(self, bank, bank_hash, roots, receipts):
        self.bank = bank
        self.bank_hash = bank_hash
        self.roots = roots
        self.receipts = receipts

    def load_signature(self, split, row):
        """Input: split and prompt record. Output: signature verified against frozen provenance."""
        path = self.roots[split] / "features" / split / f"{row['id']}.pt"
        key = f"{split}/{row['id']}.pt"
        if calculate_file_sha256(path) != self.receipts[split][key]:
            raise ValueError(f"Feature checksum changed: {path}")
        record = torch.load(path, map_location="cpu", weights_only=True)
        values = record["values"]
        if (
            record["id"] != row["id"]
            or record["kind"] != FEATURE_KIND
            or record["bank_sha256"] != self.bank_hash
            or record["adapter_signature"] != self.bank["adapter_signature"]
            or values.shape != self.bank["states"].shape
            or not torch.isfinite(values).all()
        ):
            raise ValueError(f"Incompatible frozen signature: {path}")
        return values.float()
