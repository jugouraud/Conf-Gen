"""Extract the CFG prompt effect at the common frozen midpoint probes."""

import json
import os

import torch

from .common import resolved_path, sha256
from .corpus import limited_rows
from .model import FlowAdapter
from .probes import bank_file, read_bank

FEATURE_KIND = "guided_prompt_velocity_difference"


def feature_file(cfg, split, prompt_id):
    return resolved_path(cfg, "output") / "features" / split / f"{prompt_id}.pt"


def _validate_split(cfg, split):
    if split not in ("safe_reference", "safe_validation", "safe_test", "unsafe_test"):
        raise ValueError("Only fit, calibration and test splits may be extracted")
    threshold = resolved_path(cfg, "output") / "scores" / "threshold.json"
    if split == "unsafe_test" and not threshold.exists():
        raise ValueError("Risky queries are evaluation-only: calibrate on safe prompts first")


def _validate_cached_feature(path, bank_hash, adapter_signature):
    old = torch.load(path, map_location="cpu", weights_only=True)
    if (old["bank_sha256"] != bank_hash
            or old["adapter_signature"] != adapter_signature
            or old["kind"] != FEATURE_KIND):
        raise ValueError(f"Stale cached feature: {path}")


def _save_feature(path, prompt_id, values, bank_hash, adapter_signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"values": values.float(), "id": prompt_id,
                "bank_sha256": bank_hash, "adapter_signature": adapter_signature,
                "kind": FEATURE_KIND}, temporary)
    os.replace(temporary, path)


def _evaluate_probe_field(cfg, adapter, bank, embeddings):
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
            effect = adapter.prompt_effect(states, float(midpoint), embeddings)
            outputs.extend(effect.cpu().unbind(0))
    return torch.stack(outputs)


def extract(cfg: dict, split: str, cli_limit: int | None = None) -> None:
    """Extract one prompt-effect signature per query with no new image trajectory."""
    _validate_split(cfg, split)
    adapter = FlowAdapter(cfg)
    bank = read_bank(cfg, adapter=adapter)
    bank_hash = sha256(bank_file(cfg))
    threshold_path = resolved_path(cfg, "output") / "scores" / "threshold.json"
    if split == "unsafe_test":
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
        if threshold["bank_sha256"] != bank_hash:
            raise ValueError("Risky extraction requires the current frozen threshold")
    rows = limited_rows(cfg, split, cli_limit)
    for counter, row in enumerate(rows, 1):
        path = feature_file(cfg, split, row["id"])
        if path.exists():
            _validate_cached_feature(path, bank_hash, adapter.signature)
            continue
        values = _evaluate_probe_field(cfg, adapter, bank, adapter.encode(row["prompt"]))
        if not torch.isfinite(values).all():
            raise ValueError(f"Non-finite flow signature for prompt {row['id']}")
        _save_feature(path, row["id"], values, bank_hash, adapter.signature)
        if counter % 10 == 0 or counter == len(rows):
            print(f"{split}: extracted {counter}/{len(rows)}")
