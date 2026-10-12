"""Extract per-prompt velocity fields at frozen probes."""

import json
import os

import torch

from .common import resolved_path, sha256
from .corpus import limited_rows
from .model import SanaAdapter
from .probes import bank_file, read_bank


# Hierarchy 3: atomic actions.

def feature_file(cfg, split, prompt_id):
    return resolved_path(cfg, "output") / "features" / split / f"{prompt_id}.pt"


def _load_rows(cfg, split, cli_limit):
    return limited_rows(cfg, split, cli_limit)


def _validate_split(cfg, split):
    if split not in ("safe_reference", "safe_validation", "safe_test", "unsafe_test"):
        raise ValueError("Only reference, validation and test splits may be extracted")
    threshold = resolved_path(cfg, "output") / "scores" / "threshold.json"
    if split == "unsafe_test" and not threshold.exists():
        raise ValueError("Unsafe queries are evaluation-only: calibrate on safe validation first")


def _validate_cached_feature(path, bank_hash, adapter_signature, subtract):
    old = torch.load(path, map_location="cpu", weights_only=True)
    if (old["bank_sha256"] != bank_hash
            or old["adapter_signature"] != adapter_signature
            or old["subtract_empty"] != subtract):
        raise ValueError(f"Stale cached feature: {path}")


def _save_feature(path, prompt_id, values, bank_hash, adapter_signature, subtract):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"values": values.half(), "id": prompt_id,
                "bank_sha256": bank_hash, "adapter_signature": adapter_signature,
                "subtract_empty": subtract}, temporary)
    os.replace(temporary, path)


# Hierarchy 2: subworkflows.

def _load_probe_context(cfg, split):
    adapter = SanaAdapter(cfg)
    bank = read_bank(cfg, adapter=adapter)
    bank_hash = sha256(bank_file(cfg))
    threshold = resolved_path(cfg, "output") / "scores" / "threshold.json"
    if split == "unsafe_test" and json.loads(threshold.read_text())["bank_sha256"] != bank_hash:
        raise ValueError("Unsafe extraction is gated by the CURRENT frozen threshold")
    return adapter, bank, bank_hash


def _evaluate_probe_field(cfg, adapter, bank, pair):
    chunk_size = int(cfg["experiment"].get("probe_batch_size", 4))
    if chunk_size < 1:
        raise ValueError("probe_batch_size must be positive")
    outputs = []
    # Keep the original time-major order; mismatched timesteps cannot be batched.
    for start in range(0, len(bank["times"]), chunk_size):
        end = min(start + chunk_size, len(bank["times"]))
        for index in range(start, end):
            state = bank["states"][index:index + 1].to(adapter.device)
            time = bank["times"][index:index + 1].to(adapter.device)
            outputs.append(adapter.velocity(state, time, pair)[0].cpu())
    return torch.stack(outputs)


def _empty_reference_field(cfg, adapter, bank):
    if bool(cfg["experiment"]["subtract_empty"]):
        return _evaluate_probe_field(cfg, adapter, bank, adapter.unconditional)
    return None


def _extract_row(cfg, split, row, adapter, bank, bank_hash, empty_field):
    path = feature_file(cfg, split, row["id"])
    subtract = bool(cfg["experiment"]["subtract_empty"])
    if path.exists():
        _validate_cached_feature(path, bank_hash, adapter.signature, subtract)
        return False
    values = _evaluate_probe_field(cfg, adapter, bank, adapter.encode(row["prompt"]))
    if empty_field is not None:
        values -= empty_field
    if not torch.isfinite(values).all():
        raise ValueError(f"Non-finite field for prompt {row['id']}")
    _save_feature(path, row["id"], values, bank_hash, adapter.signature, subtract)
    return True


def _extract_rows(cfg, split, rows, adapter, bank, bank_hash, empty_field):
    for counter, row in enumerate(rows, 1):
        written = _extract_row(cfg, split, row, adapter, bank, bank_hash, empty_field)
        if written and (counter % 10 == 0 or counter == len(rows)):
            print(f"{split}: extracted {counter}/{len(rows)}")


# Hierarchy 1: workflow orchestration.

def extract(cfg: dict, split: str, cli_limit: int | None = None) -> None:
    """Extract frozen-probe fields for one allowed split."""
    _validate_split(cfg, split)
    adapter, bank, bank_hash = _load_probe_context(cfg, split)
    empty_field = _empty_reference_field(cfg, adapter, bank)
    rows = _load_rows(cfg, split, cli_limit)
    _extract_rows(cfg, split, rows, adapter, bank, bank_hash, empty_field)
