"""Shared paths, cache validation, and atomic flow-feature storage."""

import os

import torch

from .common import resolved_path

FEATURE_KIND = "guided_prompt_velocity_difference"


def feature_file(cfg, split, prompt_id):
    return resolved_path(cfg, "output") / "features" / split / f"{prompt_id}.pt"


def validate_cached_feature(path, bank_hash, adapter_signature):
    old = torch.load(path, map_location="cpu", weights_only=True)
    if (old["bank_sha256"] != bank_hash
            or old["adapter_signature"] != adapter_signature
            or old["kind"] != FEATURE_KIND):
        raise ValueError(f"Stale cached feature: {path}")


def save_feature(path, prompt_id, values, bank_hash, adapter_signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"values": values.float(), "id": prompt_id,
                "bank_sha256": bank_hash, "adapter_signature": adapter_signature,
                "kind": FEATURE_KIND}, temporary)
    os.replace(temporary, path)
