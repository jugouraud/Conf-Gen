"""Compute FP64 distances on the frozen geometries after unsafe extraction."""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from backend.flow.common import ROOT, sha256, write_json
from backend.flow.distance_matrix import euclidean_matrix
from backend.flow.feature_store import feature_file
from backend.flow.metrics import flat_feature
from backend.flow.probe_bank import bank_file, read_bank
from backend.flow.t2i_prompts import SAFE_SPLITS, UNSAFE_SPLITS, read_json, read_jsonl
from backend.flow.comparison import VARIANTS, space_config


def measure_space(output, space, device="cuda", batch_size=16):
    cfg = space_config(output, space)
    root = output / space
    bank, bank_hash = read_bank(cfg), sha256(bank_file(cfg))
    groups = {s: read_jsonl(root / "splits" / f"{s}.jsonl") for s in SAFE_SPLITS + UNSAFE_SPLITS}
    manifest = read_json(output / "experiment.json")
    if any([r["id"] for r in rows] != manifest["split_ids"][s] for s, rows in groups.items()):
        raise ValueError("Frozen split IDs changed")
    safe_receipt = read_json(root / "safe_distances.provenance.json")
    unsafe_receipt = read_json(root / "unsafe_features.provenance.json")
    if (safe_receipt["bank_sha256"] != bank_hash or unsafe_receipt["bank_sha256"] != bank_hash
            or safe_receipt["distances_sha256"] != sha256(root / "safe_distances.npz")
            or unsafe_receipt["experiment_sha256"] != sha256(output / "experiment.json")):
        raise ValueError("Feature geometry or source cache changed")
    if any(safe_receipt["split_ids"][s] != manifest["split_ids"][s] for s in SAFE_SPLITS):
        raise ValueError("Reused safe distance rows differ from the frozen safe split order")
    expected = {"experiment_sha256": sha256(output / "experiment.json"), "bank_sha256": bank_hash,
                "safe_receipt_sha256": sha256(root / "safe_distances.provenance.json"),
                "unsafe_receipt_sha256": sha256(root / "unsafe_features.provenance.json"),
                "split_ids": manifest["split_ids"], "implementation": "FP64 Euclidean Gram accumulation"}
    target, receipt = root / "two_mode_distances.npz", root / "two_mode_distances.provenance.json"
    # Validate each feature against its original extraction receipt before any
    # matrix is computed. The new unsafe test never enters a fitted cutoff.
    for split, rows in groups.items():
        hashes = safe_receipt["features"] if split in SAFE_SPLITS else unsafe_receipt["features"]
        for row in rows:
            if sha256(feature_file(cfg, split, row["id"])) != hashes[f"{split}/{row['id']}.pt"]:
                raise ValueError(f"Feature checksum changed: {space}/{split}/{row['id']}")
    if target.exists() and receipt.exists():
        if read_json(receipt) != dict(expected, distances_sha256=sha256(target)):
            raise ValueError("Two-mode distance cache changed")
        return

    def load(split, row):
        return flat_feature(cfg, split, row, bank, bank_hash)

    def calculate(reference_split, query_splits):
        rows = groups[reference_split]
        first = load(reference_split, rows[0])
        refs = torch.empty((len(rows), first.numel()), dtype=torch.float64, device=device)
        refs[0] = first.to(device)
        for i, row in enumerate(rows[1:], 1):
            refs[i] = load(reference_split, row).to(device)
        norms = torch.linalg.vector_norm(refs, dim=1).square()
        tables = {}
        for split in query_splits:
            values = []
            for start in range(0, len(groups[split]), batch_size):
                batch = groups[split][start:start + batch_size]
                query = torch.stack([load(split, r) for r in batch]).to(device=device, dtype=torch.float64)
                values.append(euclidean_matrix(query, refs, reference_squared_norms=norms).cpu().numpy())
                write_json(output / "analysis_status.json", {"status": "running", "phase": "distances",
                           "space": space, "references": reference_split, "split": split,
                           "completed": start + len(batch), "total": len(groups[split])})
            tables[split] = np.concatenate(values)
            print(f"{space}: {split} -> {reference_split} complete", flush=True)
        del refs, norms, query
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return tables

    unsafe = calculate("unsafe_reference", SAFE_SPLITS + UNSAFE_SPLITS)
    np.fill_diagonal(unsafe["unsafe_reference"], 0.)
    with np.load(root / "safe_distances.npz") as cached:
        safe = {s: cached[s].copy() for s in SAFE_SPLITS}
    safe["unsafe_reference"] = unsafe["safe_reference"].T.copy()
    safe.update(calculate("safe_reference", ("unsafe_validation", "unsafe_test")))
    np.savez_compressed(target, **{f"safe__{s}": v for s, v in safe.items()},
                        **{f"unsafe__{s}": v for s, v in unsafe.items()})
    write_json(receipt, dict(expected, distances_sha256=sha256(target)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/flow_t2i_two_mode_1000")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    for space in VARIANTS:
        measure_space(args.output.resolve(), space, args.device)
    write_json(args.output / "analysis_status.json", {"status": "complete", "phase": "distances_complete"})


if __name__ == "__main__":
    main()
