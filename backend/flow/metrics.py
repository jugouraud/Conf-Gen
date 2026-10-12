"""Shared feature distances and scored-row serialization."""

import csv

import torch

from .common import resolved_path, sha256
from .corpus import limited_rows
from .features import feature_file
from .probes import bank_file, read_bank

NEAREST_BATCH_SIZE = 16


# Hierarchy 3: atomic actions.

def flat_feature(cfg, split, row, bank, probe_hash):
    path = feature_file(cfg, split, row["id"])
    feature = torch.load(path, map_location="cpu", weights_only=True)
    if (feature["bank_sha256"] != probe_hash or
            feature["subtract_empty"] != bool(cfg["experiment"]["subtract_empty"]) or
            feature["values"].shape != bank["states"].shape):
        raise ValueError(f"Incompatible flow signature: {path}")
    return (feature["values"].float() * bank["weights"].sqrt()[:, None, None, None]).flatten()


def nearest(query, references):
    return min(float(torch.linalg.vector_norm(chunk - query, dim=1).min())
               for chunk in references.split(NEAREST_BATCH_SIZE))


def write_scores(cfg, split, rows, threshold):
    for row in rows:
        row["outside_safe_history"] = row["distance"] > threshold["threshold"]
    folder = resolved_path(cfg, "output") / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{split}: {len(rows)} scores; threshold {threshold['threshold']:.5f}")


# Hierarchy 2: subworkflows.

def reference_history(cfg):
    bank = read_bank(cfg)
    probe_hash = sha256(bank_file(cfg))
    rows = limited_rows(cfg, "safe_reference")
    ids = [row["id"] for row in rows]
    vectors = torch.stack([
        flat_feature(cfg, "safe_reference", row, bank, probe_hash)
        for row in rows
    ])
    return bank, probe_hash, ids, vectors


def score_rows(cfg, split, bank, probe_hash, reference):
    return [
        {"id": row["id"], "label": row["label"], "source": row["source"],
         "split": split, "distance": nearest(
             flat_feature(cfg, split, row, bank, probe_hash), reference)}
        for row in limited_rows(cfg, split)
    ]
