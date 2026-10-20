"""Shared feature distances and scored-row serialization."""

import csv

import torch

from .anchor_scoring import (ANCHOR_BATCH_SIZE, anchor_score, fit_anchor_scoring,
                             EUCLIDEAN_SCORE_KIND, mean_anchor_distance, require_current_scoring)
from .common import resolved_path, sha256, write_json
from .prompt_records import limited_rows
from .feature_store import FEATURE_KIND, feature_file
from .probe_bank import bank_file, read_bank

NEAREST_BATCH_SIZE = ANCHOR_BATCH_SIZE


def flat_feature(cfg, split, row, bank, probe_hash):
    path = feature_file(cfg, split, row["id"])
    feature = torch.load(path, map_location="cpu", weights_only=True)
    if (feature["bank_sha256"] != probe_hash or
            feature["kind"] != FEATURE_KIND or
            feature["values"].shape != bank["states"].shape):
        raise ValueError(f"Incompatible flow signature: {path}")
    return (feature["values"].float() * bank["weights"].sqrt()[:, None, None, None]).flatten()


def nearest(query, references):
    """Original nearest-reference Euclidean score, also used by the frozen JP baseline."""
    return anchor_score(query, references, {"score_kind": EUCLIDEAN_SCORE_KIND})


def write_scores(cfg, split, rows, threshold):
    require_current_scoring(threshold, cfg)
    cutoff = float("inf") if threshold["threshold"] is None else threshold["threshold"]
    for row in rows:
        row["outside_safe_history"] = row["distance"] > cutoff
    folder = resolved_path(cfg, "output") / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(folder / f"{split}.provenance.json", {
        "score_kind": threshold["score_kind"], "split": split,
        "bank_sha256": threshold["bank_sha256"],
        "reference_ids": threshold["reference_ids"],
        "threshold_sha256": sha256(folder / "threshold.json"),
        "scores_sha256": sha256(folder / f"{split}.csv"),
    })
    print(f"{split}: {len(rows)} scores; threshold {cutoff:.5f}")


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


def score_rows(cfg, split, bank, probe_hash, reference, scoring=None):
    scoring = scoring if scoring is not None else fit_anchor_scoring(cfg, reference)
    return [
        {"id": row["id"], "label": row["label"], "source": row["source"],
         "split": split, "distance": anchor_score(
             flat_feature(cfg, split, row, bank, probe_hash), reference, scoring)}
        for row in limited_rows(cfg, split)
    ]
