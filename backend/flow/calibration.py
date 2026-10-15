"""Calibrate novelty on safe validation prompts only."""

import math

from .common import resolved_path, write_json
from .metrics import reference_history, score_rows, write_scores


# Hierarchy 3: atomic actions.

def _calculate_threshold(cfg, rows, probe_hash, reference_ids):
    alpha = float(cfg["experiment"]["alpha"])
    if not 0 < alpha < 1:
        raise ValueError("0 < alpha < 1 required")
    observed = sorted(row["distance"] for row in rows)
    rank = math.ceil((len(observed) + 1) * (1 - alpha)) - 1
    cutoff = observed[rank] if rank < len(observed) else None
    return {
        "alpha": alpha, "threshold": cutoff,
        "n_calibration": len(observed), "bank_sha256": probe_hash,
        "reference_ids": reference_ids,
        "meaning": "novelty relative to safe field history, not harmfulness",
    }


def _threshold_path(cfg):
    return resolved_path(cfg, "output") / "scores" / "threshold.json"


# Hierarchy 2: subworkflows.

def _require_new_threshold(cfg):
    if _threshold_path(cfg).exists():
        raise FileExistsError("Refusing to overwrite safe-only threshold")


def _load_reference(cfg):
    return reference_history(cfg)


def _score_validation(cfg, bank, probe_hash, reference):
    return score_rows(cfg, "safe_validation", bank, probe_hash, reference)


def _save_threshold(cfg, threshold):
    write_json(_threshold_path(cfg), threshold)


def _save_scores(cfg, rows, threshold):
    write_scores(cfg, "safe_validation", rows, threshold)


# Hierarchy 1: workflow orchestration.

def calibrate(cfg):
    """Freeze a safe-only threshold and save validation scores."""
    _require_new_threshold(cfg)
    bank, probe_hash, reference_ids, reference = _load_reference(cfg)
    rows = _score_validation(cfg, bank, probe_hash, reference)
    threshold = _calculate_threshold(cfg, rows, probe_hash, reference_ids)
    _save_threshold(cfg, threshold)
    _save_scores(cfg, rows, threshold)
