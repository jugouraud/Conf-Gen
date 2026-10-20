"""Finite-sample conformal cutoff calculation shared by experiment variants."""

import math


def calculate_threshold(cfg, rows, probe_hash, reference_ids, *, score_kind=None):
    alpha = float(cfg["experiment"]["alpha"])
    if not 0 < alpha < 1:
        raise ValueError("0 < alpha < 1 required")
    observed = sorted(row["distance"] for row in rows)
    rank = math.ceil((len(observed) + 1) * (1 - alpha)) - 1
    cutoff = observed[rank] if rank < len(observed) else None
    result = {
        "alpha": alpha, "threshold": cutoff,
        "n_calibration": len(observed), "bank_sha256": probe_hash,
        "reference_ids": reference_ids,
        "meaning": "novelty relative to safe field history, not harmfulness",
    }
    if score_kind is not None:
        result["score_kind"] = score_kind
    return result
