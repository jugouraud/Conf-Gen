"""Repair schedules, field geometry, constraint search, and sampling."""

import numpy as np

from backend.src.detection.calibration import aggregate_anchor_distances


def calculate_segment_anchor_distances(query, target, query_target, kappa):
    """Exact Hilbert-space segment identity; inputs are distances to anchors."""
    if not 0 <= kappa <= 1:
        raise ValueError("Expected 0 <= kappa <= 1")
    q, t = np.asarray(query, dtype=np.float64), np.asarray(target, dtype=np.float64)
    squared = (1 - kappa) * q**2 + kappa * t**2 - kappa * (1 - kappa) * query_target**2
    return np.sqrt(np.maximum(squared, 0))


def search_signature_steering_segment(
    query_safe,
    target_safe,
    query_unsafe,
    target_unsafe,
    query_target,
    safe_spec,
    unsafe_spec,
    safe_cutoff,
    unsafe_cutoff,
    settings,
):
    """Find first grid-detected feasible interval, refine, and verify its actual scores.

    Nearest-anchor and query-adaptive scores need not be convex. No convexity
    or global minimality is assumed; both endpoints and the full grid are checked.
    """

    def calculate_segment_scores(kappa):
        safe = aggregate_anchor_distances(
            calculate_segment_anchor_distances(
                query_safe, target_safe, query_target, kappa
            ),
            safe_spec,
        )
        unsafe = aggregate_anchor_distances(
            calculate_segment_anchor_distances(
                query_unsafe, target_unsafe, query_target, kappa
            ),
            unsafe_spec,
        )
        return float(safe), float(unsafe)

    def satisfies_segment_constraints(kappa):
        safe, unsafe = calculate_segment_scores(kappa)
        return safe <= safe_cutoff and (unsafe_cutoff is None or unsafe > unsafe_cutoff)

    if satisfies_segment_constraints(0.0):
        return {
            "kappa": 0.0,
            "safe_score": calculate_segment_scores(0.0)[0],
            "unsafe_score": calculate_segment_scores(0.0)[1],
            "meets_stop": True,
        }
    grid = np.linspace(0.0, 1.0, settings["grid_points"])
    squared_safe = (
        (1 - grid[:, None]) * np.asarray(query_safe) ** 2
        + grid[:, None] * np.asarray(target_safe) ** 2
        - (grid * (1 - grid))[:, None] * query_target**2
    )
    squared_unsafe = (
        (1 - grid[:, None]) * np.asarray(query_unsafe) ** 2
        + grid[:, None] * np.asarray(target_unsafe) ** 2
        - (grid * (1 - grid))[:, None] * query_target**2
    )
    s = aggregate_anchor_distances(np.sqrt(np.maximum(squared_safe, 0.0)), safe_spec)
    u = aggregate_anchor_distances(
        np.sqrt(np.maximum(squared_unsafe, 0.0)), unsafe_spec
    )
    indexes = np.flatnonzero(
        (s <= safe_cutoff) & (True if unsafe_cutoff is None else u > unsafe_cutoff)
    )
    if not len(indexes):
        return None
    index = int(indexes[0])
    left, right = float(grid[max(0, index - 1)]), float(grid[index])
    for _ in range(settings["bisection_steps"]):
        midpoint = (left + right) / 2
        if satisfies_segment_constraints(midpoint):
            right = midpoint
        else:
            left = midpoint
    safe, unsafe = calculate_segment_scores(right)
    if not satisfies_segment_constraints(right):
        return None
    return {
        "kappa": right,
        "safe_score": safe,
        "unsafe_score": unsafe,
        "meets_stop": True,
    }


def steer_query_signatures(
    safe,
    unsafe,
    groups,
    safe_spec,
    unsafe_spec,
    safe_cutoff,
    unsafe_cutoff,
    settings,
    query_splits=("safe_test", "unsafe_test"),
):
    eligible = (
        aggregate_anchor_distances(safe["safe_reference"], safe_spec) <= safe_cutoff
    )
    if unsafe_cutoff is not None:
        eligible &= (
            aggregate_anchor_distances(unsafe["safe_reference"], unsafe_spec)
            > unsafe_cutoff
        )
    candidates = np.flatnonzero(eligible)
    records = []
    for split in query_splits:
        for i, row in enumerate(groups[split]):
            qs, qu = safe[split][i], unsafe[split][i]
            ss, us = (
                float(aggregate_anchor_distances(qs, safe_spec)),
                float(aggregate_anchor_distances(qu, unsafe_spec)),
            )
            admitted = ss <= safe_cutoff and (
                unsafe_cutoff is None or us > unsafe_cutoff
            )
            record = {
                "id": row["id"],
                "split": split,
                "category": row.get("category", "safe"),
                "admitted_unchanged": admitted,
                "target_id": None,
                "kappa": 0.0 if admitted else None,
                "initial_safe_score": ss,
                "initial_unsafe_score": us,
                "safe_score": ss,
                "unsafe_score": us,
                "meets_stop": admitted,
            }
            if not admitted and len(candidates):
                target = int(candidates[np.argmin(qs[candidates])])
                record["target_id"] = groups["safe_reference"][target]["id"]
                repaired = search_signature_steering_segment(
                    qs,
                    safe["safe_reference"][target],
                    qu,
                    unsafe["safe_reference"][target],
                    float(qs[target]),
                    safe_spec,
                    unsafe_spec,
                    safe_cutoff,
                    unsafe_cutoff,
                    settings,
                )
                if repaired is not None:
                    record.update(repaired)
            records.append(record)
    summaries = {}
    for split in query_splits:
        rows = [r for r in records if r["split"] == split]
        requested = [r for r in rows if not r["admitted_unchanged"]]
        solved = [r for r in requested if r["meets_stop"]]
        summaries[split] = {
            "n": len(rows),
            "admitted_unchanged": len(rows) - len(requested),
            "repair_requested": len(requested),
            "verified_repairs": len(solved),
            "unresolved": len(requested) - len(solved),
            "mean_kappa_repaired": float(np.mean([r["kappa"] for r in solved]))
            if solved
            else None,
        }
    return records, summaries
