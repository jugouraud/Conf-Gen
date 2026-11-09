"""Anchor distances, frozen calibration, and detection metrics."""

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from backend.src.detection.definitions import (
    ADAPTIVE_SCORE_KIND,
    DISTANCE_METHODS,
    EUCLIDEAN_SCORE_KIND,
    METHOD_ALIASES,
    METHOD_SCORE_KINDS,
    SCORE_KIND,
)


def normalize_distance_method(method):
    method = METHOD_ALIASES.get(method, method)
    if method not in DISTANCE_METHODS:
        raise ValueError(
            f"Unknown distance method {method!r}; choose {', '.join(DISTANCE_METHODS)}"
        )
    return method


def resolve_anchor_scoring_settings(cfg):
    settings = cfg.get("scoring", {})
    legacy = settings.get("anchor_weighting")
    if legacy is not None and legacy not in ("uniform", "gaussian"):
        raise ValueError(
            "Legacy anchor_weighting must be uniform or gaussian; use scoring.method"
        )
    legacy_method = "anchor_adaptive" if legacy == "gaussian" else "anchor_static"
    method = normalize_distance_method(settings.get("method", legacy_method))
    if legacy is not None and method != legacy_method:
        raise ValueError("Conflicting scoring.method and legacy anchor_weighting")
    spec = {"method": method, "score_kind": METHOD_SCORE_KINDS[method]}
    if method != "anchor_adaptive":
        return spec
    scale = float(settings.get("bandwidth_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Expected positive finite bandwidth_scale for anchor_adaptive")
    return dict(spec, bandwidth_scale=scale)


def calculate_weighted_anchor_distance(distances, spec):
    distances = torch.as_tensor(distances, dtype=torch.float64)
    if (
        distances.ndim != 1
        or not len(distances)
        or not torch.isfinite(distances).all()
        or (distances < 0).any()
    ):
        raise ValueError("Expected finite nonnegative anchor distances")
    if spec["score_kind"] == EUCLIDEAN_SCORE_KIND:
        return float(distances.min())
    if spec["score_kind"] == SCORE_KIND:
        return float(distances.mean())
    if spec["score_kind"] != ADAPTIVE_SCORE_KIND:
        raise ValueError("Unknown anchor score")
    bandwidth = float(spec["bandwidth"])
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Expected finite positive Gaussian bandwidth")
    # Subtract the maximum log weight before exponentiation, as in JP K:c.
    logits = -0.5 * (distances / bandwidth).square()
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite Gaussian log weights")
    return float(torch.dot(torch.softmax(logits, dim=0), distances))


def calculate_conformal_cutoff(scores, alpha):
    scores = np.asarray(scores, dtype=np.float64)
    if (
        scores.ndim != 1
        or not len(scores)
        or not np.isfinite(scores).all()
        or not 0 < alpha < 1
    ):
        raise ValueError("Expected finite calibration scores and 0 < alpha < 1")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else math.inf


def cap_unsafe_cutoff(unsafe_cutoff, safe_calibration_unsafe_scores, cap_alpha):
    """JP's lower-order-statistic cap, fitted using safe calibration only.

    Ties can enlarge the closed unsafe set. No unconditional safe coverage
    guarantee is claimed for the capped intersection.
    """
    scores = np.asarray(safe_calibration_unsafe_scores, dtype=np.float64)
    rank = math.floor((len(scores) + 1) * cap_alpha)
    if (
        scores.ndim != 1
        or not len(scores)
        or not np.isfinite(scores).all()
        or not 0 < cap_alpha < 1
        or rank < 1
        or rank > len(scores)
    ):
        raise ValueError("Cap needs a valid positive lower order-statistic rank")
    return min(unsafe_cutoff, float(np.sort(scores)[rank - 1])), rank


def fit_distance_rule(method, anchor_distances, bandwidth_scale=1.0):
    spec = {"method": method}
    if method == "anchor_adaptive":
        matrix = np.asarray(anchor_distances)
        pairs = matrix[np.triu_indices(len(matrix), k=1)]
        median = float(np.median(pairs))
        if not np.isfinite(median) or median <= 0:
            raise ValueError(
                "Adaptive weighting requires a positive anchor-only median"
            )
        spec.update(
            bandwidth=bandwidth_scale * median,
            anchor_distance_median=median,
            bandwidth_scale=bandwidth_scale,
            bandwidth_fit="within_class_anchors_only",
        )
    elif method not in ("euclidean", "anchor_static"):
        raise ValueError("Unknown distance method")
    return spec


def aggregate_anchor_distances(distances, spec):
    values = np.asarray(distances, dtype=np.float64)
    if (
        values.ndim not in (1, 2)
        or not values.shape[-1]
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError("Expected finite nonnegative anchor distances")
    if spec["method"] == "euclidean":
        return values.min(axis=-1)
    if spec["method"] == "anchor_static":
        return values.mean(axis=-1)
    logits = -0.5 * (values / spec["bandwidth"]) ** 2
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (weights * values).sum(axis=-1) / weights.sum(axis=-1)


def calibrate_safe_distance_threshold(
    cfg, rows, probe_hash, reference_ids, *, score_kind=None
):
    alpha = float(cfg["experiment"]["alpha"])
    if not 0 < alpha < 1:
        raise ValueError("0 < alpha < 1 required")
    observed = sorted(row["distance"] for row in rows)
    rank = math.ceil((len(observed) + 1) * (1 - alpha)) - 1
    cutoff = observed[rank] if rank < len(observed) else None
    result = {
        "alpha": alpha,
        "threshold": cutoff,
        "n_calibration": len(observed),
        "bank_sha256": probe_hash,
        "reference_ids": reference_ids,
        "meaning": "novelty relative to safe field history, not harmfulness",
    }
    if score_kind is not None:
        result["score_kind"] = score_kind
    return result


def calculate_detection_metrics(labels, flags, scores):
    labels, flags = np.asarray(labels, bool), np.asarray(flags, bool)
    return {
        "safe_test_n": int((~labels).sum()),
        "unsafe_test_n": int(labels.sum()),
        "safe_false_positive_rate": float(flags[~labels].mean()),
        "unsafe_flagged_rate": float(flags[labels].mean()),
        "auroc_prompt_labels": float(roc_auc_score(labels, scores)),
        "auprc_prompt_labels": float(average_precision_score(labels, scores)),
    }


def calibrate_paired_detection_thresholds(
    distances, settings, *, safe_rule=None, unsafe_rule=None
):
    safe_rule = safe_rule or {"method": "euclidean"}
    unsafe_rule = unsafe_rule or {"method": "euclidean"}
    safe = calculate_conformal_cutoff(
        aggregate_anchor_distances(distances["safe__safe_validation"], safe_rule),
        settings["alpha"],
    )
    unsafe = calculate_conformal_cutoff(
        aggregate_anchor_distances(distances["unsafe__unsafe_validation"], unsafe_rule),
        settings["unsafe_alpha"],
    )
    capped, rank = cap_unsafe_cutoff(
        unsafe,
        aggregate_anchor_distances(distances["unsafe__safe_validation"], unsafe_rule),
        settings["cap_alpha"],
    )
    if not np.isfinite([safe, unsafe, capped]).all() or min(safe, capped) <= 0:
        raise ValueError("Detection needs positive finite validation cutoffs")
    return {
        "safe_cutoff": safe,
        "unsafe_conformal_cutoff": unsafe,
        "unsafe_cutoff": capped,
        "cap_rank": rank,
        "fit_policy": "validation_only",
        "alpha": settings["alpha"],
        "unsafe_alpha": settings["unsafe_alpha"],
        "cap_alpha": settings["cap_alpha"],
    }


def evaluate_paired_detection_tests(distances, groups, calibration):
    rows = [
        dict(row, split=split)
        for split in ("safe_test", "unsafe_test")
        for row in groups[split]
    ]
    safe = np.concatenate(
        [
            aggregate_anchor_distances(
                distances[f"safe__{s}"],
                calibration.get("safe_rule", {"method": "euclidean"}),
            )
            for s in ("safe_test", "unsafe_test")
        ]
    )
    unsafe = np.concatenate(
        [
            aggregate_anchor_distances(
                distances[f"unsafe__{s}"],
                calibration.get("unsafe_rule", {"method": "euclidean"}),
            )
            for s in ("safe_test", "unsafe_test")
        ]
    )
    labels = np.array([row["label"] == "unsafe" for row in rows])
    reports, decisions = {}, []
    for mode in ("safe_attraction", "safe_attraction_unsafe_avoidance"):
        flags = safe > calibration["safe_cutoff"]
        scores = safe / calibration["safe_cutoff"]
        if mode == "safe_attraction_unsafe_avoidance":
            flags |= unsafe <= calibration["unsafe_cutoff"]
            scores = np.maximum(scores, 2 - unsafe / calibration["unsafe_cutoff"])
        metrics = calculate_detection_metrics(labels, flags, scores)
        metrics.update(
            true_positive=int((labels & flags).sum()),
            false_positive=int((~labels & flags).sum()),
            true_negative=int((~labels & ~flags).sum()),
            false_negative=int((labels & ~flags).sum()),
            accuracy=float((labels == flags).mean()),
            by_category={
                category: calculate_detection_metrics(
                    labels[mask], flags[mask], scores[mask]
                )
                for category in sorted({row["category"] for row in rows})
                if (
                    mask := np.array([row["category"] == category for row in rows])
                ).any()
            },
        )
        reports[mode] = metrics
        decisions.extend(
            dict(
                row,
                mode=mode,
                safe_score=float(s),
                unsafe_score=float(u),
                flagged=bool(flag),
                ranking_score=float(score),
            )
            for row, s, u, flag, score in zip(
                rows, safe, unsafe, flags, scores, strict=True
            )
        )
    return reports, decisions
