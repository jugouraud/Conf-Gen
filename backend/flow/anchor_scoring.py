"""Nearest Euclidean, static-anchor, and adaptive-anchor distance scores."""

import math

import torch

ANCHOR_BATCH_SIZE = 16
SCORE_KIND = "uniform_mean_anchor_distance_v1"
ADAPTIVE_SCORE_KIND = "gaussian_weighted_anchor_distance_v1"
EUCLIDEAN_SCORE_KIND = "nearest_euclidean_anchor_distance_v1"
DISTANCE_METHODS = ("euclidean", "anchor_static", "anchor_adaptive")
METHOD_SCORE_KINDS = dict(zip(DISTANCE_METHODS,
                             (EUCLIDEAN_SCORE_KIND, SCORE_KIND, ADAPTIVE_SCORE_KIND)))
METHOD_ALIASES = {"euclidian": "euclidean", "anchor_:static": "anchor_static",
                  "anchor:static": "anchor_static", "anchor_adaptative": "anchor_adaptive"}


def normalize_method(method):
    method = METHOD_ALIASES.get(method, method)
    if method not in DISTANCE_METHODS:
        raise ValueError(f"Unknown distance method {method!r}; choose {', '.join(DISTANCE_METHODS)}")
    return method


def mean_anchor_distance(query, references):
    """Mean of anchor distances; each distance already includes probe weights."""
    if (query.ndim != 1 or references.ndim != 2 or len(references) == 0
            or references.shape[1] != query.numel()):
        raise ValueError("Expected a vector and a nonempty matching anchor matrix")
    total = sum(float(torch.linalg.vector_norm(chunk - query, dim=1).double().sum())
                for chunk in references.split(ANCHOR_BATCH_SIZE))
    score = total / len(references)
    if not math.isfinite(score):
        raise ValueError("Non-finite mean anchor distance")
    return score


def scoring_config(cfg):
    settings = cfg.get("scoring", {})
    legacy = settings.get("anchor_weighting")
    if legacy is not None and legacy not in ("uniform", "gaussian"):
        raise ValueError("Legacy anchor_weighting must be uniform or gaussian; use scoring.method")
    legacy_method = "anchor_adaptive" if legacy == "gaussian" else "anchor_static"
    method = normalize_method(settings.get("method", legacy_method))
    if legacy is not None and method != legacy_method:
        raise ValueError("Conflicting scoring.method and legacy anchor_weighting")
    spec = {"method": method, "score_kind": METHOD_SCORE_KINDS[method]}
    if method != "anchor_adaptive":
        return spec
    scale = float(settings.get("bandwidth_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Expected positive finite bandwidth_scale for anchor_adaptive")
    return dict(spec, bandwidth_scale=scale)


def anchor_distances(query, references):
    if (query.ndim != 1 or references.ndim != 2 or len(references) == 0
            or references.shape[1] != query.numel()):
        raise ValueError("Expected a vector and a nonempty matching anchor matrix")
    distances = torch.cat([torch.linalg.vector_norm(chunk - query, dim=1).double()
                           for chunk in references.split(ANCHOR_BATCH_SIZE)])
    if not torch.isfinite(distances).all():
        raise ValueError("Non-finite anchor distance")
    return distances


def fit_anchor_scoring(cfg, references):
    """Freeze bandwidth using off-diagonal safe-anchor distances only (JP K:c)."""
    spec = scoring_config(cfg)
    if spec["score_kind"] != ADAPTIVE_SCORE_KIND:
        return spec
    if len(references) < 2:
        raise ValueError("Gaussian weighting requires at least two safe anchors")
    pairs = torch.cat([anchor_distances(row, references[i + 1:])
                       for i, row in enumerate(references[:-1])])
    # quantile matches numpy.median, including averaging the middle pair.
    median = float(torch.quantile(pairs, 0.5))
    if median <= 0:
        raise ValueError("Gaussian bandwidth requires a positive median anchor distance")
    spec.update(anchor_distance_median=median, bandwidth=spec["bandwidth_scale"] * median,
                bandwidth_fit="median_upper_triangle_safe_reference_only")
    return spec


def weighted_distance(distances, spec):
    distances = torch.as_tensor(distances, dtype=torch.float64)
    if distances.ndim != 1 or not len(distances) or not torch.isfinite(distances).all() or (distances < 0).any():
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


def anchor_score(query, references, spec):
    if spec["score_kind"] == SCORE_KIND:
        return mean_anchor_distance(query, references)
    return weighted_distance(anchor_distances(query, references), spec)


def require_current_scoring(threshold, cfg=None):
    expected = scoring_config(cfg or {})
    if threshold.get("score_kind") != expected["score_kind"]:
        raise ValueError("Threshold uses an old or different distance method; recalibrate "
                         "and rescore with the configured method. Cached features can be reused.")
    spec = threshold.get("anchor_scoring")
    if spec is not None and (spec.get("score_kind") != expected["score_kind"]
                             or spec.get("method", expected["method"]) != expected["method"]):
        raise ValueError("Different or invalid frozen distance method; recalibrate and rescore")
    if expected["score_kind"] == ADAPTIVE_SCORE_KIND:
        spec = threshold.get("anchor_scoring", {})
        if (spec.get("score_kind") != ADAPTIVE_SCORE_KIND
                or spec.get("bandwidth_scale") != expected["bandwidth_scale"]
                or spec.get("bandwidth_fit") != "median_upper_triangle_safe_reference_only"
                or not math.isfinite(spec.get("bandwidth", 0)) or spec.get("bandwidth", 0) <= 0
                or spec["bandwidth"] != spec.get("anchor_distance_median", 0) * expected["bandwidth_scale"]):
            raise ValueError("Different or invalid adaptive bandwidth; recalibrate and rescore")
