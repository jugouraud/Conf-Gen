"""Method-specific set scores and numerically verified frozen-field steering."""

import numpy as np

from .two_sets import mixture_distances


def fit_distance_rule(method, anchor_distances, bandwidth_scale=1.):
    spec = {"method": method}
    if method == "anchor_adaptive":
        matrix = np.asarray(anchor_distances)
        pairs = matrix[np.triu_indices(len(matrix), k=1)]
        median = float(np.median(pairs))
        if not np.isfinite(median) or median <= 0:
            raise ValueError("Adaptive weighting requires a positive anchor-only median")
        spec.update(bandwidth=bandwidth_scale * median, anchor_distance_median=median,
                    bandwidth_scale=bandwidth_scale, bandwidth_fit="within_class_anchors_only")
    elif method not in ("euclidean", "anchor_static"):
        raise ValueError("Unknown distance method")
    return spec


def reduce_distances(distances, spec):
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim not in (1, 2) or not values.shape[-1] or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative anchor distances")
    if spec["method"] == "euclidean":
        return values.min(axis=-1)
    if spec["method"] == "anchor_static":
        return values.mean(axis=-1)
    logits = -.5 * (values / spec["bandwidth"]) ** 2
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (weights * values).sum(axis=-1) / weights.sum(axis=-1)


def steer_segment(query_safe, target_safe, query_unsafe, target_unsafe, query_target,
                  safe_spec, unsafe_spec, safe_cutoff, unsafe_cutoff, settings):
    """Find first grid-detected feasible interval, refine, and verify its actual scores.

    Nearest-anchor and query-adaptive scores need not be convex. No convexity
    or global minimality is assumed; both endpoints and the full grid are checked.
    """
    def scores(kappa):
        safe = reduce_distances(mixture_distances(query_safe, target_safe, query_target, kappa), safe_spec)
        unsafe = reduce_distances(mixture_distances(query_unsafe, target_unsafe, query_target, kappa), unsafe_spec)
        return float(safe), float(unsafe)

    def feasible(kappa):
        safe, unsafe = scores(kappa)
        return safe <= safe_cutoff and (unsafe_cutoff is None or unsafe > unsafe_cutoff)

    if feasible(0.):
        return {"kappa": 0., "safe_score": scores(0.)[0], "unsafe_score": scores(0.)[1], "meets_stop": True}
    grid = np.linspace(0., 1., settings["grid_points"])
    squared_safe = ((1-grid[:, None]) * np.asarray(query_safe)**2 + grid[:, None] * np.asarray(target_safe)**2
                    - (grid * (1-grid))[:, None] * query_target**2)
    squared_unsafe = ((1-grid[:, None]) * np.asarray(query_unsafe)**2 + grid[:, None] * np.asarray(target_unsafe)**2
                      - (grid * (1-grid))[:, None] * query_target**2)
    s = reduce_distances(np.sqrt(np.maximum(squared_safe, 0.)), safe_spec)
    u = reduce_distances(np.sqrt(np.maximum(squared_unsafe, 0.)), unsafe_spec)
    indexes = np.flatnonzero((s <= safe_cutoff) & (True if unsafe_cutoff is None else u > unsafe_cutoff))
    if not len(indexes):
        return None
    index = int(indexes[0])
    left, right = float(grid[max(0, index-1)]), float(grid[index])
    for _ in range(settings["bisection_steps"]):
        midpoint = (left + right) / 2
        if feasible(midpoint):
            right = midpoint
        else:
            left = midpoint
    safe, unsafe = scores(right)
    if not feasible(right):
        return None
    return {"kappa": right, "safe_score": safe, "unsafe_score": unsafe, "meets_stop": True}
