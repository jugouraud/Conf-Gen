"""Static safe/unsafe field sets and certified repair on frozen linear probes.

The probe signature of a velocity-field mixture is linear. This does not imply
linearity of generated latents, images, or JP's CLIP sampler-map (L2) features.
"""

import math

import numpy as np


def conformal_cutoff(scores, alpha):
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or not 0 < alpha < 1):
        raise ValueError("Expected finite calibration scores and 0 < alpha < 1")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else math.inf


def capped_cutoff(unsafe_cutoff, safe_calibration_unsafe_scores, cap_alpha):
    """JP's lower-order-statistic cap, fitted using safe calibration only.

    Ties can enlarge the closed unsafe set. No unconditional safe coverage
    guarantee is claimed for the capped intersection.
    """
    scores = np.asarray(safe_calibration_unsafe_scores, dtype=np.float64)
    rank = math.floor((len(scores) + 1) * cap_alpha)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or not 0 < cap_alpha < 1 or rank < 1 or rank > len(scores)):
        raise ValueError("Cap needs a valid positive lower order-statistic rank")
    return min(unsafe_cutoff, float(np.sort(scores)[rank - 1])), rank


def membership(safe_scores, unsafe_scores, safe_cutoff, unsafe_cutoff):
    safe = np.asarray(safe_scores) <= safe_cutoff
    unsafe = np.asarray(unsafe_scores) <= unsafe_cutoff
    return {"safe_only": safe & ~unsafe, "unsafe_only": ~safe & unsafe,
            "ambiguous": safe & unsafe, "neither": ~safe & ~unsafe}


def mixture_distances(query, target, query_target, kappa):
    """Exact Hilbert-space segment identity; inputs are distances to anchors."""
    if not 0 <= kappa <= 1:
        raise ValueError("Expected 0 <= kappa <= 1")
    q, t = np.asarray(query, dtype=np.float64), np.asarray(target, dtype=np.float64)
    squared = (1 - kappa) * q ** 2 + kappa * t ** 2 - kappa * (1 - kappa) * query_target ** 2
    return np.sqrt(np.maximum(squared, 0))


def _convex_sublevel(score, cutoff, iterations):
    """Numerical interval {k in [0,1]: score(k) <= cutoff} for convex score."""
    if math.isinf(cutoff):
        return (0., 1.)
    # Golden-section search locates the minimum even when both endpoints fail.
    left, right = 0., 1.
    ratio = (math.sqrt(5) - 1) / 2
    x, y = right - ratio * (right - left), left + ratio * (right - left)
    fx, fy = score(x), score(y)
    for _ in range(iterations):
        if fx <= fy:
            right, y, fy = y, x, fx
            x = right - ratio * (right - left)
            fx = score(x)
        else:
            left, x, fx = x, y, fy
            y = left + ratio * (right - left)
            fy = score(y)
    mid = min((0., (left + right) / 2, 1.), key=score)
    if score(mid) > cutoff:
        return None
    lo, hi = 0., mid
    if score(0.) > cutoff:
        for _ in range(iterations):
            point = (lo + hi) / 2
            if score(point) <= cutoff:
                hi = point
            else:
                lo = point
        lower = hi
    else:
        lower = 0.
    lo, hi = mid, 1.
    if score(1.) > cutoff:
        for _ in range(iterations):
            point = (lo + hi) / 2
            if score(point) <= cutoff:
                lo = point
            else:
                hi = point
        upper = lo
    else:
        upper = 1.
    return lower, upper


def repair_segment(query_safe, target_safe, query_unsafe, target_unsafe,
                   query_target, safe_cutoff, unsafe_cutoff=None, iterations=48):
    """Earliest numerically feasible point on a segment, with a verified stop.

    Subtracting the unsafe convex sublevel interval avoids assuming that the
    intersection with the unsafe set's complement is convex or monotone.
    Failure returns None, including for targets that do not meet both stops.
    """
    safe = lambda k: float(mixture_distances(query_safe, target_safe, query_target, k).mean())
    unsafe = lambda k: float(mixture_distances(query_unsafe, target_unsafe, query_target, k).mean())
    inside = _convex_sublevel(safe, safe_cutoff, iterations)
    if inside is None:
        return None
    lower, upper = inside
    if unsafe_cutoff is not None:
        forbidden = _convex_sublevel(unsafe, unsafe_cutoff, iterations)
        if forbidden is not None:
            _u0, u1 = forbidden
            # Interval roots are numerical: evaluate the actual closed-boundary
            # score when lower is within roundoff of the unsafe entry root.
            if unsafe(lower) <= unsafe_cutoff and lower <= u1 + 1e-10:
                # Move beyond the CLOSED unsafe boundary; verify in score space.
                lower = u1 + 1e-10
    if lower > upper:
        return None
    if safe(lower) > safe_cutoff:
        lower += min(1e-10, (upper - lower) / 2)
    if safe(lower) > safe_cutoff or (unsafe_cutoff is not None and unsafe(lower) <= unsafe_cutoff):
        return None
    return {"kappa": float(lower), "safe_score": safe(lower),
            "unsafe_score": unsafe(lower), "meets_stop": True}
