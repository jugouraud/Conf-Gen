"""Conformal safe sets as geometric regions, with projection operators.

Every score function in wasserstein.py / ecf.py defines, once calibrated at a
conformal radius, a REGION in R^d. This module turns each of those regions into
a first-class object with a uniform interface

    region.score(Z)      nonconformity score of each row
    region.contains(Z)   score <= radius
    region.project(Z)    the nearest point of the region to each row

so the filter can be deployed as a PROJECTION STEP rather than a reject/accept
gate: a blocked prompt is not refused, it is moved to the closest conforming
point of the safe set and generation proceeds from there.

THE GEOMETRY OF EACH SCORE
--------------------------
    score                       region                       convex  projection
    ------------------------------------------------------------------------------
    ECF / s_(d,1) [Mahalanobis] ellipsoid                     yes     exact, closed form
    s_(d,1), g_(d,1), s~_(d,1)  weighted sum-of-norms         yes     exact to tolerance
                                sublevel set                          (KKT bisection + IRLS)
    s_(d,inf), g^(0,K)_(d,inf)  union of balls                no      EXACT, closed form
    s~^(n)_(d,inf)              union of balls, per-ball r    no      EXACT, closed form
    s^(n)_(d,inf) general       union of intersections of     no      exact when c0-n <= 1;
                                unions of balls                       enumerated otherwise

WHY THE ORDER-INF REGIONS ARE UNIONS OF BALLS
---------------------------------------------
otinf_region_equivalence.tex, Theorem thm:regions, proves

    R_bn = R_cc = R_geo = { theta : tau(theta) >= c0 - n }

where G_eps(D) is the threshold graph on the anchors (edge iff d <= eps), c0 is
its number of connected components, and tau(theta) counts the components that
theta is within eps of. Hence:

    c0 - n <= 0   ->  the whole space
    c0 - n == 1   ->  a UNION OF BALLS, union_i B(theta_i, eps)
    c0 - n  > 1   ->  theta must touch at least (c0-n) of the c0 components:
                      a union over (c0-n)-subsets S of the intersections
                      intersect_{j in S} ( union_{theta_i in C_j} B(theta_i, eps) )

On the CLIP prompt data c0 = 1 in every split tested (the calibrated
eps = 13.76 exceeds the reference bottleneck B(T_M) = 10.70), so the deployed
region is the plain union of balls with the exact closed-form projection.

A NOTE ON NON-CONVEXITY
-----------------------
The order-inf regions are not convex, but projection onto a finite UNION is
still exact: project onto each piece and keep the nearest result, since
dist(z, U_k S_k) = min_k dist(z, S_k). Non-convexity costs uniqueness (a point
equidistant from two balls has two projections), not correctness.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from itertools import combinations, product


class InfeasibleProjection(RuntimeError):
    """Raised when project() cannot land inside the region (empty/unreachable)."""


def _atleast2(Z):
    Z = np.asarray(Z, dtype=float)
    return Z[None, :] if Z.ndim == 1 else Z


# ============================================================== base class ==

class ConformalRegion(ABC):
    """A calibrated conformal safe set in R^d.

    Subclasses guarantee: project(Z) lies in the region (up to `tol`), equals Z
    wherever Z is already inside, and is idempotent.
    """

    #: True if the region is a convex set
    is_convex: bool = False
    #: True if project() returns the exact metric projection (not a heuristic)
    projection_exact: bool = True
    #: numerical slack allowed when checking feasibility after projection
    tol: float = 1e-7
    #: the data points the region was built from (anchors / centroids), if known.
    #: Used by repair(); set by the builders.
    support: np.ndarray | None = None

    @abstractmethod
    def score(self, Z) -> np.ndarray:
        """Nonconformity score, one per row."""

    @abstractmethod
    def _project(self, Z) -> np.ndarray:
        """Project rows assumed to be OUTSIDE the region."""

    @property
    def is_whole_space(self) -> bool:
        return False

    @property
    def is_empty(self) -> bool:
        return False

    def contains(self, Z) -> np.ndarray:
        return self.score(_atleast2(Z)) <= self.radius + self.tol

    def project(self, Z) -> np.ndarray:
        """Nearest point of the region. Rows already inside are returned unchanged."""
        Z = _atleast2(Z)
        if self.is_whole_space:
            return Z.copy()
        if self.is_empty:
            raise ValueError(f"{type(self).__name__} is empty: nothing to project onto")
        out = Z.copy()
        outside = ~self.contains(Z)
        if outside.any():
            out[outside] = self._project(Z[outside])
            bad = ~self.contains(out)
            if bad.any():
                raise InfeasibleProjection(
                    f"{type(self).__name__}: projection landed outside the region for "
                    f"{int(bad.sum())}/{len(Z)} rows. The region is empty or "
                    f"disconnected in a way this projector cannot reach. "
                    f"(A conformally calibrated radius always yields a non-empty "
                    f"region, since the calibration quantile is attained.)")
        return out

    def repair(self, Z, t=1.0):
        """Certified repair: interpolate from the metric projection toward real data.

        The metric projection lands on the region's BOUNDARY. In high dimension
        that boundary sits about one anchor-spacing beyond the data cloud
        (measured eps/spacing ~ 1.2-1.4 here for every metric tried), so the
        projected point is inside the certified region but in a part of it that
        contains no data -- which is why projection alone leaves the generated
        image unchanged.

        This returns

            repair_t(z) = (1-t) * proj(z) + t * a*(z)

        with a*(z) the nearest SUPPORT point that is itself inside the region.
        Every point of that segment is inside the region:

          * convex regions (ellipsoid, sum-of-norms): both endpoints are in a
            convex set, so the whole segment is;
          * union of balls: project() lands in the ball around the nearest
            centre, a*(z) is that centre, and balls are convex.

        So the guarantee is preserved for every t in [0, 1]. t = 0 is the
        minimum-displacement choice; t = 1 lands exactly on a real safe point
        (maximum support fidelity). t trades one for the other.

        Regions that are neither convex nor a union of balls (a general
        ThresholdGraphRegion with q > 1) do not admit the argument; repair()
        certifies the result there and raises if it fails.
        """
        Z = _atleast2(Z)
        P = self.project(Z)
        if t <= 0 or self.support is None or len(self.support) == 0:
            return P
        sup = np.asarray(self.support, dtype=float)
        inside = self.contains(sup)
        if not inside.any():
            return P
        sup = sup[inside]
        k = np.argmin(cdist(Z, sup), axis=1)
        out = (1.0 - t) * P + t * sup[k]
        bad = ~self.contains(out)
        if bad.any():
            raise InfeasibleProjection(
                f"{type(self).__name__}.repair(t={t}) left the region for "
                f"{int(bad.sum())}/{len(Z)} rows; the region is not convex and "
                f"not a union of balls, so the segment argument does not apply.")
        return out

    def displacement(self, Z) -> np.ndarray:
        """Euclidean distance moved by project(), one per row."""
        Z = _atleast2(Z)
        return np.linalg.norm(self.project(Z) - Z, axis=1)


# =============================================================== ellipsoid ==

@dataclass
class EllipsoidRegion(ConformalRegion):
    """{ z : 1 + (z-mu)' Sigma^-1 (z-mu) <= alpha }, the ECF / Mahalanobis order-1 set.

    Sigma is the probabilistic-PCA covariance of ecf.py: top-r eigenpairs
    explicit, the remaining directions isotropic at gamma. Stored low-rank, so
    nothing of size d x d is ever formed.

    Two projections, both exact:

    mahalanobis  argmin ||y - z||_{Sigma^-1} over the ellipsoid. Scaling (z-mu)
                 by s scales the score by s^2, so s = r/d_M(z) lands exactly on
                 the boundary: a radial contraction toward mu. Closed form.

    euclidean    argmin ||y - z||_2. NOT closed form on an ellipsoid. The KKT
                 system gives y = mu + V diag(1/(1 + lam/lambda_i)) V'(z-mu),
                 with lam the root of a decreasing secular function; solved by
                 bisection to machine precision.
    """
    mean: np.ndarray            # (d,)
    V: np.ndarray               # (d, r) top-r eigenvectors
    eig: np.ndarray             # (r,) top-r eigenvalues
    gamma: float                # isotropic tail variance
    alpha: float                # conformal threshold on the score
    metric: str = "mahalanobis"
    support: np.ndarray | None = None   # the fit points, for repair()

    is_convex: bool = True

    @classmethod
    def from_safe_set(cls, ss, metric="mahalanobis", support=None):
        """Build from an ecf.EllipsoidalSafeSet."""
        r = ss.V.shape[1]
        return cls(mean=ss.mean.numpy().astype(float), V=ss.V.numpy().astype(float),
                   eig=ss.eig_reg.numpy().astype(float)[:r], gamma=float(ss.gamma),
                   alpha=float(ss.alpha), metric=metric,
                   support=None if support is None else np.asarray(support, dtype=float))

    @property
    def radius(self):
        return self.alpha

    @property
    def is_whole_space(self):
        return not np.isfinite(self.alpha)

    @property
    def is_empty(self):
        return np.isfinite(self.alpha) and self.alpha < 1.0

    def _mahal_sq(self, Z):
        diff = Z - self.mean
        c = diff @ self.V
        g_inv = 1.0 / self.gamma
        return (diff ** 2).sum(1) * g_inv + (c ** 2 * (1.0 / self.eig - g_inv)).sum(1)

    def score(self, Z):
        return 1.0 + self._mahal_sq(_atleast2(Z))

    def _project(self, Z):
        r_sq = self.alpha - 1.0
        r = np.sqrt(r_sq)
        if self.metric == "mahalanobis":
            d_m = np.sqrt(self._mahal_sq(Z))
            s = (r * (1 - 1e-12)) / np.maximum(d_m, 1e-300)
            return self.mean + (Z - self.mean) * s[:, None]

        if self.metric != "euclidean":
            raise ValueError(f"metric must be 'mahalanobis' or 'euclidean', got {self.metric}")

        # Euclidean: y = mu + (I + lam Sigma^-1)^-1 (z - mu). In the eigenbasis
        # the i-th coordinate scales by 1/(1 + lam/lambda_i); the isotropic tail
        # scales by 1/(1 + lam/gamma). f(lam) = score(y(lam)) - alpha is strictly
        # decreasing, f(0) > 0, f(inf) < 0, so bisection converges.
        out = np.empty_like(Z)
        inv_e = 1.0 / self.eig
        g_inv = 1.0 / self.gamma
        for i, z in enumerate(Z):
            diff = z - self.mean
            c = diff @ self.V                       # (r,) in-subspace coords
            p_sq = float(diff @ diff - c @ c)       # squared norm of the tail residual

            def f(lam):
                a = c / (1.0 + lam * inv_e)
                b_sq = p_sq / (1.0 + lam * g_inv) ** 2
                return float((a ** 2 * inv_e).sum() + b_sq * g_inv - r_sq)

            lo, hi = 0.0, 1.0
            while f(hi) > 0:
                hi *= 2.0
                if hi > 1e18:
                    break
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if f(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            lam = 0.5 * (lo + hi)
            a = c / (1.0 + lam * inv_e)
            tail = (diff - c @ self.V.T) / (1.0 + lam * g_inv)
            out[i] = self.mean + a @ self.V.T + tail
        return out


# ================================================== weighted sum of norms ==

@dataclass
class SumOfNormsRegion(ConformalRegion):
    """{ z : (offset + sum_i w_i ||z - a_i||) / scale <= radius }.

    Covers the whole order-1 family with one object:

        s_(d,1)    anchors = the M reference points, w_i = 1/M,      scale = M+1
        g_(d,1)    anchors = the M reference points, w = task mass,  scale = M+1
        s~_(d,1)   anchors = the K centroids,        w_k = |C_k|/M,  scale = M+1,
                   offset = sum_k sum_{i in C_k} d(theta_i, cbar_k)

    The constraint function is a nonnegative combination of norms, hence convex,
    so the region is a convex body (the sublevel set of the Fermat-Weber
    objective). Its minimum is attained at the weighted geometric median; if
    that minimum exceeds the budget the region is EMPTY, which can happen when
    the radius was calibrated under a different mass vector.

    PROJECTION. min ||y-z||^2/2 s.t. f(y) <= tau. Either z is already feasible,
    or the constraint is active and y = prox_{lam f}(z) for the lam making it
    tight. f(prox_{lam f}(z)) is non-increasing in lam, so lam is found by
    bisection; each prox is computed by the IRLS/Weiszfeld fixed point

        y <- (z + lam sum_i w_i a_i / r_i) / (1 + lam sum_i w_i / r_i),  r_i = ||y - a_i||

    which is the stationarity condition of the strongly convex prox objective
    and converges monotonically. Exact to `tol`; validated against CVXPY.
    """
    anchors: np.ndarray          # (M, d)
    weights: np.ndarray          # (M,)
    scale: float = 1.0
    offset: float = 0.0
    radius: float = np.inf
    support: np.ndarray | None = None

    is_convex: bool = True
    max_outer: int = 100
    max_inner: int = 500

    def __post_init__(self):
        self.anchors = np.asarray(self.anchors, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.support is None:
            self.support = self.anchors

    @property
    def budget(self) -> float:
        """tau: the allowance for sum_i w_i ||z - a_i|| implied by the radius."""
        return self.radius * self.scale - self.offset

    @property
    def is_whole_space(self):
        return not np.isfinite(self.radius)

    @property
    def is_empty(self):
        if self.is_whole_space:
            return False
        return self.budget < self._min_value() - 1e-9

    def _f(self, Z):
        return (cdist(_atleast2(Z), self.anchors) @ self.weights)

    def score(self, Z):
        return (self.offset + self._f(Z)) / self.scale

    def _min_value(self):
        """min_z sum_i w_i ||z - a_i||, at the weighted geometric median."""
        if not hasattr(self, "_cached_min"):
            y = self.anchors.T @ self.weights / self.weights.sum()
            for _ in range(self.max_inner):
                r = np.maximum(np.linalg.norm(self.anchors - y, axis=1), 1e-12)
                wr = self.weights / r
                y_new = (self.anchors.T @ wr) / wr.sum()
                if np.linalg.norm(y_new - y) <= 1e-12 * (1 + np.linalg.norm(y)):
                    y = y_new
                    break
                y = y_new
            self._cached_min = float(self._f(y[None, :])[0])
            self._geometric_median = y
        return self._cached_min

    def geometric_median(self):
        """The lowest-score point of the region (Corollary cor:minimal_score_fusion)."""
        self._min_value()
        return self._geometric_median

    def _prox(self, z, lam, y0=None):
        """prox_{lam f}(z) by IRLS on the stationarity condition."""
        y = z.copy() if y0 is None else y0.copy()
        for _ in range(self.max_inner):
            r = np.maximum(np.linalg.norm(self.anchors - y, axis=1), 1e-12)
            wr = self.weights / r
            denom = 1.0 + lam * wr.sum()
            y_new = (z + lam * (self.anchors.T @ wr)) / denom
            if np.linalg.norm(y_new - y) <= 1e-12 * (1 + np.linalg.norm(y)):
                return y_new
            y = y_new
        return y

    def _project(self, Z):
        tau = self.budget
        out = np.empty_like(Z)
        for i, z in enumerate(Z):
            # bracket lam
            lo, hi = 0.0, 1.0
            y = self._prox(z, hi)
            while self._f(y[None, :])[0] > tau:
                hi *= 2.0
                y = self._prox(z, hi, y0=y)
                if hi > 1e18:
                    break
            for _ in range(self.max_outer):
                mid = 0.5 * (lo + hi)
                y = self._prox(z, mid)
                if self._f(y[None, :])[0] > tau:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= 1e-14 * max(1.0, hi):
                    break
            out[i] = self._prox(z, hi)
        return out


# =========================================================== union of balls ==

@dataclass
class UnionOfBallsRegion(ConformalRegion):
    """{ z : min_k ( ||z - c_k|| - r_k ) <= 0 }, i.e. union_k B(c_k, r_k).

    The exact geometry of the order-inf scores when the threshold graph is
    connected (c0 = 1), by otinf_region_equivalence.tex cor:geo:

        s_(d,inf)          centers = anchors,   r_k = eps for all k
        g^(0,K)_(d,inf)    centers = the K task-nearest anchors, r_k = eps
        s~^(n)_(d,inf)     centers = centroids, r_k = eps - eta_inf - r_k^cluster
                           (per-ball radii, from the cluster padding)

    Non-convex, but the projection is EXACT and closed form: for a finite union,
    dist(z, U_k S_k) = min_k dist(z, S_k), so project onto every ball and keep
    the nearest. Balls with r_k <= 0 are empty and are skipped.
    """
    centers: np.ndarray          # (K, d)
    radii: np.ndarray            # (K,)
    support: np.ndarray | None = None

    is_convex: bool = False

    def __post_init__(self):
        self.centers = np.asarray(self.centers, dtype=float)
        self.radii = np.asarray(self.radii, dtype=float).ravel()
        if self.radii.size == 1:
            self.radii = np.full(len(self.centers), float(self.radii[0]))
        self._live = self.radii > 0
        if self.support is None:
            self.support = self.centers

    @property
    def radius(self):
        return 0.0

    @property
    def is_empty(self):
        return not self._live.any()

    def score(self, Z):
        """Signed distance to the union: <= 0 inside."""
        Z = _atleast2(Z)
        D = cdist(Z, self.centers) - self.radii[None, :]
        D[:, ~self._live] = np.inf
        return D.min(axis=1)

    def _project(self, Z):
        C = self.centers[self._live]
        R = self.radii[self._live]
        D = cdist(Z, C)                                  # (n, K)
        # distance to each ball's surface (0 if already inside that ball)
        gap = np.maximum(D - R[None, :], 0.0)
        k = np.argmin(gap, axis=1)
        d_k = np.maximum(D[np.arange(len(Z)), k], 1e-300)
        c_k, r_k = C[k], R[k]
        # walk from z toward c_k until on the sphere; shrink by 1 ulp of slack
        s = (r_k * (1 - 1e-12)) / d_k
        return c_k + (Z - c_k) * s[:, None]


# ======================================================= threshold graph ====

@dataclass
class ThresholdGraphRegion(ConformalRegion):
    """The general order-inf n-cuts region, { theta : tau(theta) >= c0 - n }.

    otinf_region_equivalence.tex thm:regions. G_eps(D) is the threshold graph on
    the anchors; c0 its component count; tau(theta) the number of components
    within eps of theta. Three regimes:

        q := c0 - n <= 0   the whole space (cor:geo)
        q == 1             a union of balls   -> exact closed-form projection
        q  > 1             theta must touch >= q of the c0 components:
                           union over q-subsets S of  intersect_{j in S} U_j,
                           with U_j the union of balls of component j

    For q > 1 the projection enumerates q-subsets and, within each, one ball per
    component (a `selection`); projecting onto an INTERSECTION of balls is a
    convex problem solved by Dykstra's alternating projections. The result is
    always feasible; it is the exact minimum when `exhaustive=True` (all
    selections examined) and otherwise a certified-feasible upper bound obtained
    from the nearest ball of each component. Enumeration is capped by
    `max_selections`; `projection_exact` reports which happened on the last call.

    In practice on the CLIP prompt data q = 1 always, so the exact closed-form
    branch is the one that runs.
    """
    anchors: np.ndarray
    eps: float
    n_cuts: int = 0
    exhaustive: bool = False
    max_selections: int = 20000
    support: np.ndarray | None = None

    is_convex: bool = False

    def __post_init__(self):
        self.anchors = np.asarray(self.anchors, dtype=float)
        D = cdist(self.anchors, self.anchors)
        A = csr_array(((D <= self.eps) & ~np.eye(len(D), dtype=bool)).astype(np.int8))
        self.c0, self.labels_ = connected_components(A, directed=False)
        self.projection_exact = True
        if self.support is None:
            self.support = self.anchors

    @property
    def q(self) -> int:
        """Number of components the candidate must be eps-adjacent to."""
        return self.c0 - int(self.n_cuts)

    @property
    def radius(self):
        return 0.0

    @property
    def is_whole_space(self):
        return self.q <= 0

    @property
    def is_empty(self):
        """Probe for feasibility.

        q <= 1 is never empty (every anchor is in its own ball). For q > 1 the
        candidate must be eps-close to q different components at once, which is
        impossible when those components lie more than 2 eps apart -- the usual
        situation for a radius that was NOT conformally calibrated. The probe
        projects one candidate and reports whether it landed inside.
        """
        if self.is_whole_space or self.q <= 1:
            return False
        if getattr(self, "_empty_cache", None) is None:
            z = self.anchors.mean(axis=0)[None, :]
            try:
                self._empty_cache = not bool(self.contains(self._project(z))[0])
            except Exception:
                self._empty_cache = True
        return self._empty_cache

    def _tau(self, Z):
        D = cdist(_atleast2(Z), self.anchors) <= self.eps
        return np.array([[D[i, self.labels_ == j].any() for j in range(self.c0)]
                         for i in range(len(D))])

    def score(self, Z):
        """(q - tau) : <= 0 exactly when the candidate is feasible."""
        return self.q - self._tau(_atleast2(Z)).sum(axis=1)

    def repair(self, Z, t=1.0):
        if self.q <= 1:
            return self.as_union_of_balls().repair(Z, t)
        return super().repair(Z, t)

    def as_union_of_balls(self) -> UnionOfBallsRegion:
        """Valid only when q <= 1; the region is then union_i B(theta_i, eps)."""
        if self.q > 1:
            raise ValueError(f"q = {self.q} > 1: the region is not a plain union of balls")
        return UnionOfBallsRegion(self.centers_all(), np.full(len(self.anchors), self.eps))

    def centers_all(self):
        return self.anchors

    def _project_onto_ball_intersection(self, z, centers, radius, iters=2000):
        """Dykstra's algorithm: exact projection onto an intersection of balls."""
        x = z.copy()
        p = [np.zeros_like(z) for _ in centers]
        for _ in range(iters):
            x_prev = x.copy()
            for j, c in enumerate(centers):
                y = x + p[j]
                dv = y - c
                nrm = np.linalg.norm(dv)
                x_new = c + dv * (radius / nrm) if nrm > radius else y
                p[j] = y - x_new
                x = x_new
            if np.linalg.norm(x - x_prev) <= 1e-13 * (1 + np.linalg.norm(x)):
                break
        return x

    def _project(self, Z):
        if self.q <= 1:
            return self.as_union_of_balls()._project(Z)

        comps = [np.flatnonzero(self.labels_ == j) for j in range(self.c0)]
        out = np.empty_like(Z)
        exact = True
        for i, z in enumerate(Z):
            best, best_d = None, np.inf
            for S in combinations(range(self.c0), self.q):
                pools = [comps[j] for j in S]
                n_sel = int(np.prod([len(pool) for pool in pools]))
                if not self.exhaustive or n_sel > self.max_selections:
                    # keep only each component's ball nearest to z: always
                    # feasible, but no longer certified minimal
                    exact = False
                    pools = [pool[[np.argmin(np.linalg.norm(self.anchors[pool] - z, axis=1))]]
                             for pool in pools]
                for sel in product(*pools):
                    y = self._project_onto_ball_intersection(
                        z, self.anchors[list(sel)], self.eps * (1 - 1e-12))
                    d = np.linalg.norm(y - z)
                    if d < best_d:
                        best, best_d = y, d
            out[i] = best
        self.projection_exact = exact
        return out


# ================================================================ builders ==

def region_from_order1(anchors, radius, weights=None, scale=None, offset=0.0,
                       support=None):
    """s_(d,1) / g_(d,1) / s~_(d,1)  ->  SumOfNormsRegion."""
    M = len(anchors)
    if weights is None:
        weights = np.full(M, 1.0 / M)
    if scale is None:
        scale = M + 1.0
    return SumOfNormsRegion(anchors=anchors, weights=weights, scale=float(scale),
                            offset=float(offset), radius=float(radius),
                            support=None if support is None else np.asarray(support, float))


def region_from_orderinf(anchors, radius, n_cuts=0):
    """s^(n)_(d,inf) / g^(n,K)_(d,inf)  ->  union of balls, or the general graph region."""
    reg = ThresholdGraphRegion(anchors=anchors, eps=float(radius), n_cuts=int(n_cuts))
    if reg.q == 1:
        return reg.as_union_of_balls()
    return reg


def region_from_lowcard_orderinf(lc, radius):
    """s~^(n)_(d,inf)  ->  union of balls with per-centroid radii.

    The score is eta + max(B^(n), min_k [||z - c_k|| + r_k]) <= radius, and eta,
    B^(n) are constants, so the region is { z : min_k [||z-c_k|| + r_k] <= budget }
    with budget = radius - eta, i.e. union_k B(c_k, budget - r_k). If
    budget < B^(n) the score floor already exceeds the radius and the region is
    empty; if every (budget - r_k) <= 0 it is empty too.
    """
    budget = float(radius) - float(lc.eta_inf)
    if budget < float(lc.B_n):
        return UnionOfBallsRegion(np.asarray(lc.centroids), np.zeros(len(lc.centroids)))
    return UnionOfBallsRegion(np.asarray(lc.centroids), budget - np.asarray(lc.r_k))


def region_from_ecf(safe_set, metric="mahalanobis"):
    """ecf.EllipsoidalSafeSet  ->  EllipsoidRegion."""
    return EllipsoidRegion.from_safe_set(safe_set, metric=metric)
