"""Correctness tests for the conformal regions and their projection operators.

Three kinds of check:

  GEOMETRY    the region's membership test agrees with the score function it was
              derived from -- in particular a numerical verification of
              otinf_region_equivalence.tex thm:regions, R_bn = R_geo, against a
              real MST computation.
  PROJECTION  every projector returns a point that is (a) inside the region,
              (b) unchanged if the input was already inside, (c) idempotent, and
              (d) where a reference exists, EQUAL to the true metric projection
              computed independently (CVXPY for the convex regions, brute force
              for the unions).
  CONFORMAL   building a region from a calibrated radius preserves coverage.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from conformal_regions import (
    EllipsoidRegion, SumOfNormsRegion, UnionOfBallsRegion, ThresholdGraphRegion,
    InfeasibleProjection, region_from_order1, region_from_orderinf,
    region_from_lowcard_orderinf, region_from_ecf,
)
from wasserstein import score_order1, score_orderinf, conformal_radius, LowCardOrderInf

RNG = np.random.default_rng(0)
try:
    import cvxpy as cp
except ImportError:                                        # pragma: no cover
    cp = None


# ================================================================ geometry ==

class TestRegionEquivalence:
    """R_bn = R_geo (otinf_region_equivalence.tex thm:regions), checked on real MSTs."""

    @pytest.mark.parametrize("n_cuts", [0, 1, 2, 3])
    @pytest.mark.parametrize("d", [2, 8])
    def test_bottleneck_region_equals_threshold_graph_region(self, n_cuts, d):
        rng = np.random.default_rng(n_cuts * 10 + d)
        A = rng.normal(size=(24, d))
        Z = rng.normal(size=(120, d)) * 1.6
        # sweep eps across the whole interesting range
        for eps in np.quantile(cdist(A, A)[np.triu_indices(24, 1)], [.05, .15, .3, .5, .8]):
            bn = score_orderinf(Z, A, n_cuts=n_cuts) <= eps + 1e-12
            geo = ThresholdGraphRegion(A, eps=eps, n_cuts=n_cuts).contains(Z)
            assert np.array_equal(bn, geo), (
                f"n={n_cuts} d={d} eps={eps:.4f}: {np.sum(bn != geo)} disagreements")

    def test_union_of_balls_branch_matches_bottleneck(self):
        """When c0 - n == 1 the region must be exactly union_i B(a_i, eps)."""
        rng = np.random.default_rng(5)
        A = rng.normal(size=(30, 6))
        eps = np.quantile(cdist(A, A)[np.triu_indices(30, 1)], .5)
        reg = region_from_orderinf(A, eps, n_cuts=0)
        assert isinstance(reg, UnionOfBallsRegion)
        Z = rng.normal(size=(200, 6)) * 1.5
        assert np.array_equal(reg.contains(Z), score_orderinf(Z, A, 0) <= eps + 1e-12)

    def test_whole_space_when_n_ge_c0(self):
        rng = np.random.default_rng(6)
        A = rng.normal(size=(10, 4))
        reg = ThresholdGraphRegion(A, eps=1e-6, n_cuts=20)     # n >= c0
        assert reg.is_whole_space
        Z = rng.normal(size=(5, 4)) * 100
        assert reg.contains(Z).all()
        assert np.allclose(reg.project(Z), Z)

    def test_disconnected_with_no_cuts_is_empty(self):
        A = np.array([[0., 0.], [100., 0.], [0., 100.]])
        reg = ThresholdGraphRegion(A, eps=1.0, n_cuts=0)
        assert reg.c0 == 3 and reg.q == 3 and reg.is_empty
        with pytest.raises(ValueError):
            reg.project(np.zeros((1, 2)))


# ============================================================== projection ==

class TestEllipsoidProjection:
    def _region(self, d=6, metric="mahalanobis", seed=0):
        rng = np.random.default_rng(seed)
        V, _ = np.linalg.qr(rng.normal(size=(d, d)))
        r = 3
        return EllipsoidRegion(mean=rng.normal(size=d), V=V[:, :r],
                               eig=np.array([9.0, 4.0, 2.0]), gamma=0.5,
                               alpha=5.0, metric=metric)

    @pytest.mark.parametrize("metric", ["mahalanobis", "euclidean"])
    def test_lands_on_the_region(self, metric):
        reg = self._region(metric=metric)
        Z = np.random.default_rng(1).normal(size=(30, 6)) * 6
        assert reg.contains(reg.project(Z)).all()

    @pytest.mark.parametrize("metric", ["mahalanobis", "euclidean"])
    def test_inside_points_untouched_and_idempotent(self, metric):
        reg = self._region(metric=metric)
        Z = np.random.default_rng(2).normal(size=(30, 6)) * 6
        P = reg.project(Z)
        inside = reg.contains(Z)
        assert np.allclose(P[inside], Z[inside])
        assert np.allclose(reg.project(P), P, atol=1e-8)

    def test_mahalanobis_projection_is_on_the_boundary(self):
        reg = self._region(metric="mahalanobis")
        Z = np.random.default_rng(3).normal(size=(20, 6)) * 10
        out = ~reg.contains(Z)
        assert np.allclose(reg.score(reg.project(Z))[out], reg.alpha, atol=1e-6)

    @pytest.mark.skipif(cp is None, reason="cvxpy not installed")
    def test_euclidean_projection_matches_cvxpy(self):
        reg = self._region(metric="euclidean")
        d = len(reg.mean)
        g_inv = 1.0 / reg.gamma
        Sinv = g_inv * np.eye(d) + reg.V @ np.diag(1.0 / reg.eig - g_inv) @ reg.V.T
        Sinv = (Sinv + Sinv.T) / 2
        Z = np.random.default_rng(4).normal(size=(6, d)) * 5
        P = reg.project(Z)
        for z, p in zip(Z, P):
            y = cp.Variable(d)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(y - z)),
                              [cp.quad_form(y - reg.mean, cp.psd_wrap(Sinv)) <= reg.alpha - 1.0])
            prob.solve(solver=cp.CLARABEL)
            # compare the achieved objective: near a flat optimum the argmin can
            # differ by more than the solver's own tolerance
            assert np.linalg.norm(p - z) <= np.linalg.norm(y.value - z) + 1e-6
            assert np.linalg.norm(p - y.value) < 1e-3

    def test_euclidean_is_closer_than_mahalanobis_in_l2(self):
        """Sanity: the Euclidean projector really does minimise the L2 distance."""
        rm = self._region(metric="mahalanobis")
        re = self._region(metric="euclidean")
        Z = np.random.default_rng(7).normal(size=(20, 6)) * 8
        out = ~rm.contains(Z)
        d_m = np.linalg.norm(rm.project(Z) - Z, axis=1)[out]
        d_e = np.linalg.norm(re.project(Z) - Z, axis=1)[out]
        assert (d_e <= d_m + 1e-8).all()
        assert (d_e < d_m - 1e-6).any()


class TestSumOfNormsProjection:
    def _region(self, M=20, d=5, seed=0, radius=None):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(M, d))
        reg = region_from_order1(A, radius=1.0)
        if radius is None:                       # pick a radius that bites but is feasible
            lo = reg._min_value() / (M + 1.0)
            radius = lo * 1.35
        return region_from_order1(A, radius=radius)

    def test_lands_on_the_region(self):
        reg = self._region()
        Z = np.random.default_rng(1).normal(size=(25, 5)) * 4
        assert reg.contains(reg.project(Z)).all()

    def test_boundary_is_active(self):
        reg = self._region()
        Z = np.random.default_rng(2).normal(size=(25, 5)) * 4
        out = ~reg.contains(Z)
        assert np.allclose(reg.score(reg.project(Z))[out], reg.radius, rtol=1e-6)

    def test_inside_untouched_and_idempotent(self):
        reg = self._region()
        Z = np.random.default_rng(3).normal(size=(25, 5)) * 4
        P = reg.project(Z)
        assert np.allclose(P[reg.contains(Z)], Z[reg.contains(Z)])
        assert np.allclose(reg.project(P), P, atol=1e-7)

    @pytest.mark.skipif(cp is None, reason="cvxpy not installed")
    def test_matches_cvxpy(self):
        reg = self._region()
        Z = np.random.default_rng(4).normal(size=(6, 5)) * 4
        P = reg.project(Z)
        tau = reg.budget
        for z, p in zip(Z, P):
            y = cp.Variable(5)
            con = cp.sum(cp.multiply(reg.weights,
                                     cp.norm(y[None, :] - reg.anchors, axis=1))) <= tau
            prob = cp.Problem(cp.Minimize(cp.sum_squares(y - z)), [con])
            prob.solve(solver=cp.CLARABEL)
            assert np.linalg.norm(p - z) <= np.linalg.norm(y.value - z) + 1e-6
            assert np.linalg.norm(p - y.value) < 1e-3

    def test_empty_region_is_detected(self):
        """A budget below the geometric-median value makes the region empty."""
        rng = np.random.default_rng(9)
        A = rng.normal(size=(15, 4))
        reg = region_from_order1(A, radius=1e-6)
        assert reg.is_empty
        with pytest.raises(ValueError):
            reg.project(rng.normal(size=(1, 4)))

    def test_geometric_median_is_the_lowest_score_point(self):
        reg = self._region()
        gm = reg.geometric_median()
        Z = np.random.default_rng(11).normal(size=(200, 5)) * 2
        assert reg.score(gm[None, :])[0] <= reg.score(Z).min() + 1e-9

    def test_weighted_case_matches_adaptive_score(self):
        """g_(d,1) with a task mass vector is the same object with `weights` set."""
        rng = np.random.default_rng(12)
        A = rng.normal(size=(18, 5))
        w = rng.random(18); w /= w.sum()
        reg = SumOfNormsRegion(anchors=A, weights=w, scale=len(A) + 1.0, radius=1.0)
        Z = rng.normal(size=(10, 5))
        want = (cdist(Z, A) @ w) / (len(A) + 1.0)
        assert np.allclose(reg.score(Z), want)


class TestUnionOfBallsProjection:
    def test_matches_brute_force(self):
        rng = np.random.default_rng(1)
        C = rng.normal(size=(12, 5))
        R = rng.uniform(0.4, 1.2, 12)
        reg = UnionOfBallsRegion(C, R)
        Z = rng.normal(size=(40, 5)) * 3
        P = reg.project(Z)
        for z, p in zip(Z, P):
            # brute force: project onto every ball, keep the nearest
            best = None
            for c, r in zip(C, R):
                v = z - c
                nv = np.linalg.norm(v)
                q = z if nv <= r else c + v * (r / nv)
                if best is None or np.linalg.norm(q - z) < np.linalg.norm(best - z):
                    best = q
            assert np.linalg.norm(p - z) <= np.linalg.norm(best - z) + 1e-8

    def test_lands_inside_and_idempotent(self):
        rng = np.random.default_rng(2)
        reg = UnionOfBallsRegion(rng.normal(size=(10, 6)), np.full(10, 0.9))
        Z = rng.normal(size=(30, 6)) * 3
        P = reg.project(Z)
        assert reg.contains(P).all()
        assert np.allclose(reg.project(P), P, atol=1e-9)

    def test_zero_radius_balls_are_skipped(self):
        C = np.array([[0., 0.], [5., 0.], [10., 0.]])
        reg = UnionOfBallsRegion(C, np.array([-1.0, 1.0, 0.0]))
        p = reg.project(np.array([[9.0, 0.0]]))[0]
        assert np.allclose(p, [6.0, 0.0], atol=1e-6)      # only ball 1 is live

    def test_all_empty_raises(self):
        reg = UnionOfBallsRegion(np.zeros((3, 2)), np.array([0.0, -1.0, -2.0]))
        assert reg.is_empty
        with pytest.raises(ValueError):
            reg.project(np.ones((1, 2)))

    def test_lowcard_orderinf_builder(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(60, 5))
        lc = LowCardOrderInf.fit(A, K=8, n_cuts=0)
        Z = rng.normal(size=(30, 5)) * 2
        radius = float(np.quantile(lc(A), 0.9))
        reg = region_from_lowcard_orderinf(lc, radius)
        assert np.array_equal(reg.contains(Z), lc(Z) <= radius + 1e-9)
        if not reg.is_empty:
            assert reg.contains(reg.project(Z)).all()


# =============================================================== conformal ==

def _assert_valid_coverage(cov, eps, K):
    """Coverage must match split CP's EXACT expectation, within Monte-Carlo error.

    Split conformal delivers ceil((K+1)(1-eps))/(K+1), which sits just above
    1-eps; asserting >= 1-eps on a finite number of splits fails on noise alone.
    """
    cov = np.asarray(cov)
    exact = np.ceil((K + 1) * (1 - eps)) / (K + 1)
    se = cov.std(ddof=1) / np.sqrt(len(cov))
    assert cov.mean() >= exact - 3 * se, (
        f"coverage {cov.mean():.4f} vs exact expectation {exact:.4f} "
        f"(s.e. {se:.4f}, z = {(cov.mean() - exact) / se:.2f})")


class TestConformalValidityIsPreserved:
    @pytest.mark.parametrize("eps", [0.05, 0.1, 0.2])
    def test_order1_region_coverage(self, eps):
        rng = np.random.default_rng(20)
        cov = []
        for _ in range(150):
            A, cal, tst = (rng.normal(size=(40, 4)), rng.normal(size=(60, 4)),
                           rng.normal(size=(60, 4)))
            r = conformal_radius(score_order1(cal, A), eps)
            cov.append(region_from_order1(A, r).contains(tst).mean())
        _assert_valid_coverage(cov, eps, K=60)

    @pytest.mark.parametrize("eps", [0.05, 0.1, 0.2])
    def test_orderinf_region_coverage(self, eps):
        rng = np.random.default_rng(21)
        cov = []
        for _ in range(150):
            A, cal, tst = (rng.normal(size=(40, 4)), rng.normal(size=(60, 4)),
                           rng.normal(size=(60, 4)))
            r = conformal_radius(score_orderinf(cal, A, 0), eps)
            cov.append(region_from_orderinf(A, r, 0).contains(tst).mean())
        _assert_valid_coverage(cov, eps, K=60)

    def test_projection_makes_everything_conforming(self):
        """The point of a projection filter: after it, nothing is out of the set."""
        rng = np.random.default_rng(22)
        A, cal = rng.normal(size=(40, 5)), rng.normal(size=(60, 5))
        for build, score in [
            (lambda r: region_from_order1(A, r), lambda Z: score_order1(Z, A)),
            (lambda r: region_from_orderinf(A, r, 0), lambda Z: score_orderinf(Z, A, 0)),
        ]:
            reg = build(conformal_radius(score(cal), 0.1))
            Z = rng.normal(size=(80, 5)) * 2.5
            assert reg.contains(reg.project(Z)).all()


# ================================================================== repair ==

class TestCertifiedRepair:
    """repair_t interpolates from the metric projection toward real data.

    The defining property: every t in [0, 1] stays inside the region, so the
    conformal guarantee is preserved while support fidelity improves.
    """

    def _regions(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(40, 10))
        cal = rng.normal(size=(60, 10))
        r_inf = conformal_radius(score_orderinf(cal, A, 0), 0.05)
        r_1 = conformal_radius(score_order1(cal, A), 0.05)
        return A, {"union_of_balls": region_from_orderinf(A, r_inf),
                   "sum_of_norms": region_from_order1(A, r_1)}

    @pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_every_t_stays_inside_the_region(self, t):
        A, regs = self._regions()
        Z = np.random.default_rng(1).normal(size=(30, 10)) * 2.5
        for name, reg in regs.items():
            assert reg.contains(reg.repair(Z, t)).all(), (name, t)

    def test_t_zero_is_the_metric_projection(self):
        A, regs = self._regions()
        Z = np.random.default_rng(2).normal(size=(20, 10)) * 2.5
        for reg in regs.values():
            assert np.allclose(reg.repair(Z, 0.0), reg.project(Z))

    def test_t_one_lands_exactly_on_a_support_point(self):
        A, regs = self._regions()
        Z = np.random.default_rng(3).normal(size=(20, 10)) * 2.5
        for name, reg in regs.items():
            out = reg.repair(Z, 1.0)
            d = cdist(out, A).min(axis=1)
            assert np.allclose(d, 0.0, atol=1e-9), (name, d.max())

    def test_support_fidelity_improves_monotonically(self):
        A, regs = self._regions()
        Z = np.random.default_rng(4).normal(size=(30, 10)) * 2.5
        for name, reg in regs.items():
            dnn = [cdist(reg.repair(Z, t), A).min(axis=1).mean()
                   for t in [0.0, 0.25, 0.5, 0.75, 1.0]]
            assert all(a > b for a, b in zip(dnn, dnn[1:])), (name, dnn)

    def test_displacement_grows_monotonically(self):
        A, regs = self._regions()
        Z = np.random.default_rng(5).normal(size=(30, 10)) * 2.5
        for name, reg in regs.items():
            mv = [np.linalg.norm(reg.repair(Z, t) - Z, axis=1).mean()
                  for t in [0.0, 0.25, 0.5, 0.75, 1.0]]
            assert all(a < b for a, b in zip(mv, mv[1:])), (name, mv)

    def test_points_already_inside_are_untouched_at_any_t(self):
        A, regs = self._regions()
        for reg in regs.values():
            inside = A[reg.contains(A)][:5]
            for t in [0.0, 0.5, 1.0]:
                assert np.allclose(reg.repair(inside, t), inside, atol=1e-9)

    def test_no_support_falls_back_to_projection(self):
        A, regs = self._regions()
        reg = regs["union_of_balls"]
        reg.support = None
        Z = np.random.default_rng(6).normal(size=(10, 10)) * 2.5
        assert np.allclose(reg.repair(Z, 1.0), reg.project(Z))
