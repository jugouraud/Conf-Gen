"""Tests for the Wasserstein support-estimation scores.

Checks each score against the manuscript's definition, including the identities
that must hold between the adaptive and non-adaptive forms, and the
upper-bound properties that the low-cardinality propositions rely on.
"""

import math
import numpy as np
import torch
import pytest
from scipy.spatial.distance import cdist
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree

from wasserstein import (
    score_order1, score_orderinf, bottleneck_of,
    reweighted_score_order1, reweighted_score_orderinf,
    Reweighter, uniform_mass, effective_support,
    LowCardOrder1, LowCardOrderInf, conformal_radius,
)

RNG = np.random.default_rng(0)


def naive_orderinf(theta, ref, n_cuts=0, metric="euclidean"):
    """Reference implementation: full (M+1)x(M+1) MST per candidate.

    This is exact_score_orderinf from meta_linreg_benchtest_summer_2026.py.
    """
    theta = np.atleast_2d(theta)
    out = np.empty(len(theta))
    for i, th in enumerate(theta):
        Y = np.vstack([ref, th.reshape(1, -1)])
        D = cdist(Y, Y, metric=metric)
        mst = minimum_spanning_tree(csr_array(D))
        out[i] = np.sort(mst.toarray().astype(float).flatten())[-(int(n_cuts) + 1)]
    return out


# ------------------------------------------------------------------ order 1 --

class TestOrder1:
    def test_matches_definition(self):
        ref = RNG.normal(size=(20, 5))
        th = RNG.normal(size=(3, 5))
        M = len(ref)
        want = cdist(th, ref).sum(1) / (M * (M + 1))
        assert np.allclose(score_order1(th, ref), want)

    def test_adaptive_reduces_to_exact_under_uniform_mass(self):
        """g_{d,1} with w_i = 1/M is exactly s_{d,1} (adaptation.tex)."""
        ref = RNG.normal(size=(30, 8))
        th = RNG.normal(size=(10, 8))
        a = score_order1(th, ref)
        b = reweighted_score_order1(th, ref, uniform_mass(len(ref)))
        assert np.allclose(a, b)

    def test_minimised_at_geometric_median(self):
        """The lowest-score point is the (weighted) geometric median, not far away."""
        ref = RNG.normal(size=(40, 4))
        centre = score_order1(ref.mean(0, keepdims=True), ref)
        far = score_order1(ref.mean(0, keepdims=True) + 50.0, ref)
        assert centre < far


# ---------------------------------------------------------------- order inf --

class TestOrderInf:
    @pytest.mark.parametrize("n_cuts", [0, 1, 2])
    def test_cycle_property_shortcut_matches_naive(self, n_cuts):
        """MST(D u {theta}) subset MST(D) u delta(theta): the shortcut is exact."""
        ref = RNG.normal(size=(25, 6))
        th = RNG.normal(size=(15, 6)) * 2.0
        got = score_orderinf(th, ref, n_cuts=n_cuts)
        want = naive_orderinf(th, ref, n_cuts=n_cuts)
        assert np.allclose(got, want), f"max diff {np.abs(got - want).max()}"

    def test_cuts_lower_the_score(self):
        """B^{(n)} is the (n+1)-th heaviest edge, so it decreases in n."""
        ref = RNG.normal(size=(30, 5))
        th = RNG.normal(size=(8, 5)) * 3
        s0 = score_orderinf(th, ref, n_cuts=0)
        s1 = score_orderinf(th, ref, n_cuts=1)
        s2 = score_orderinf(th, ref, n_cuts=2)
        assert np.all(s1 <= s0 + 1e-12) and np.all(s2 <= s1 + 1e-12)

    def test_bottleneck_is_nn_distance_when_far(self):
        """A far-away candidate attaches by its nearest-neighbour edge, which is
        then the MST bottleneck."""
        ref = RNG.normal(size=(20, 3))
        th = np.array([[100.0, 100.0, 100.0]])
        nn = cdist(th, ref).min()
        assert np.isclose(score_orderinf(th, ref, n_cuts=0)[0], nn)

    def test_adaptive_orderinf_equals_exact_on_selected_subset(self):
        ref = RNG.normal(size=(30, 5))
        th = RNG.normal(size=(6, 5))
        w = RNG.random(30)
        K = 8
        idx = np.argsort(w)[-K:]
        got = reweighted_score_orderinf(th, ref, w, K=K, n_cuts=1)
        want = score_orderinf(th, ref[idx], n_cuts=1)
        assert np.allclose(got, want)


# ------------------------------------------------------------- reweighting ---

class TestReweighter:
    def _clouds(self, n=12, L=7, d=4, spread=1.0):
        return [RNG.normal(size=(L, d)) * spread + i for i in range(n)]

    def test_masses_are_on_the_simplex(self):
        clouds = self._clouds()
        for scheme in ["uniform", "inv_power", "softmax", "knn", "knn_uniform", "top1"]:
            rw = Reweighter(clouds, scheme=scheme, knn=3)
            w = rw.weights_for(clouds[0])
            assert np.all(w >= 0) and np.isclose(w.sum(), 1.0), scheme

    def test_identity_condition_inv_power(self):
        """eq:reweight_identity: a coincident task takes (essentially) all the mass."""
        clouds = self._clouds()
        rw = Reweighter(clouds, scheme="inv_power", rescaling_order=1.0)
        w = rw.weights_for(clouds[3])
        assert np.argmax(w) == 3 and w[3] > 0.99

    def test_identity_condition_top1_exact(self):
        clouds = self._clouds()
        rw = Reweighter(clouds, scheme="top1")
        w = rw.weights_for(clouds[5])
        assert w[5] == 1.0 and w.sum() == 1.0

    def test_softmax_does_not_satisfy_identity(self):
        """Documented trade-off: softmax keeps mass everywhere at finite bandwidth."""
        clouds = self._clouds()
        rw = Reweighter(clouds, scheme="softmax", rescaling_order=1.0)
        w = rw.weights_for(clouds[3])
        assert w[3] < 0.99 and np.all(w > 0)

    def test_effective_support_ordering(self):
        """top1 concentrates hardest, uniform least."""
        clouds = self._clouds(n=20)
        es = {}
        for scheme, kw in [("uniform", {}), ("softmax", {}), ("inv_power", {}),
                           ("knn", dict(knn=5)), ("top1", {})]:
            rw = Reweighter(clouds, scheme=scheme, **kw)
            es[scheme] = effective_support(rw.weights_for(clouds[0]))
        assert np.isclose(es["uniform"], 20.0)
        assert es["top1"] == pytest.approx(1.0)
        assert es["knn"] <= 5.0 + 1e-9


# -------------------------------------------------------- low cardinality ----

class TestLowCardinality:
    def test_order1_upper_bounds_exact(self):
        """pr:suboptimal_cluster: s~_{d,1} >= W_{d,1} to the clustered measure.

        We check the weaker, directly meaningful statement that the clustered
        score upper-bounds the exact one, which is what makes it a valid
        (conservative) support estimator.
        """
        ref = RNG.normal(size=(120, 6))
        th = RNG.normal(size=(25, 6)) * 1.5
        lc = LowCardOrder1.fit(ref, K=15)
        assert np.all(lc(th) >= score_order1(th, ref) - 1e-9)

    def test_order1_tends_to_exact_as_K_grows(self):
        ref = RNG.normal(size=(80, 4))
        th = RNG.normal(size=(20, 4))
        exact = score_order1(th, ref)
        gaps = [np.mean(LowCardOrder1.fit(ref, K=K)(th) - exact) for K in (5, 20, 80)]
        assert gaps[0] > gaps[1] > gaps[2]
        assert gaps[-1] == pytest.approx(0.0, abs=1e-9)

    def test_orderinf_upper_bounds_exact(self):
        ref = RNG.normal(size=(100, 5))
        th = RNG.normal(size=(30, 5)) * 1.5
        lc = LowCardOrderInf.fit(ref, K=12, n_cuts=0)
        assert np.all(lc(th) >= score_orderinf(th, ref, n_cuts=0) - 1e-9)

    def test_orderinf_memory_is_K_plus_scalars(self):
        ref = RNG.normal(size=(200, 7))
        lc = LowCardOrderInf.fit(ref, K=10)
        assert lc.centroids.shape == (10, 7) and lc.r_k.shape == (10,)
        assert np.isfinite(lc.eta_inf) and np.isfinite(lc.B_n)


# ------------------------------------------------------------- conformal -----

class TestConformal:
    def test_matches_order_statistic(self):
        s = np.arange(1.0, 21.0)              # K = 20
        # ceil(0.9 * 21) = 19  ->  19th smallest = 19.0
        assert conformal_radius(s, 0.10) == 19.0

    def test_returns_inf_when_too_few_calibration_points(self):
        """K < ceil(1/eps) - 1  ->  radius is +inf, region is everything."""
        assert math.isinf(conformal_radius(np.arange(10.0), 0.05))
        assert not math.isinf(conformal_radius(np.arange(19.0), 0.05))

    @pytest.mark.parametrize("eps", [0.05, 0.1, 0.2])
    def test_empirical_coverage_is_valid(self, eps):
        rng = np.random.default_rng(1)
        cov = []
        for _ in range(400):
            ref = rng.normal(size=(40, 3))
            cal = rng.normal(size=(60, 3))
            tst = rng.normal(size=(60, 3))
            r = conformal_radius(score_order1(cal, ref), eps)
            cov.append(np.mean(score_order1(tst, ref) <= r))
        assert np.mean(cov) >= 1 - eps, f"{np.mean(cov):.4f} < {1 - eps}"


# ---------------------------------------------------------- deployable API ---

class TestPromptFilter:
    def _fixture(self, n=90, d=64, L=6, seed=0):
        rng = np.random.default_rng(seed)
        theta = rng.normal(size=(n, d)) @ np.diag(np.linspace(4, 0.3, d))
        clouds = [theta[i][None, :] + rng.normal(0, 0.05, (L, d)) for i in range(n)]
        return theta, clouds

    def test_both_variants_fit_and_hold_coverage(self):
        from prompt_filter import PromptFilter
        theta, clouds = self._fixture()
        ref, cal, tst = slice(0, 40), slice(40, 75), slice(75, 90)
        for v in ("orderinf", "order1"):
            f = PromptFilter.fit(theta[ref], clouds[ref], theta[cal], clouds[cal],
                                 epsilon=0.10, variant=v)
            assert np.isfinite(f.radius)
            acc = f.accepts(theta[tst], clouds[tst]).mean()
            assert acc >= 0.6, f"{v} accepted only {acc}"

    def test_outliers_are_rejected(self):
        from prompt_filter import PromptFilter
        theta, clouds = self._fixture()
        f = PromptFilter.fit(theta[:40], clouds[:40], theta[40:75], clouds[40:75],
                             epsilon=0.10, variant="orderinf")
        out = theta[:5] + 50.0
        out_clouds = [o[None, :] + np.zeros((6, theta.shape[1])) for o in out]
        assert not f.accepts(out, out_clouds).any()

    def test_cuts_warn(self):
        from prompt_filter import PromptFilter
        theta, clouds = self._fixture()
        with pytest.warns(RuntimeWarning, match="n_cuts"):
            PromptFilter.fit(theta[:40], clouds[:40], theta[40:75], clouds[40:75],
                             epsilon=0.10, variant="orderinf", n_cuts=1)


# ------------------------------------------------------------ concept erasure --

class TestConceptEraser:
    def _paired(self, n=30, d=48, seed=0):
        """Synthetic pairs: emb_with = emb_without + concept + scene-specific noise."""
        rng = np.random.default_rng(seed)
        concept = rng.normal(size=d)
        concept /= np.linalg.norm(concept)
        without = rng.normal(size=(n, d))
        with_ = without + 3.0 * concept + 0.4 * rng.normal(size=(n, d))
        return without, with_, concept

    def test_recovers_the_planted_direction(self):
        from concept_erasure import ConceptEraser
        without, with_, concept = self._paired()
        er = ConceptEraser.fit_from_embeddings(without, with_, rank=1)
        align = abs(float(concept @ er.Q[:, 0]))
        assert align > 0.95, align

    def test_erasure_moves_toward_the_untreated_twin(self):
        from concept_erasure import ConceptEraser
        without, with_, _ = self._paired()
        er = ConceptEraser.fit_from_embeddings(without[:20], with_[:20], rank=2)
        cos = lambda u, v: (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
        before = cos(with_[20:], without[20:]).mean()      # held out
        after = cos(er.erase(with_[20:]), without[20:]).mean()
        assert after > before, (before, after)

    def test_is_a_projection_at_strength_one(self):
        from concept_erasure import ConceptEraser
        without, with_, _ = self._paired()
        er = ConceptEraser.fit_from_embeddings(without, with_, rank=2)
        X = np.random.default_rng(1).normal(size=(10, 48))
        once = er.erase(X)
        assert np.allclose(once, er.erase(once), atol=1e-10)   # idempotent
        assert np.allclose(once @ er.Q, 0.0, atol=1e-10)       # orthogonal to the subspace

    def test_strength_zero_is_identity(self):
        from concept_erasure import ConceptEraser
        without, with_, _ = self._paired()
        er = ConceptEraser.fit_from_embeddings(without, with_, rank=2, strength=0.0)
        X = np.random.default_rng(2).normal(size=(5, 48))
        assert np.allclose(er.erase(X), X)

    def test_sequence_erasure_preserves_bos_and_padding(self):
        from concept_erasure import ConceptEraser
        without, with_, _ = self._paired(d=48)
        er = ConceptEraser.fit_from_embeddings(without, with_, rank=2)
        seq = torch.tensor(np.random.default_rng(3).normal(size=(2, 12, 48)))
        out = er.erase_sequence(seq, eos_idx=[5, 7])
        assert torch.allclose(out[:, 0], seq[:, 0])            # BOS untouched
        assert torch.allclose(out[0, 6:], seq[0, 6:])          # padding untouched
        assert torch.allclose(out[1, 8:], seq[1, 8:])
        assert not torch.allclose(out[0, 1:6], seq[0, 1:6])    # content edited
