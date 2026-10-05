"""Tests for the bimodal safe-set experiment (§17)."""

import numpy as np
import pytest

import prompts_multimodal as PM
from run_multimodal_experiment import bottleneck_n, score_orderinf_D, features
from run_residual_experiment import bottleneck
from scipy.spatial.distance import cdist


class TestCorpus:
    def test_modes_are_distinct_and_deduplicated(self):
        a, b = PM.mode_a(200), PM.mode_b(200)
        assert len(set(a)) == 200 and len(set(b)) == 200
        assert not (set(a) & set(b))

    def test_modes_use_disjoint_landform_vocabulary(self):
        assert not (set(PM.A_LAND) & set(PM.B_LAND))
        assert not (set(PM.A_MOD) & set(PM.B_MOD))

    def test_gap_prompts_mix_both_vocabularies(self):
        """Each gap prompt must draw from BOTH modes -- that is what makes it a gap."""
        g = PM.gap(80)
        a_terms = {w for t in PM.A_LAND + PM.A_MOD for w in t.split()}
        b_terms = {w for t in PM.B_LAND + PM.B_MOD for w in t.split()}
        mixed = sum(bool(set(p.split()) & a_terms) and bool(set(p.split()) & b_terms)
                    for p in g)
        assert mixed >= 0.8 * len(g), mixed

    def test_other_types_are_available_and_distinct(self):
        for k in PM.OTHER:
            v = PM.other(k, 40)
            assert len(set(v)) == 40

    def test_generators_are_deterministic(self):
        assert PM.mode_a(50) == PM.mode_a(50)
        assert PM.gap(30) == PM.gap(30)


class TestBottleneck:
    def test_matches_the_shared_implementation_at_n_zero(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(18, 5))
        D = cdist(X, X)
        assert np.isclose(bottleneck_n(D, 0), bottleneck(D))

    def test_is_decreasing_in_n(self):
        rng = np.random.default_rng(1)
        D = cdist(rng.normal(size=(25, 4)), rng.normal(size=(25, 4)))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        v = [bottleneck_n(D, n) for n in range(4)]
        assert all(a >= b for a, b in zip(v, v[1:])), v

    def test_score_matches_a_direct_mst_on_the_augmented_set(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 6))
        ref = np.arange(20)
        Dca = cdist(X, X[ref])
        Daa = Dca[ref]
        rows = np.arange(20, 30)
        got = score_orderinf_D(Dca, Daa, rows, 0)
        for i, r in enumerate(rows):
            Y = np.vstack([X[ref], X[r][None, :]])
            assert np.isclose(got[i], bottleneck(cdist(Y, Y)))


class TestFeatureMaps:
    def _data(self):
        rng = np.random.default_rng(0)
        th = rng.normal(size=(120, 40)) @ np.diag(np.linspace(5, .4, 40))
        return th, np.arange(60)

    @pytest.mark.parametrize("metric", ["full", "subspace", "raw"])
    def test_shapes_and_finiteness(self, metric):
        th, ref = self._data()
        W, A, B = features(th, ref, metric, m=8)
        assert len(W) == len(th) and np.isfinite(W).all()

    def test_subspace_is_narrower_than_full(self):
        """`full` concatenates the m subspace coords with the d-dim residual
        VECTOR: m + d columns, a redundant but isometric parametrisation
        (||cw||^2 + ||pw||^2 = A + B). `subspace` keeps only the m coords."""
        th, ref = self._data()
        Wf, _, _ = features(th, ref, "full", m=8)
        Ws, _, _ = features(th, ref, "subspace", m=8)
        assert Ws.shape[1] == 8
        assert Wf.shape[1] == 8 + th.shape[1]
        assert Ws.shape[1] < Wf.shape[1]

    def test_raw_is_a_pure_translation(self):
        th, ref = self._data()
        Wr, _, _ = features(th, ref, "raw", m=8)
        d0 = cdist(th[:10], th[10:20])
        d1 = cdist(Wr[:10], Wr[10:20])
        assert np.allclose(d0, d1)

    def test_full_metric_equals_A_plus_B(self):
        th, ref = self._data()
        W, A, B = features(th, ref, "full", m=8)
        assert np.allclose((W ** 2).sum(1), A + B)
