"""Tests for the §15.5 direction-1 and direction-2 implementations."""

import numpy as np
import pytest
import torch

from run_ordered_experiment import augment
from run_corpus_experiment import gen_compositional, gen_vocab_grounded, d_eff
from run_residual_experiment import Basis
from ecf import fit_regularized_covariance, mahalanobis_sq


class TestPositionalAugmentation:
    def test_lambda_zero_leaves_clouds_unchanged_up_to_the_zero_column(self):
        cl = [np.random.default_rng(i).normal(size=(5 + i, 4)) for i in range(3)]
        out = augment(cl, 0.0, "normalised")
        for a, b in zip(cl, out):
            assert np.allclose(b[:, :-1], a) and np.allclose(b[:, -1], 0.0)

    def test_normalised_position_spans_zero_to_one(self):
        cl = [np.zeros((7, 3))]
        out = augment(cl, 1.0, "normalised")[0]
        assert out[0, -1] == 0.0 and np.isclose(out[-1, -1], 1.0)

    def test_absolute_position_is_length_dependent(self):
        short = augment([np.zeros((3, 2))], 1.0, "absolute")[0]
        long = augment([np.zeros((9, 2))], 1.0, "absolute")[0]
        assert short[-1, -1] == 2.0 and long[-1, -1] == 8.0

    def test_single_token_cloud_is_handled(self):
        out = augment([np.zeros((1, 3))], 1.0, "normalised")[0]
        assert out.shape == (1, 4) and np.isfinite(out).all()

    def test_lambda_scales_the_positional_cost(self):
        cl = [np.zeros((5, 2))]
        a = augment(cl, 1.0, "normalised")[0]
        b = augment(cl, 3.0, "normalised")[0]
        assert np.allclose(b[:, -1], 3.0 * a[:, -1])


class TestCorpusGenerators:
    def test_compositional_produces_distinct_prompts(self):
        g = gen_compositional(300, seed=0)
        assert len(g) == 300 and len(set(g)) == 300

    def test_compositional_varies_in_length(self):
        g = gen_compositional(300, seed=1)
        lens = {len(p.split()) for p in g}
        assert len(lens) >= 6, lens

    def test_length_matching_respects_the_target(self):
        target = {6, 7, 8}
        g = gen_compositional(120, seed=2, target_lengths=target)
        assert {len(p.split()) for p in g} <= target

    def test_vocab_grounded_reuses_source_words(self):
        src = ["a misty forest at dawn with tall pines",
               "a calm lake under low cloud and rain",
               "a rocky shore beneath a pale winter sun"]
        g = gen_vocab_grounded(50, src, seed=0)
        src_words = set(" ".join(src).split())
        assert all(set(p.split()) <= src_words for p in g)

    def test_generators_are_deterministic_given_a_seed(self):
        assert gen_compositional(50, seed=7) == gen_compositional(50, seed=7)


class TestIntrinsicDimension:
    def test_recovers_a_planted_dimension(self):
        rng = np.random.default_rng(0)
        for k in (2, 5, 10):
            X = rng.normal(size=(600, k)) @ rng.normal(size=(k, 40))
            est = d_eff(X)
            assert 0.5 * k <= est <= 2.5 * k, (k, est)

    def test_is_larger_for_a_richer_cloud(self):
        rng = np.random.default_rng(1)
        low = rng.normal(size=(300, 3)) @ rng.normal(size=(3, 30))
        high = rng.normal(size=(300, 20)) @ rng.normal(size=(20, 30))
        assert d_eff(high) > d_eff(low)


class TestBasisDecomposition:
    def test_A_plus_B_equals_the_mahalanobis_distance(self):
        """The decomposition §14.1 rests on: d_M^2 = A + B, exactly."""
        rng = np.random.default_rng(0)
        fit = rng.normal(size=(80, 60)) @ np.diag(np.linspace(4, .3, 60))
        B_ = Basis(fit, 16)
        Z = rng.normal(size=(25, 60)) * 2
        A, Bt, cw, pw = B_.parts(Z)
        mean, V, er, ei, g = fit_regularized_covariance(torch.tensor(fit), m=16)
        want = mahalanobis_sq(torch.tensor(Z), mean, V, ei, float(g)).numpy()
        assert np.allclose(A + Bt, want, rtol=1e-9), np.abs(A + Bt - want).max()

    def test_whitened_parts_reproduce_the_terms(self):
        rng = np.random.default_rng(2)
        fit = rng.normal(size=(70, 40))
        B_ = Basis(fit, 12)
        A, Bt, cw, pw = B_.parts(rng.normal(size=(15, 40)))
        assert np.allclose((cw ** 2).sum(1), A)
        assert np.allclose((pw ** 2).sum(1), Bt)

    def test_residual_is_orthogonal_to_the_retained_subspace(self):
        rng = np.random.default_rng(3)
        fit = rng.normal(size=(70, 40))
        B_ = Basis(fit, 12)
        _, _, _, pw = B_.parts(rng.normal(size=(10, 40)))
        assert np.allclose(pw @ B_.V, 0.0, atol=1e-9)
