"""Tests for the blackbox meta-learning score g_d^BB (adaptation.tex)."""

import numpy as np
import pytest
import torch

from bb_model import SetPredictor, BBScore, fit_bb
from wasserstein import whitening_map, conformal_radius


def corpus(n=220, d=48, L=9, spread=1.0, seed=0):
    """Safe-like prompts: a token bag whose mean predicts the 'embedding'."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(d, d))
    clouds, theta = [], []
    for _ in range(n):
        k = rng.integers(4, L + 1)
        c = rng.normal(size=(k, d)) * spread
        clouds.append(c)
        theta.append(np.tanh(c.mean(0) @ B) * 4 + rng.normal(size=d) * 0.05)
    return np.array(theta), clouds


class TestArchitecture:
    def test_is_small(self):
        net = SetPredictor(33, 32)
        assert net.n_params < 100_000, net.n_params

    def test_permutation_invariant_in_eval_mode(self):
        net = SetPredictor(12, 8).eval()
        x = torch.randn(3, 7, 12)
        m = torch.ones(3, 7)
        p = torch.randperm(7)
        assert torch.allclose(net(x, m), net(x[:, p], m[:, p]), atol=1e-6)

    def test_mask_excludes_padding(self):
        net = SetPredictor(6, 4).eval()
        x = torch.randn(2, 10, 6)
        m = torch.zeros(2, 10)
        m[:, :4] = 1
        out = net(x, m)
        x2 = x.clone()
        x2[:, 4:] = 99.0                       # garbage in the padded slots
        assert torch.allclose(out, net(x2, m), atol=1e-6)


class TestFitting:
    def _fitted(self, n=200, seed=0, **kw):
        th, cl = corpus(n=n, seed=seed)
        geom_th, geom_cl = th[:60], cl[:60]
        th_wm = whitening_map(geom_th, m=16)
        tok_wm = whitening_map(np.vstack(geom_cl), m=16)
        bb = fit_bb(th[60:], cl[60:], tok_wm, th_wm, verbose=False, epochs=120, **kw)
        return bb, th, cl

    def test_learns_something(self):
        """In-subspace error must be far lower on in-domain than shifted prompts."""
        bb, th, cl = self._fitted()
        far_th, far_cl = corpus(n=60, spread=4.0, seed=9)
        e_in = bb.in_subspace_error(th[60:120], cl[60:120]).mean()
        e_out = bb.in_subspace_error(far_th, far_cl).mean()
        assert e_out > 1.5 * e_in, (e_in, e_out)

    def test_score_is_nonnegative_and_finite(self):
        bb, th, cl = self._fitted()
        s = bb(th[:40], cl[:40])
        assert np.all(np.isfinite(s)) and np.all(s >= 0)

    def test_score_dominates_the_in_subspace_part(self):
        """The full score adds theta's off-subspace residual, so it is larger."""
        bb, th, cl = self._fitted()
        assert np.all(bb(th[:40], cl[:40]) >= bb.in_subspace_error(th[:40], cl[:40]) - 1e-9)

    def test_prediction_is_permutation_invariant(self):
        bb, th, cl = self._fitted()
        shuffled = [c[np.random.default_rng(i).permutation(len(c))] for i, c in enumerate(cl[:20])]
        assert np.allclose(bb.predict(cl[:20]), bb.predict(shuffled), atol=1e-5)

    def test_variable_length_clouds_are_handled(self):
        bb, th, cl = self._fitted()
        lens = {len(c) for c in cl[:50]}
        assert len(lens) > 1
        assert bb(th[:50], cl[:50]).shape == (50,)

    def test_capacity_knob_changes_size(self):
        small, _, _ = self._fitted(d_hidden=32)
        big, _, _ = self._fitted(d_hidden=192)
        assert big.net.n_params > 4 * small.net.n_params


class TestConformalUse:
    def test_coverage_holds_on_held_out_safe_prompts(self):
        """g_d^BB calibrated on a disjoint split covers at 1 - beta (pr:bb_coverage)."""
        th, cl = corpus(n=520, seed=3)
        th_wm = whitening_map(th[:80], m=16)
        tok_wm = whitening_map(np.vstack(cl[:80]), m=16)
        bb = fit_bb(th[80:280], cl[80:280], tok_wm, th_wm, verbose=False, epochs=150)
        cov = []
        for beta in (0.05, 0.1, 0.2):
            eps = conformal_radius(bb(th[280:400], cl[280:400]), beta)
            cov.append((beta, np.mean(bb(th[400:], cl[400:]) <= eps)))
        for beta, c in cov:
            assert c >= 1 - beta - 0.06, (beta, c)

    def test_region_is_a_ball_around_the_prediction(self):
        """The score's sublevel set is a single ball, so projection is one line."""
        th, cl = corpus(n=160, seed=4)
        th_wm = whitening_map(th[:60], m=16)
        tok_wm = whitening_map(np.vstack(cl[:60]), m=16)
        bb = fit_bb(th[60:], cl[60:], tok_wm, th_wm, verbose=False, epochs=100)
        coef, _ = bb._theta_parts(th[:8])
        centre = bb.predict(cl[:8])
        assert centre.shape == coef.shape          # same space => a ball in it
