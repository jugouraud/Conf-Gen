"""Tests for the deployable conformal projection guard."""

import numpy as np
import pytest
import torch

from conformal_guard import (ConformalGuard, pooled_from, content_rows,
                             LIFTS, SCORES, attach)


def synthetic(N=80, L=12, d=24, seed=0):
    """Sequences with a shared 'domain' mean plus per-prompt variation."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=d) * 3
    seq = base + rng.normal(size=(N, L, d)) * 0.6
    eos = rng.integers(4, L - 1, size=N)
    return seq, eos


class TestExtraction:
    def test_pooled_eos_picks_the_eos_row(self):
        seq, eos = synthetic()
        p = pooled_from(seq, eos, "eos")
        assert np.allclose(p, seq[np.arange(len(seq)), eos])

    def test_pooled_mean_averages_content(self):
        seq, eos = synthetic()
        p = pooled_from(seq, eos, "mean")
        assert np.allclose(p[0], seq[0, 1:eos[0]].mean(axis=0))

    def test_content_rows_excludes_bos_and_padding(self):
        seq, eos = synthetic()
        blocks = content_rows(seq, eos)
        for b, e in zip(blocks, eos):
            assert len(b) == max(int(e) - 1, 1)


class TestGuard:
    @pytest.mark.parametrize("lift", LIFTS)
    @pytest.mark.parametrize("score", SCORES)
    def test_post_condition_everything_ends_inside(self, score, lift):
        """The defining property of a projection filter."""
        seq, eos = synthetic()
        g = ConformalGuard.fit(seq[:60], eos[:60], epsilon=0.1, score=score,
                               lift=lift, n_ref=30, lowcard_K=8)
        rng = np.random.default_rng(7)
        out_seq = seq[60:] + rng.normal(size=seq[60:].shape) * 4     # push outside
        new, info = g.guard(out_seq, eos[60:])
        after = np.atleast_1d(info["score_after"])
        assert np.all(after <= g.region.radius + 1e-6), (score, lift, after.max())

    @pytest.mark.parametrize("lift", LIFTS)
    def test_inside_prompts_are_untouched(self, lift):
        seq, eos = synthetic()
        g = ConformalGuard.fit(seq[:60], eos[:60], epsilon=0.1, score="orderinf",
                               lift=lift, n_ref=30)
        new, info = g.guard(seq[60:], eos[60:])
        inside = ~np.atleast_1d(info["was_outside"])
        assert inside.any()
        assert np.allclose(new[inside], seq[60:][inside], atol=1e-8)

    def test_pooled_lift_is_exact(self):
        """Translating every row by delta shifts the pooled vector by exactly delta."""
        seq, eos = synthetic()
        g = ConformalGuard.fit(seq[:60], eos[:60], epsilon=0.1, score="orderinf",
                               lift="pooled", n_ref=30)
        rng = np.random.default_rng(3)
        z = seq[60:] + rng.normal(size=seq[60:].shape) * 5
        new, info = g.guard(z, eos[60:])
        # the projected pooled vector must be exactly the region's projection
        pw = g.whiten(pooled_from(new, eos[60:], "eos"))
        assert g.region.contains(pw).all()
        # and the edit is a rigid translation: all rows moved by the same vector
        delta = new - z
        for b in range(len(delta)):
            assert np.allclose(delta[b], delta[b, 0][None, :], atol=1e-9)

    def test_torch_in_torch_out(self):
        seq, eos = synthetic()
        g = ConformalGuard.fit(seq[:60], eos[:60], epsilon=0.1, lift="pooled", n_ref=30)
        t = torch.tensor(seq[60:65], dtype=torch.float32)
        new, _ = g.guard(t, eos[60:65])
        assert torch.is_tensor(new) and new.dtype == torch.float32
        assert new.shape == t.shape

    def test_rejects_bad_arguments(self):
        seq, eos = synthetic()
        with pytest.raises(ValueError):
            ConformalGuard.fit(seq, eos, lift="nope")
        with pytest.raises(ValueError):
            ConformalGuard.fit(seq, eos, score="nope")

    def test_adaptive_region_is_rebuilt_per_prompt(self):
        seq, eos = synthetic()
        g = ConformalGuard.fit(seq[:60], eos[:60], epsilon=0.1, score="orderinf",
                               lift="pooled", n_ref=30, adaptive_K=4)
        M = len(g.anchors_w)
        r1 = g._region_for(np.arange(M))            # anchors 0..3 nearest
        r2 = g._region_for(np.arange(M)[::-1])      # anchors M-1..M-4 nearest
        assert r1.centers.shape[0] == 4
        assert not np.allclose(r1.centers, r2.centers)


class TestAttach:
    def test_attach_and_detach_restore_the_pipeline(self):
        class FakePipe:
            def __init__(self):
                self.tokenizer = None
                self.calls = 0
            def encode_prompt(self, *a, **k):
                self.calls += 1
                return "pe", "ne"
        seq, eos = synthetic()
        g = ConformalGuard.fit(seq[:60], eos[:60], epsilon=0.1, lift="pooled", n_ref=30)
        pipe = FakePipe()
        detach = attach(pipe, g)
        assert pipe._conformal_original_encode is not None
        with pytest.raises(RuntimeError):
            attach(pipe, g)                       # double attach is refused
        detach()
        # bound methods are re-created on every attribute access, so compare
        # behaviour rather than identity: the wrapper is gone
        assert pipe._conformal_original_encode is None
        assert pipe.encode_prompt("p", None, 1, False) == ("pe", "ne")


class TestRepairInGuard:
    def test_repair_t_is_wired_through_every_lift(self):
        seq, eos = synthetic()
        rng = np.random.default_rng(11)
        z = seq[60:] + rng.normal(size=seq[60:].shape) * 4
        for lift in LIFTS:
            g = ConformalGuard.fit(seq[:60], eos[:60], epsilon=0.1,
                                   score="orderinf", lift=lift, n_ref=30)
            moved = []
            for t in [0.0, 0.5, 1.0]:
                g.repair_t = t
                new, info = g.guard(z, eos[60:])
                after = np.atleast_1d(info["score_after"])
                assert np.all(after <= g.region.radius + 1e-6), (lift, t)
                moved.append(info["displacement"].mean())
            assert moved[0] < moved[1] < moved[2], (lift, moved)

    def test_sequence_lift_at_t_one_lands_on_a_real_anchor(self):
        """The property that makes repair semantically meaningful."""
        seq, eos = synthetic(N=80, L=10, d=16)
        g = ConformalGuard.fit(seq[:60], eos[:60], epsilon=0.1, score="orderinf",
                               lift="sequence", n_ref=40, seed=0)
        ref_i = np.random.default_rng(0).permutation(60)[:40]
        g.repair_t = 1.0
        rng = np.random.default_rng(12)
        z = seq[60:70] + rng.normal(size=seq[60:70].shape) * 5
        out, _ = g.guard(z, eos[60:70])
        R = seq[:60][ref_i].reshape(40, -1)
        d = np.linalg.norm(out.reshape(len(out), -1)[:, None, :] - R[None], axis=2)
        assert d.min(axis=1).max() < 1e-6, d.min(axis=1).max()
