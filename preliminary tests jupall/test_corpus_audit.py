"""Tests for the external-holdout corpus audit.

The audit must (a) fire when the calibration corpus is unrepresentative,
(b) stay silent when it is not, and (c) have a correctly calibrated null --
a test that fires at the wrong rate is worse than no test.
"""

import numpy as np
import pytest
import warnings

from corpus_audit import audit_scores, CorpusAudit, _beta_needed
from wasserstein import conformal_radius


def scores(n, loc=0.0, scale=1.0, seed=0):
    """Nonconformity-like scores: non-negative, right-skewed."""
    return np.abs(np.random.default_rng(seed).normal(loc, scale, n)) + loc


class TestVerdicts:
    def test_same_distribution_passes(self):
        r = audit_scores(scores(400, seed=1), scores(150, seed=2), beta=0.05, n_perm=2000)
        assert r.passed, str(r)
        assert r.shift == "none detected"

    def test_shifted_audit_fails(self):
        """Audit drawn from a wider distribution: the corpus is too narrow."""
        r = audit_scores(scores(400, scale=1.0, seed=1),
                         scores(150, scale=2.2, seed=2), beta=0.05, n_perm=2000)
        assert r.failed, str(r)
        assert "too narrow" in r.shift
        assert r.frr > 0.10 and r.p_value < 0.01

    def test_narrower_audit_reports_the_other_direction(self):
        r = audit_scores(scores(400, scale=2.0, seed=1),
                         scores(200, scale=0.6, seed=2), beta=0.20, n_perm=2000)
        assert "easier" in r.shift, r.shift
        assert not r.failed

    def test_verdict_thresholds_are_configurable(self):
        cal, aud = scores(400, seed=1), scores(150, scale=1.35, seed=2)
        strict = audit_scores(cal, aud, beta=0.05, n_perm=2000,
                              warn_ratio=1.01, fail_ratio=1.05, alpha_test=0.5)
        lax = audit_scores(cal, aud, beta=0.05, n_perm=2000, warn_ratio=50,
                           fail_ratio=100, alpha_test=1e-12, alpha_warn=1e-12)
        assert strict.verdict == "fail" and lax.verdict == "pass"


class TestNullCalibration:
    def test_permutation_null_rejects_at_beta(self):
        """Under exchangeability the null rejection rate must sit at beta."""
        for beta in (0.05, 0.10, 0.20):
            r = audit_scores(scores(300, seed=3), scores(150, seed=4),
                             beta=beta, n_perm=3000)
            assert abs(r.null_frr - beta) < 0.02, (beta, r.null_frr)

    def test_false_alarm_rate_is_conservative(self):
        """The audit must not cry wolf on exchangeable data.

        frr is discrete (k / n_audit), so the permutation null is heavily tied
        and the p-value is conservative by construction. The requirement is
        therefore one-sided: fire at no MORE than the nominal rate, while still
        being informative (not always returning p = 1).
        """
        rng = np.random.default_rng(0)
        p = np.array([audit_scores(x[:250], x[250:], beta=0.1, n_perm=400, seed=i).p_value
                      for i, x in enumerate(np.abs(rng.normal(size=(250, 350))))])
        assert np.mean(p < 0.05) <= 0.05, np.mean(p < 0.05)       # conservative
        assert np.mean(p < 0.05) > 0.0                            # not degenerate
        assert 0.4 < p.mean() < 0.7, p.mean()


class TestQuantities:
    def test_radius_factor_is_one_when_matched(self):
        r = audit_scores(scores(500, seed=5), scores(300, seed=6), beta=0.1, n_perm=500)
        assert 0.85 < r.radius_factor < 1.15

    def test_radius_factor_exceeds_one_when_shifted(self):
        r = audit_scores(scores(400, seed=5), scores(200, scale=2.0, seed=6),
                         beta=0.05, n_perm=500)
        assert r.radius_factor > 1.3

    def test_beta_needed_recovers_nominal_coverage(self):
        """Calibrating at beta_needed really does cover the audit source."""
        cal, aud = scores(600, seed=7), scores(400, scale=1.25, seed=8)
        r = audit_scores(cal, aud, beta=0.05, n_perm=500)
        assert r.beta_needed > 0
        wider = conformal_radius(cal, r.beta_needed)
        assert np.mean(aud > wider) <= 0.05 + 1e-9

    def test_beta_needed_is_zero_when_unreachable(self):
        cal = np.linspace(1.0, 1.05, 300)          # extremely narrow calibration
        aud = np.linspace(5.0, 6.0, 100)           # entirely beyond it
        r = audit_scores(cal, aud, beta=0.05, n_perm=200)
        assert r.beta_needed == 0.0 and r.radius_factor > r.cal_span

    def test_beta_needed_helper_directly(self):
        s = np.arange(1.0, 101.0)                  # K = 100
        b = _beta_needed(s, 90.0)
        assert conformal_radius(s, b) >= 90.0

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError):
            audit_scores([], [1.0, 2.0])
        with pytest.raises(ValueError):
            audit_scores([1.0, 2.0], [])


class TestGuardIntegration:
    def _seqs(self, n, L=10, d=20, spread=1.0, seed=0):
        rng = np.random.default_rng(seed)
        base = rng.normal(size=d) * 3
        return base + rng.normal(size=(n, L, d)) * spread, rng.integers(4, L - 1, size=n)

    def test_matched_audit_attaches_a_passing_report(self):
        from conformal_guard import ConformalGuard
        seq, eos = self._seqs(160, seed=0)
        g = ConformalGuard.fit(seq[:110], eos[:110], epsilon=0.1, score="order1",
                               lift="pooled", n_ref=60,
                               audit_sequences=seq[110:], audit_eos=eos[110:])
        assert g.audit is not None and not g.audit.failed

    def test_shifted_audit_warns(self):
        from conformal_guard import ConformalGuard
        seq, eos = self._seqs(120, spread=1.0, seed=0)
        bad, bad_eos = self._seqs(80, spread=6.0, seed=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g = ConformalGuard.fit(seq, eos, epsilon=0.05, score="order1",
                                   lift="pooled", n_ref=60,
                                   audit_sequences=bad, audit_eos=bad_eos)
        assert g.audit.failed
        assert any("corpus audit FAILED" in str(x.message) for x in w)

    def test_shifted_audit_can_raise(self):
        from conformal_guard import ConformalGuard, UnrepresentativeCorpus
        seq, eos = self._seqs(120, spread=1.0, seed=0)
        bad, bad_eos = self._seqs(80, spread=6.0, seed=1)
        with pytest.raises(UnrepresentativeCorpus):
            ConformalGuard.fit(seq, eos, epsilon=0.05, score="order1", lift="pooled",
                               n_ref=60, audit_sequences=bad, audit_eos=bad_eos,
                               audit_raises=True)

    def test_no_audit_leaves_the_attribute_none(self):
        from conformal_guard import ConformalGuard
        seq, eos = self._seqs(120, seed=0)
        g = ConformalGuard.fit(seq, eos, epsilon=0.1, lift="pooled", n_ref=60)
        assert g.audit is None
