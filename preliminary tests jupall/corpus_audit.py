"""Validate a conformal safe corpus against a holdout from a DIFFERENT source.

WHY THIS EXISTS
---------------
Split conformal prediction guarantees

    P( s(Z_new) <= eps ) >= 1 - beta

whenever Z_new is exchangeable with the calibration set. It says nothing about
whether the calibration set resembles what the filter will actually see. If the
corpus is unrepresentative, coverage is a valid number about the wrong
distribution -- and it gives no warning at all.

Measured in this project (README section 11): a safe set calibrated on 1000
machine-generated nature prompts reported 0.95 coverage on its own held-out
prompts while rejecting 0.56-0.72 of genuine hand-written nature prompts. Every
diagnostic internal to the calibration set looked perfect.

The fix is one extra holdout, drawn from a DIFFERENT source than the calibration
data, scored with the same score function. Under exchangeability its scores are
exchangeable with the calibration scores; if the corpus is unrepresentative they
are stochastically larger. That is a testable statement, and this module tests
it exactly.

WHAT IS REPORTED
----------------
    frr             fraction of the audit holdout wrongly rejected. Under
                    exchangeability its expectation is <= beta.
    p_value         EXACT permutation test. Pool calibration and audit scores,
                    re-split at random into the same sizes, recompute the radius
                    from the synthetic calibration half and the rejection rate on
                    the synthetic audit half. This is the conditional null
                    distribution of `frr` under exchangeability, so no
                    independence assumption is needed -- audit p-values are
                    dependent (they share one calibration set), which makes a
                    plain binomial test wrong here.
    radius_factor   how much larger the radius would have to be to cover the
                    audit source at 1-beta. 1.0 means the corpus is fine.
    beta_needed     the largest nominal beta that, on THIS corpus, would still
                    deliver 1-beta coverage on the audit source. 0 means no
                    non-vacuous level suffices -- even the largest calibration
                    score falls short -- so the corpus, not the level, is wrong.
    cal_span        max(calibration score) / radius. Read it against
                    radius_factor: when radius_factor > cal_span the calibration
                    distribution is so narrow that none of its order statistics
                    reaches the audit source at all.
    verdict         "pass" / "warn" / "fail"

USE
---
    from corpus_audit import audit_scores
    report = audit_scores(cal_scores, holdout_scores, beta=0.05)
    print(report)
    if report.failed:
        ...                       # do not ship this corpus

Run it BEFORE comparing estimators. An unrepresentative corpus changes every
downstream number -- in section 11 it moved subtle-prompt detection from 0.41 to
0.95, none of which was real.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from wasserstein import conformal_radius


@dataclass
class CorpusAudit:
    """Result of validating a calibrated safe set against an external holdout."""
    beta: float
    n_cal: int
    n_audit: int
    radius: float
    frr: float                       # observed false-rejection rate on the holdout
    frr_lo: float                    # Wilson interval, conditional on the calibration set
    frr_hi: float
    p_value: float                   # exact permutation test
    null_frr: float                  # mean rejection rate under the exchangeable null
    radius_factor: float
    cal_span: float
    beta_needed: float
    shift: str                       # "none" | "audit harder" | "audit easier"
    verdict: str
    n_perm: int = 0

    @property
    def failed(self) -> bool:
        return self.verdict == "fail"

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def __str__(self):
        mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[self.verdict]
        L = [
            f"corpus audit [{mark}]  beta={self.beta:g}  "
            f"n_cal={self.n_cal}  n_audit={self.n_audit}",
            f"  false rejection on the external holdout : {self.frr:.3f} "
            f"[{self.frr_lo:.3f}, {self.frr_hi:.3f}]   (target <= {self.beta:g})",
            f"  exchangeability permutation test        : p = {self.p_value:.4g}   "
            f"(null rejection rate {self.null_frr:.3f}, {self.n_perm} permutations)",
            f"  direction of shift                      : {self.shift}",
            f"  radius needed to cover the holdout      : {self.radius_factor:.2f}x "
            f"the calibrated radius   (the ENTIRE calibration range reaches only "
            f"{self.cal_span:.2f}x)",
        ]
        if self.beta_needed > 0:
            L.append(f"  nominal beta that would deliver 1-{self.beta:g} there : "
                     f"{self.beta_needed:.4g}")
        else:
            L.append("  nominal beta that would deliver it      : none — no level on this "
                     "corpus covers the holdout; the CORPUS is the problem")
        if self.verdict != "pass":
            L.append("  -> the calibration corpus is not representative of the holdout "
                     "source.")
            L.append("     Coverage on held-out CALIBRATION data is still valid and will "
                     "still look fine;")
            L.append("     it is simply a statement about the wrong distribution. Fix the "
                     "corpus, not the level.")
        return "\n".join(L)


def _wilson(k, n, z=1.96):
    """Wilson score interval; conditional on the calibration set, so approximate."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def _beta_needed(cal, target_radius):
    """Largest nominal beta whose radius on `cal` still reaches `target_radius`.

    conformal_radius is non-increasing in beta, so a smaller beta gives a larger
    radius and the answer is the least conservative level that works. Returns
    0.0 when the largest calibration score is below the target: only a vacuous
    (+inf) radius covers the audit source, which means the corpus is wrong.
    """
    s = np.sort(np.asarray(cal, dtype=float))
    K = len(s)
    if s[-1] < target_radius:
        return 0.0
    # radius at level beta is s[ceil((K+1)(1-beta)) - 1]; find the smallest rank
    # whose score clears the target, then invert.
    rank = int(np.searchsorted(s, target_radius, side="left")) + 1
    rank = min(max(rank, 1), K)
    return float(1.0 - (rank / (K + 1)))


def audit_scores(cal_scores, audit_scores_, beta=0.05, n_perm=5000,
                 warn_ratio=1.5, fail_ratio=2.0, alpha_test=0.01, alpha_warn=0.05,
                 seed=0):
    """Validate a calibrated safe set against scores from an external holdout.

    cal_scores    : (K,) nonconformity scores of the CALIBRATION set
    audit_scores_ : (n,) scores of a holdout drawn from a DIFFERENT source,
                    believed to be genuinely in-domain
    beta          : the nominal miscoverage the safe set was calibrated at

    Both must come from the same score function and the same reference/anchor
    set; only the source of the prompts differs.

    Verdict thresholds. "fail" needs the shift to be both statistically clear
    (p < alpha_test) and practically large (frr > fail_ratio * beta); "warn"
    needs either p < alpha_warn or frr > warn_ratio * beta. All four are
    exposed so the audit can be tuned to how costly a false rejection is.

    NOTE the permutation test is CONSERVATIVE. `frr` takes values k/n_audit, so
    the permutation null is discrete and heavily tied, and resolving ties in
    favour of the null inflates p. Measured on exchangeable data with n_audit =
    100 and beta = 0.1: the test fires at 0.027 against a nominal 0.05. It
    under-detects rather than over-detects, which is the right direction for a
    check whose alarm means "throw away the corpus".
    """
    cal = np.asarray(cal_scores, dtype=float).ravel()
    aud = np.asarray(audit_scores_, dtype=float).ravel()
    K, n = len(cal), len(aud)
    if K == 0 or n == 0:
        raise ValueError("both the calibration and the audit set must be non-empty")

    radius = conformal_radius(cal, beta)
    rejected = int(np.sum(aud > radius))
    frr = rejected / n
    lo, hi = _wilson(rejected, n)

    # --- exact permutation test of exchangeability ---
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([cal, aud])
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(pooled)
        r = conformal_radius(perm[:K], beta)
        null[b] = np.mean(perm[K:] > r)
    p_value = (1.0 + np.sum(null >= frr)) / (n_perm + 1.0)

    # --- effect size: how far off is the radius? ---
    target = float(np.quantile(aud, 1.0 - beta))
    ok = np.isfinite(radius) and radius > 0
    radius_factor = target / radius if ok else np.inf
    cal_span = float(cal.max() / radius) if ok else np.inf
    beta_need = _beta_needed(cal, target)

    if frr > beta and p_value < 0.05:
        shift = "audit harder (calibration corpus is too narrow)"
    elif frr < beta and (1.0 + np.sum(null <= frr)) / (n_perm + 1.0) < 0.05:
        shift = "audit easier (calibration corpus is broader than the holdout)"
    else:
        shift = "none detected"

    if p_value < alpha_test and frr > fail_ratio * beta:
        verdict = "fail"
    elif p_value < alpha_warn or frr > warn_ratio * beta:
        verdict = "warn"
    else:
        verdict = "pass"

    return CorpusAudit(beta=beta, n_cal=K, n_audit=n, radius=radius, frr=frr,
                       frr_lo=lo, frr_hi=hi, p_value=float(p_value),
                       null_frr=float(null.mean()), radius_factor=float(radius_factor),
                       cal_span=cal_span, beta_needed=beta_need, shift=shift,
                       verdict=verdict, n_perm=n_perm)


def audit_guard(guard, cal_seq, cal_eos, audit_seq, audit_eos, **kw):
    """Convenience: audit a fitted ConformalGuard against an external holdout.

    Scores both sets with the guard's own score function and reference set, then
    defers to audit_scores. `audit_seq` should be sequences of prompts from a
    different source that are believed to be in-domain.
    """
    def score(seq, eos):
        _, info = guard.guard(np.asarray(seq, dtype=float), eos)
        return np.atleast_1d(info["score_before"])

    return audit_scores(score(cal_seq, cal_eos), score(audit_seq, audit_eos),
                        beta=guard.epsilon, **kw)
