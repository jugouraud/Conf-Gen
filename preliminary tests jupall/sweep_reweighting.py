"""Which task-space reweighting scheme should the adaptive scores use?

The manuscript's default mass scheme is inv_power (w_i ~ 1/d_i, adaptation.tex).
That was calibrated on a linear-regression task space where dataset distances
vary over orders of magnitude. CLIP content-token clouds do not: every prompt
sits ~31 +/- 4 away from every other in task-space W1, so an inverse-power law
is numerically indistinguishable from uniform mass and the adaptation does
nothing. This sweep measures that directly and finds the schemes that survive.

Protocol, repeated over N_SPLITS random splits of the 120 safe prompts:
    reference  M = 60   the anchors the score is computed against
    calibration K = 40  sets the conformal radius (K >= ceil(1/eps)-1 = 19)
    test        20      held-out safe prompts; coverage must stay >= 1-eps

Reported per scheme:
    coverage  P[safe test prompt accepted]   -- must be >= 1-eps to be valid
    degen     fraction of DEGENERATE blocked -- the easy tier
    subtle    fraction of SUBTLE blocked     -- the tier that discriminates
    AUROC     safe-test vs subtle, threshold-free
    eff.supp  1/sum(w^2), how many anchors actually carry mass

    python sweep_reweighting.py
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.spatial.distance import cdist

import data_cache as dc
from wasserstein import conformal_radius, score_orderinf, effective_support, whitening_map

EPS = 0.05
M_REF, K_CAL = 60, 40
N_SPLITS = 200
M_PCA = 32          # PCA cutoff of the whitening covariance (matches ecf.py)
SEED = 0

# name -> (scheme, kwargs)
SCHEMES = [
    ("uniform",        "uniform",     {}),
    ("inv^1",          "inv_power",   dict(rescaling_order=1.0)),
    ("inv^2",          "inv_power",   dict(rescaling_order=2.0)),
    ("inv^4",          "inv_power",   dict(rescaling_order=4.0)),
    ("inv^8",          "inv_power",   dict(rescaling_order=8.0)),
    ("softmax r=1",    "softmax",     dict(rescaling_order=1.0)),
    ("softmax r=5",    "softmax",     dict(rescaling_order=5.0)),
    ("softmax r=10",   "softmax",     dict(rescaling_order=10.0)),
    ("softmax r=20",   "softmax",     dict(rescaling_order=20.0)),
    ("softmax r=40",   "softmax",     dict(rescaling_order=40.0)),
    ("softmax r=80",   "softmax",     dict(rescaling_order=80.0)),
    ("knn-3",          "knn",         dict(knn=3, rescaling_order=1.0)),
    ("knn-5",          "knn",         dict(knn=5, rescaling_order=1.0)),
    ("knn-10",         "knn",         dict(knn=10, rescaling_order=1.0)),
    ("knn-20",         "knn",         dict(knn=20, rescaling_order=1.0)),
    ("knn-40",         "knn",         dict(knn=40, rescaling_order=1.0)),
    ("knnU-10",        "knn_uniform", dict(knn=10)),
    ("top1",           "top1",        {}),
]


def masses(dists, scheme, rescaling_order=1.0, knn=None):
    """Vectorised mass map: (n_cand, M) task distances -> (n_cand, M) simplex rows.

    Same schemes as wasserstein.Reweighter, batched over candidates. The
    softmax bandwidth is each candidate's OWN median distance, which makes the
    scheme scale-free -- the property that matters when all distances sit in a
    narrow band.
    """
    d = np.asarray(dists, dtype=float) + 1e-10
    if scheme == "uniform":
        s = np.ones_like(d)
    elif scheme == "inv_power":
        s = (1.0 / d) ** rescaling_order
    elif scheme == "softmax":
        tau = np.median(d, axis=1, keepdims=True)
        s = np.exp(-rescaling_order * d / np.maximum(tau, 1e-12))
    elif scheme in ("knn", "knn_uniform"):
        k = int(min(knn, d.shape[1]))
        idx = np.argsort(d, axis=1)[:, :k]
        s = np.zeros_like(d)
        rows = np.arange(len(d))[:, None]
        s[rows, idx] = 1.0 if scheme == "knn_uniform" else (1.0 / d[rows, idx]) ** rescaling_order
    elif scheme == "top1":
        s = np.zeros_like(d)
        s[np.arange(len(d)), np.argmin(d, axis=1)] = 1.0
    else:
        raise ValueError(scheme)
    tot = s.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(tot)) | (tot <= 0)
    if bad.any():                                   # coincident task overflows inv_power
        for i in np.flatnonzero(bad.ravel()):
            s[i] = 0.0
            s[i, np.argmin(d[i])] = 1.0
        tot = s.sum(axis=1, keepdims=True)
    return s / tot


def auroc(neg, pos):
    """P[score(pos) > score(neg)] with ties at 0.5 -- higher = better separation."""
    neg, pos = np.asarray(neg), np.asarray(pos)
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    allv = np.concatenate([neg, pos])
    r = np.argsort(np.argsort(allv)) + 1.0
    # average ranks over ties
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    rp = r[len(neg):].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def main():
    theta, labels, Dtask, prompts = dc.load()
    m = dc.masks(labels)
    safe_idx = np.flatnonzero(m["safe"])

    tiers = {t: np.flatnonzero(m[t]) for t in dc.TIERS if t != "safe"}
    rng = np.random.default_rng(SEED)

    acc = {n: {k: [] for k in ["cov", "degen", "subtle", "border", "auroc", "eff", "inf"]}
           for n, _, _ in SCHEMES}

    for _ in range(N_SPLITS):
        perm = rng.permutation(len(safe_idx))
        ref_l, cal_l, tst_l = perm[:M_REF], perm[M_REF:M_REF + K_CAL], perm[M_REF + K_CAL:]
        ref_g = safe_idx[ref_l]

        # candidate blocks, all scored against the same reference anchors
        cand = {"cal": safe_idx[cal_l], "tst": safe_idx[tst_l], **tiers}
        rows = np.concatenate([cand[k] for k in cand])
        sizes = [len(cand[k]) for k in cand]
        # Weight-space distances in the MAHALANOBIS metric of the reference split.
        # ecf.tex links s_(d,1) under this metric to the order-1 inverse ECF; raw
        # Euclidean distance in CLIP space is dominated by the leading "nature
        # prompt" directions and discriminates far worse. Fitted on ref only, so
        # the score stays a fixed function of the fit split and CP stays valid.
        wm = whitening_map(theta[ref_g], m=M_PCA)
        Dw = cdist(wm(theta[rows]), wm(theta[ref_g]), metric="euclidean")
        Dt = Dtask[rows][:, ref_l]        # task space  (n, M)

        for name, scheme, kw in SCHEMES:
            W = masses(Dt, scheme, **kw)
            sc = (Dw * W).sum(axis=1) / (M_REF + 1)        # g_{d,1}, eq:adaptive_W1
            parts = dict(zip(cand.keys(), np.split(sc, np.cumsum(sizes)[:-1])))
            r = conformal_radius(parts["cal"], EPS)
            a = acc[name]
            a["inf"].append(not np.isfinite(r))
            a["cov"].append(np.mean(parts["tst"] <= r))
            a["degen"].append(np.mean(parts["degenerate"] > r))
            a["subtle"].append(np.mean(parts["subtle"] > r))
            a["border"].append(np.mean(parts["borderline"] > r))
            a["auroc"].append(auroc(parts["tst"], parts["subtle"]))
            a["eff"].append(effective_support(W[0]))

    # Exact expected coverage of split CP at this K: ceil((K+1)(1-eps))/(K+1).
    exact = np.ceil((K_CAL + 1) * (1 - EPS)) / (K_CAL + 1)
    se = np.std(acc["uniform"]["cov"]) / np.sqrt(N_SPLITS)
    print(f"\nADAPTIVE ORDER-1  g_(d,1)  [Mahalanobis metric]   eps={EPS}  "
          f"M={M_REF} ref / K={K_CAL} cal / {len(safe_idx)-M_REF-K_CAL} test   "
          f"{N_SPLITS} splits")
    print(f"target coverage {1-EPS:.3f};  exact split-CP expectation "
          f"ceil((K+1)(1-eps))/(K+1) = {exact:.4f};  MC s.e. ~{se:.4f}")
    print(f"{'scheme':<14}{'coverage':>9}{'degen':>8}{'subtle':>8}{'border':>8}"
          f"{'AUROC':>8}{'eff.supp':>10}{'vs unif':>10}")
    print("-" * 76)
    base_sub = np.array(acc["uniform"]["subtle"])
    for name, _, _ in SCHEMES:
        a = acc[name]
        cov = np.mean(a["cov"])
        sub_v = np.array(a["subtle"])
        d = sub_v - base_sub                       # paired over shared splits
        if name == "uniform":
            tag = "  (base)"
        else:
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
            tag = f"{d.mean():+.3f}{'*' if abs(t) > 2 else ' '}"
        flag = "" if cov >= exact - 2 * se else "  <-under"
        print(f"{name:<14}{cov:>9.3f}{np.mean(a['degen']):>8.3f}{sub_v.mean():>8.3f}"
              f"{np.mean(a['border']):>8.3f}{np.mean(a['auroc']):>8.3f}"
              f"{np.mean(a['eff']):>10.1f}{tag:>10}{flag}")
    print("-" * 76)
    print("* = paired difference vs uniform mass exceeds 2 s.e. over shared splits")
    best = max((n for n, _, _ in SCHEMES), key=lambda n: np.mean(acc[n]["subtle"]))
    print(f"best subtle-detection: {best}  "
          f"({np.mean(acc[best]['subtle']):.3f} vs {base_sub.mean():.3f} non-adaptive)")
    return acc


if __name__ == "__main__":
    main()
