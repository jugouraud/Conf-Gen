"""Task-space reweighting schemes, including ones built for a narrow distance band.

adaptation.tex asks for a map from task-space distances d_1..d_M to a probability
vector w on the simplex, with the identity property

    tau = tau_j  =>  w_j = 1,  w_i = 0 for i != j.               (eq:reweight_identity)

The manuscript's default is `inv_power`, w_i ~ d_i^-r. That is calibrated for a
task space where dataset distances vary over orders of magnitude. CLIP token
clouds do not: the p95/p5 ratio of within-corpus W1 distances is 1.26 raw and
1.36 whitened. On such a band an inverse power law is numerically
indistinguishable from uniform mass -- at r = 4 its effective support is still
111 of 119 anchors -- so the adaptation does nothing.

The schemes below fall into three families:

  ABSOLUTE   use d directly: inv_power, softmax. Fail on a narrow band unless
             the temperature is tuned to it, which makes them corpus-specific.
  STANDARDISED  use (d - mu)/sigma or (d - min)/(max - min). Scale- and
             shift-free, so a narrow band is rescaled to a usable one
             automatically. `std_softmax` and `minmax_softmax`.
  ORDINAL    use only the rank of d, discarding magnitude entirely. Maximally
             robust to the band problem. `rank_exp`, `rank_power`, `knn`, `top1`.

Standardised and ordinal schemes satisfy eq:reweight_identity only in the limit
(a coincident task has the smallest distance, hence rank 1, hence the largest
weight, but the others keep some mass at finite temperature); `knn` and `top1`
satisfy it exactly.
"""

from __future__ import annotations

import numpy as np

#: name -> (family, kwargs) for everything the sweeps try
SCHEMES = {
    "uniform":          ("uniform", {}),
    # absolute
    "inv^1":            ("inv_power", dict(r=1.0)),
    "inv^2":            ("inv_power", dict(r=2.0)),
    "inv^4":            ("inv_power", dict(r=4.0)),
    "inv^8":            ("inv_power", dict(r=8.0)),
    "inv^16":           ("inv_power", dict(r=16.0)),
    "softmax r=1":      ("softmax", dict(r=1.0)),
    "softmax r=5":      ("softmax", dict(r=5.0)),
    "softmax r=20":     ("softmax", dict(r=20.0)),
    "softmax r=50":     ("softmax", dict(r=50.0)),
    # standardised
    "std-softmax r=0.5": ("std_softmax", dict(r=0.5)),
    "std-softmax r=1":  ("std_softmax", dict(r=1.0)),
    "std-softmax r=2":  ("std_softmax", dict(r=2.0)),
    "std-softmax r=4":  ("std_softmax", dict(r=4.0)),
    "std-softmax r=8":  ("std_softmax", dict(r=8.0)),
    "minmax-soft r=5":  ("minmax_softmax", dict(r=5.0)),
    "minmax-soft r=15": ("minmax_softmax", dict(r=15.0)),
    # ordinal
    "rank-exp k=3":     ("rank_exp", dict(kappa=3.0)),
    "rank-exp k=10":    ("rank_exp", dict(kappa=10.0)),
    "rank-exp k=30":    ("rank_exp", dict(kappa=30.0)),
    "rank-exp k=100":   ("rank_exp", dict(kappa=100.0)),
    "rank^-1":          ("rank_power", dict(p=1.0)),
    "rank^-2":          ("rank_power", dict(p=2.0)),
    "knn-3":            ("knn", dict(k=3)),
    "knn-5":            ("knn", dict(k=5)),
    "knn-10":           ("knn", dict(k=10)),
    "knn-25":           ("knn", dict(k=25)),
    "knn-50":           ("knn", dict(k=50)),
    "knn-100":          ("knn", dict(k=100)),
    "top1":             ("top1", {}),
    # gated: hard cut then a smooth decay inside it
    "gate10+soft":      ("gate_softmax", dict(q=0.10, r=2.0)),
    "gate25+soft":      ("gate_softmax", dict(q=0.25, r=2.0)),
}

FAMILY = {n: f for n, (f, _) in SCHEMES.items()}


def masses(dists, family, **kw):
    """(n, M) task distances -> (n, M) rows on the simplex.

    Vectorised over candidates. `dists` may contain a zero (a coincident task);
    every scheme handles that without dividing by zero.
    """
    d = np.asarray(dists, dtype=float)
    d = d + 1e-12

    if family == "uniform":
        s = np.ones_like(d)

    elif family == "inv_power":
        s = (1.0 / d) ** kw["r"]

    elif family == "softmax":
        tau = np.median(d, axis=1, keepdims=True)
        s = np.exp(-kw["r"] * d / np.maximum(tau, 1e-12))

    elif family == "std_softmax":
        mu = d.mean(axis=1, keepdims=True)
        sd = d.std(axis=1, keepdims=True)
        s = np.exp(-kw["r"] * (d - mu) / np.maximum(sd, 1e-12))

    elif family == "minmax_softmax":
        lo = d.min(axis=1, keepdims=True)
        hi = d.max(axis=1, keepdims=True)
        s = np.exp(-kw["r"] * (d - lo) / np.maximum(hi - lo, 1e-12))

    elif family in ("rank_exp", "rank_power"):
        order = np.argsort(d, axis=1)
        rank = np.empty_like(d)
        np.put_along_axis(rank, order,
                          np.tile(np.arange(1, d.shape[1] + 1, dtype=float), (len(d), 1)),
                          axis=1)
        s = (np.exp(-(rank - 1) / kw["kappa"]) if family == "rank_exp"
             else rank ** (-kw["p"]))

    elif family == "knn":
        k = int(min(kw["k"], d.shape[1]))
        idx = np.argsort(d, axis=1)[:, :k]
        s = np.zeros_like(d)
        np.put_along_axis(s, idx, 1.0, axis=1)

    elif family == "top1":
        s = np.zeros_like(d)
        s[np.arange(len(d)), np.argmin(d, axis=1)] = 1.0

    elif family == "gate_softmax":
        k = max(1, int(round(kw["q"] * d.shape[1])))
        idx = np.argsort(d, axis=1)[:, :k]
        mu = d.mean(axis=1, keepdims=True)
        sd = np.maximum(d.std(axis=1, keepdims=True), 1e-12)
        soft = np.exp(-kw["r"] * (d - mu) / sd)
        s = np.zeros_like(d)
        np.put_along_axis(s, idx, np.take_along_axis(soft, idx, axis=1), axis=1)

    else:
        raise ValueError(f"unknown reweighting family: {family}")

    tot = s.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(tot)) | (tot <= 0)
    if bad.any():                       # overflow, e.g. a coincident task under inv_power
        for i in np.flatnonzero(bad.ravel()):
            s[i] = 0.0
            s[i, np.argmin(d[i])] = 1.0
        tot = s.sum(axis=1, keepdims=True)
    return s / tot


def masses_by_name(dists, name):
    fam, kw = SCHEMES[name]
    return masses(dists, fam, **kw)


def effective_support(w):
    """Participation ratio 1/sum(w^2): how many anchors actually carry mass."""
    w = np.asarray(w, dtype=float)
    return 1.0 / np.sum(w ** 2, axis=-1)
