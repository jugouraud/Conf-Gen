"""Does restoring ORDER rescue the token-sequence representation?

README §15.5 direction 1(a). §11.5 found that scoring token clouds with W1 loses
to the pooled [EOS] vector (AUROC* 0.708 vs 0.757), and attributed it to W1 over
a bag discarding order and contextual integration. This tests the order half of
that explanation directly.

Each whitened token vector is augmented with its position,

    x_j  ->  [ x_j ,  lambda * pos_j ],

so the transport cost between token j of one prompt and token k of another gains
a term lambda^2 (pos_j - pos_k)^2. lambda = 0 is exactly the unordered W1 already
measured; large lambda forces a near-monotone alignment. Two encodings:

    normalised   pos_j = j / (L-1)     aligns beginnings and ends, length-invariant
    absolute     pos_j = j             also penalises length differences

If order is the missing ingredient, AUROC* should rise with lambda and approach
or exceed the pooled baseline.

    python run_ordered_experiment.py            # ~4 min
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
import ot as pot

import cache_large as cl
from wasserstein import whitening_map, conformal_radius
from sweep_reweighting import auroc

OUT = Path(__file__).parent / "outputs"
BETA = 0.05
M_REF, K_CAL, N_TEST = 500, 300, 200
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def augment(clouds, lam, mode):
    """Append lam * position to each token vector."""
    out = []
    for c in clouds:
        L = len(c)
        if mode == "normalised":
            pos = np.arange(L) / max(L - 1, 1)
        else:
            pos = np.arange(L, dtype=float)
        out.append(np.concatenate([c, (lam * pos)[:, None]], axis=1))
    return out


def w1_block(rows_cl, cols_cl):
    D = np.empty((len(rows_cl), len(cols_cl)))
    for a, ci in enumerate(rows_cl):
        ai = np.ones(len(ci)) / len(ci)
        for b, cj in enumerate(cols_cl):
            C = np.ascontiguousarray(cdist(ci, cj))
            D[a, b] = pot.emd2(ai, np.ones(len(cj)) / len(cj), C, numThreads=1)
    return D


def main():
    d = cl.load()
    ng, npool = d["n_geom"], d["n_pool"]
    labels, clouds, theta = d["labels"], d["clouds"], d["theta"]

    tok_wm = whitening_map(np.vstack([clouds[i] for i in d["geom"]]), m=32)
    cw = [tok_wm(c) for c in clouds]
    scale = float(np.median([np.linalg.norm(c, axis=1).mean() for c in cw]))
    print(f"typical whitened token norm: {scale:.2f}  "
          f"(lambda is quoted in these units)")

    pool = np.arange(ng, ng + npool)
    hw = np.flatnonzero(labels == "safe_handwritten")
    sub = np.flatnonzero(labels == "subtle")
    deg = np.flatnonzero(labels == "degenerate")

    rng = np.random.default_rng(0)
    p = rng.permutation(npool)
    ref = pool[p[:M_REF]]
    cal = pool[p[M_REF:M_REF + K_CAL]]
    tst = pool[p[M_REF + K_CAL:M_REF + K_CAL + N_TEST]]
    rows = np.concatenate([cal, tst, hw, sub, deg])
    sizes = np.cumsum([len(cal), len(tst), len(hw), len(sub)])

    # pooled-vector baseline, same split
    th_wm = whitening_map(theta[d["geom"]], m=32)
    Tw = th_wm(theta)
    Dp = cdist(Tw[rows], Tw[ref])
    s_p = Dp.sum(1) / (M_REF * (M_REF + 1))
    b_cal, b_tst, b_hw, b_sub, b_deg = np.split(s_p, sizes)
    eps = conformal_radius(b_cal, BETA)
    base = dict(auroc=auroc(b_hw, b_sub), cov=np.mean(b_tst <= eps),
                deg=np.mean(b_deg > eps), sub=np.mean(b_sub > eps))
    print(f"\npooled [EOS] baseline: AUROC* {base['auroc']:.3f}  "
          f"cov {base['cov']:.3f}  degen {base['deg']:.3f}  subtle {base['sub']:.3f}")

    results = {}
    for mode in ("normalised", "absolute"):
        print(f"\n{'='*76}\nposition encoding: {mode}\n{'='*76}")
        print(f"{'lambda':>8}{'AUROC*':>9}{'cov':>8}{'degen':>8}{'subtle':>8}{'sec':>7}")
        print("-" * 48)
        results[mode] = []
        for lam in LAMBDAS:
            t0 = time.time()
            aug = augment(cw, lam * scale, mode)
            D = w1_block([aug[i] for i in rows], [aug[i] for i in ref])
            s = D.sum(1) / (M_REF * (M_REF + 1))
            c_, t_, h_, u_, g_ = np.split(s, sizes)
            e = conformal_radius(c_, BETA)
            r = dict(lam=lam, auroc=auroc(h_, u_), cov=np.mean(t_ <= e),
                     deg=np.mean(g_ > e), sub=np.mean(u_ > e))
            results[mode].append(r)
            print(f"{lam:>8.2f}{r['auroc']:>9.3f}{r['cov']:>8.3f}"
                  f"{r['deg']:>8.3f}{r['sub']:>8.3f}{time.time() - t0:>7.0f}")

    # ---------------------------------------------------------------- plot --
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for mode, mk in zip(results, ["o-", "s-"]):
        ax[0].plot(LAMBDAS, [r["auroc"] for r in results[mode]], mk, label=mode)
        ax[1].plot(LAMBDAS, [r["sub"] for r in results[mode]], mk, label=mode)
    for a, key, ttl in [(ax[0], "auroc", "AUROC(hand-written safe vs subtle)"),
                        (ax[1], "sub", "subtle blocked at β=0.05")]:
        a.axhline(base[key], ls="--", c="k", lw=1.3, label="pooled [EOS] baseline")
        a.set_xlabel("λ  (positional weight, in whitened-token-norm units)")
        a.set_title(ttl); a.legend(fontsize=8); a.grid(alpha=.3)
    ax[0].set_ylabel("AUROC*")
    fig.suptitle("Does restoring token order rescue the sequence representation?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(OUT / "s_ordered.png", dpi=150)
    print(f"\nfigure -> {OUT}/s_ordered.png")


if __name__ == "__main__":
    main()
