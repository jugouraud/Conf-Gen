"""Full comparison of the Wasserstein support estimators as a prompt filter.

Runs four studies on the CLIP prompt-embedding space and writes figures to
outputs/:

  1. ADAPTIVE ORDER-INF   g^(n,K)_(d,inf): K-nearest task selection x n cuts.
     The n-cuts score is only informative when K is small -- with a large
     reference set the candidate's own MST edge is almost never among the n+1
     heaviest, so B^(n) stops depending on theta at all.
  2. LOW CARDINALITY      s~_(d,1) and s~^(n)_(d,inf) against K k-means centroids.
  3. HEAD-TO-HEAD         best of each family vs the ECF baseline.
  4. FIGURES              score distributions, the K/n heatmap, and a
                          per-tier detection summary.

All scores use the MAHALANOBIS metric induced by the reference split's
probabilistic-PCA covariance (see wasserstein.whitening_map): ecf.tex links
s_(d,1) under that metric to the order-1 inverse ECF, and raw Euclidean
distance in CLIP space discriminates far worse because it is dominated by the
leading "this is a nature prompt" directions.

    python run_wasserstein_experiment.py
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist

import data_cache as dc
from wasserstein import (conformal_radius, score_order1, score_orderinf,
                         whitening_map, LowCardOrder1, LowCardOrderInf)
from ecf import build_safe_set, ecf_score
from sweep_reweighting import masses, auroc

OUT = Path(__file__).parent / "outputs"
EPS = 0.05
M_REF, K_CAL = 60, 40
M_PCA = 32
N_SPLITS = 100
SEED = 0
TIERS = ["degenerate", "borderline", "subtle"]


# ------------------------------------------------------------------ harness --

def splits(theta, safe_idx, n_splits=N_SPLITS, seed=SEED):
    """Yield (whitened coords by block, task distances, reference block) per split."""
    rng = np.random.default_rng(seed)
    for _ in range(n_splits):
        p = rng.permutation(len(safe_idx))
        ref_l, cal_l, tst_l = p[:M_REF], p[M_REF:M_REF + K_CAL], p[M_REF + K_CAL:]
        yield ref_l, cal_l, tst_l


def evaluate(score_of, blocks, eps=EPS):
    """score_of: block -> scores. Returns coverage / per-tier detection / AUROC."""
    r = conformal_radius(score_of("cal"), eps)
    tst = score_of("tst")
    out = {"cov": float(np.mean(tst <= r)), "inf": not np.isfinite(r)}
    for t in TIERS:
        out[t] = float(np.mean(score_of(t) > r))
    out["auroc"] = auroc(tst, score_of("subtle"))
    return out


def agg(rows):
    keys = [k for k in rows[0] if k != "inf"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def stat_line(name, a, extra=""):
    return (f"{name:<26}{a['cov']:>8.3f}{a['degenerate']:>11.3f}"
            f"{a['borderline']:>11.3f}{a['subtle']:>9.3f}{a['auroc']:>8.3f}  {extra}")


HEADER = (f"{'estimator':<26}{'coverage':>8}{'degenerate':>11}"
          f"{'borderline':>11}{'subtle':>9}{'AUROC':>8}")


# ------------------------------------------------- 1. adaptive order-inf -----

def study_orderinf(theta, labels, Dtask, safe_idx, tiers):
    K_LIST = [2, 3, 5, 10, 20, 60]
    N_LIST = [0, 1, 2]
    res = {(K, n): [] for K in K_LIST for n in N_LIST if n <= K - 1}

    for ref_l, cal_l, tst_l in splits(theta, safe_idx):
        ref_g = safe_idx[ref_l]
        wm = whitening_map(theta[ref_g], m=M_PCA)
        Rw = wm(theta[ref_g])
        blocks = {"cal": safe_idx[cal_l], "tst": safe_idx[tst_l], **tiers}
        Zw = {b: wm(theta[idx]) for b, idx in blocks.items()}
        Dt = {b: Dtask[idx][:, ref_l] for b, idx in blocks.items()}

        for K in K_LIST:
            # K task-nearest anchors per candidate (adaptation.tex eq:K_neighbourhood)
            sel = {b: np.argsort(Dt[b], axis=1)[:, :K] for b in blocks}
            for n in N_LIST:
                if n > K - 1:
                    continue
                def sc(b, K=K, n=n, sel=sel, Zw=Zw):
                    return np.array([
                        score_orderinf(Zw[b][i:i + 1], Rw[sel[b][i]], n_cuts=n)[0]
                        for i in range(len(Zw[b]))])
                res[(K, n)].append(evaluate(sc, blocks))

    print("\n=== 1. ADAPTIVE ORDER-INF  g^(n,K)_(d,inf)  (K task-nearest anchors) ===")
    print(HEADER)
    best, best_v = None, -1
    for (K, n), rows in res.items():
        a = agg(rows)
        print(stat_line(f"  K={K:<3} n_cuts={n}", a))
        if a["subtle"] > best_v:
            best, best_v = (K, n), a["subtle"]
    print(f"  best: K={best[0]}, n_cuts={best[1]}  (subtle {best_v:.3f})")
    return res, best


# ---------------------------------------------------- 2. low cardinality -----

def study_lowcard(theta, labels, safe_idx, tiers):
    K_LIST = [5, 10, 20, 30, 60]
    res = {("order1", K): [] for K in K_LIST}
    res.update({("orderinf", K): [] for K in K_LIST})

    for ref_l, cal_l, tst_l in splits(theta, safe_idx):
        ref_g = safe_idx[ref_l]
        wm = whitening_map(theta[ref_g], m=M_PCA)
        Rw = wm(theta[ref_g])
        blocks = {"cal": safe_idx[cal_l], "tst": safe_idx[tst_l], **tiers}
        Zw = {b: wm(theta[idx]) for b, idx in blocks.items()}

        for K in K_LIST:
            lc1 = LowCardOrder1.fit(Rw, K=K)
            lci = LowCardOrderInf.fit(Rw, K=K, n_cuts=0)
            res[("order1", K)].append(evaluate(lambda b: lc1(Zw[b]), blocks))
            res[("orderinf", K)].append(evaluate(lambda b: lci(Zw[b]), blocks))

    print("\n=== 2. LOW CARDINALITY  (K k-means centroids replace M=60 anchors) ===")
    print(HEADER)
    for fam in ("order1", "orderinf"):
        for K in K_LIST:
            a = agg(res[(fam, K)])
            tag = "s~_(d,1)" if fam == "order1" else "s~_(d,inf)"
            print(stat_line(f"  {tag:<12} K={K:<3}", a,
                            f"mem {K}/{M_REF} anchors"))
    return res


# ------------------------------------------------------- 3. head-to-head -----

def study_head_to_head(theta, labels, Dtask, safe_idx, tiers, best_inf):
    K_inf, n_inf = best_inf
    names = ["ECF (baseline)", "s_(d,1)  exact", "s_(d,inf)  exact",
             "g_(d,1)  knn-3", f"g_(d,inf) K={K_inf} n={n_inf}",
             "s~_(d,1)  K=20", "s~_(d,inf) K=20"]
    res = {k: [] for k in names}
    raw = {k: {} for k in names}          # last split's scores, for the figures

    for ref_l, cal_l, tst_l in splits(theta, safe_idx):
        ref_g = safe_idx[ref_l]
        wm = whitening_map(theta[ref_g], m=M_PCA)
        Rw = wm(theta[ref_g])
        blocks = {"cal": safe_idx[cal_l], "tst": safe_idx[tst_l], **tiers}
        Zw = {b: wm(theta[idx]) for b, idx in blocks.items()}
        Dt = {b: Dtask[idx][:, ref_l] for b, idx in blocks.items()}

        ss = build_safe_set(torch.tensor(theta[ref_g]), torch.tensor(theta[safe_idx[cal_l]]),
                            epsilon=EPS, m=M_PCA)
        lc1 = LowCardOrder1.fit(Rw, K=20)
        lci = LowCardOrderInf.fit(Rw, K=20, n_cuts=0)
        sel = {b: np.argsort(Dt[b], axis=1)[:, :K_inf] for b in blocks}
        Wk = {b: masses(Dt[b], "knn", knn=3, rescaling_order=1.0) for b in blocks}

        fns = {
            "ECF (baseline)":  lambda b: ecf_score(torch.tensor(theta[blocks[b]]), ss).numpy(),
            "s_(d,1)  exact":  lambda b: score_order1(Zw[b], Rw),
            "s_(d,inf)  exact": lambda b: score_orderinf(Zw[b], Rw, 0),
            "g_(d,1)  knn-3":  lambda b: (cdist(Zw[b], Rw) * Wk[b]).sum(1) / (M_REF + 1),
            f"g_(d,inf) K={K_inf} n={n_inf}": lambda b: np.array([
                score_orderinf(Zw[b][i:i + 1], Rw[sel[b][i]], n_cuts=n_inf)[0]
                for i in range(len(Zw[b]))]),
            "s~_(d,1)  K=20":  lambda b: lc1(Zw[b]),
            "s~_(d,inf) K=20": lambda b: lci(Zw[b]),
        }
        for k, f in fns.items():
            res[k].append(evaluate(f, blocks))
            raw[k] = {b: f(b) for b in blocks}
            raw[k]["_r"] = conformal_radius(f("cal"), EPS)

    exact = np.ceil((K_CAL + 1) * (1 - EPS)) / (K_CAL + 1)
    print(f"\n=== 3. HEAD-TO-HEAD  eps={EPS}  M={M_REF}/K={K_CAL}/"
          f"{120-M_REF-K_CAL}  {N_SPLITS} splits ===")
    print(f"    valid coverage target: {exact:.4f} (exact split-CP expectation)")
    print(HEADER)
    for k in names:
        print(stat_line("  " + k, agg(res[k])))
    return res, raw, names


# ------------------------------------------------------------- 4. figures ----

def figures(res_inf, res_lc, res_h2h, raw, names):
    OUT.mkdir(exist_ok=True)

    # (a) K x n_cuts heatmap for the adaptive order-inf
    K_LIST, N_LIST = [2, 3, 5, 10, 20, 60], [0, 1, 2]
    Mt = np.full((len(K_LIST), len(N_LIST)), np.nan)
    for i, K in enumerate(K_LIST):
        for j, n in enumerate(N_LIST):
            if (K, n) in res_inf:
                Mt[i, j] = agg(res_inf[(K, n)])["subtle"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(Mt, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(N_LIST)), [f"n={n}" for n in N_LIST])
    ax.set_yticks(range(len(K_LIST)), [f"K={K}" for K in K_LIST])
    for i in range(len(K_LIST)):
        for j in range(len(N_LIST)):
            if not np.isnan(Mt[i, j]):
                ax.text(j, i, f"{Mt[i,j]:.2f}", ha="center", va="center",
                        color="w" if Mt[i, j] < np.nanmax(Mt) * 0.6 else "k", fontsize=9)
    ax.set_title("Adaptive order-∞: subtle-prompt detection\n$g^{(n,K)}_{d,\\infty}$")
    fig.colorbar(im, ax=ax, label="fraction of SUBTLE blocked")
    fig.tight_layout(); fig.savefig(OUT / "w_orderinf_K_ncuts.png", dpi=150); plt.close(fig)

    # (b) low-cardinality: detection vs number of centroids
    K_LC = [5, 10, 20, 30, 60]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, fam, tag in [(ax[0], "order1", r"$\tilde{s}_{d,1}$"),
                        (ax[1], "orderinf", r"$\tilde{s}_{d,\infty}$")]:
        for tier, c in [("subtle", "tab:red"), ("borderline", "tab:orange"),
                        ("degenerate", "tab:green")]:
            a.plot(K_LC, [agg(res_lc[(fam, K)])[tier] for K in K_LC], "o-", color=c, label=tier)
        a.plot(K_LC, [agg(res_lc[(fam, K)])["cov"] for K in K_LC], "s--",
               color="tab:blue", label="safe coverage")
        a.axhline(1 - EPS, ls=":", c="k", lw=1)
        a.set_xlabel("K centroids"); a.set_title(f"{tag} — lower cardinality")
        a.set_ylim(0, 1.05); a.grid(alpha=.3)
    ax[0].set_ylabel("rate"); ax[1].legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(OUT / "w_lowcardinality.png", dpi=150); plt.close(fig)

    # (c) score distributions per tier for each estimator (last split)
    show = names
    fig, axes = plt.subplots(len(show), 1, figsize=(9, 2.05 * len(show)), sharex=False)
    for a, k in zip(np.atleast_1d(axes), show):
        r = raw[k]["_r"]
        for b, c in [("tst", "tab:blue"), ("borderline", "tab:orange"),
                     ("subtle", "tab:red"), ("degenerate", "tab:green")]:
            v = np.asarray(raw[k][b], dtype=float)
            a.scatter(v, np.random.default_rng(0).normal(0, .06, len(v)),
                      s=14, alpha=.75, color=c, label=b if k == show[0] else None)
        if np.isfinite(r):
            a.axvline(r, color="k", ls="--", lw=1.4)
        a.set_yticks([]); a.set_ylabel(k, rotation=0, ha="right", fontsize=8, va="center")
        a.grid(alpha=.25, axis="x")
    np.atleast_1d(axes)[0].legend(ncol=4, fontsize=8, loc="upper center",
                                  bbox_to_anchor=(.5, 1.9))
    np.atleast_1d(axes)[-1].set_xlabel("nonconformity score  (dashed = conformal radius)")
    fig.suptitle("Score separation by prompt tier", y=.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(OUT / "w_score_distributions.png", dpi=150); plt.close(fig)

    # (d) head-to-head detection bars
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(show)); w = .26
    for i, (tier, c) in enumerate([("degenerate", "tab:green"),
                                   ("borderline", "tab:orange"), ("subtle", "tab:red")]):
        ax.bar(x + (i - 1) * w, [agg(res_h2h[k])[tier] for k in show], w, color=c, label=tier)
    ax.plot(x, [agg(res_h2h[k])["cov"] for k in show], "kd--", ms=6, label="safe coverage")
    ax.axhline(1 - EPS, ls=":", c="k", lw=1)
    ax.set_xticks(x, show, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("fraction blocked"); ax.set_ylim(0, 1.08); ax.legend(fontsize=8, ncol=4)
    ax.set_title(f"Prompt-filter comparison (ε={EPS}, {N_SPLITS} splits)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "w_head_to_head.png", dpi=150); plt.close(fig)
    print(f"\nfigures -> {OUT}/w_*.png")


def main():
    theta, labels, Dtask, prompts = dc.load()
    m = dc.masks(labels)
    safe_idx = np.flatnonzero(m["safe"])
    tiers = {t: np.flatnonzero(m[t]) for t in TIERS}

    res_inf, best_inf = study_orderinf(theta, labels, Dtask, safe_idx, tiers)
    res_lc = study_lowcard(theta, labels, safe_idx, tiers)
    res_h2h, raw, names = study_head_to_head(theta, labels, Dtask, safe_idx, tiers, best_inf)
    figures(res_inf, res_lc, res_h2h, raw, names)


if __name__ == "__main__":
    main()
