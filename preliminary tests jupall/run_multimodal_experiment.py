"""A BIMODAL safe set: does the manuscript's n-cuts machinery finally pay off?

README §15.5 direction 3. §4.3 found graph cuts catastrophic (degenerate
detection 1.00 -> 0.05) and blamed the safe set being unimodal. This builds a
genuinely bimodal safe set -- desert AND polar landscapes, nothing else -- and
tests the prediction that the sign flips.

THE PREDICTION, stated before measuring
---------------------------------------
For a bimodal D, MST(D) contains one long BRIDGE edge between the clusters.

    n = 0   s_(d,inf)(theta) = B(MST(D u {theta})) is dominated by that bridge
            for every in-distribution theta, so the calibrated radius is
            eps_0 ~ BRIDGE LENGTH. The region is a union of balls of that
            radius -- large enough to swallow the gap between the modes.

    n = 1   the bridge is cut, so the score becomes the largest WITHIN-cluster
            edge and eps_1 ~ within-cluster scale << eps_0. The region is a
            union of much smaller balls, one family per mode, and the gap is
            excluded.

So on a bimodal safe set cuts should HELP, reversing §4.3 -- and the tier that
should separate the two geometries is GAP: prompts mixing desert and polar
vocabulary, which land between the modes. A unimodal model (ECF, order-1) puts
its mass exactly there and should accept them.

Negatives are other LANDSCAPE types, not gibberish: forest, coast, meadow,
wetland, tropical, alpine. `alpine` shares snow vocabulary with the polar mode
and is reported separately.

    python run_multimodal_experiment.py            # ~2 min
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
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

import prompts_multimodal as PM
from wasserstein import whitening_map, conformal_radius, LowCardOrderInf
from run_residual_experiment import Basis
from corpus_audit import audit_scores
from sweep_reweighting import auroc

OUT = Path(__file__).parent / "outputs"
CACHE = Path(__file__).parent / "cache_multimodal.npz"
BETA = 0.05
N_MODE = 300
M_REF, K_CAL, N_TEST = 300, 200, 100
N_SPLITS = 40
M_PCA = 32
DEVICE = "mps"
OTHERS = ["forest", "coast", "meadow", "wetland", "tropical", "alpine"]


# ------------------------------------------------------------------ data ---

def build():
    from transformers import CLIPTokenizer, CLIPTextModel
    groups = {"mode_a_desert": PM.mode_a(N_MODE), "mode_b_polar": PM.mode_b(N_MODE),
              "gap": PM.gap(60), **{f"other_{k}": PM.other(k, 60) for k in OTHERS}}
    prompts, labels = [], []
    for k, v in groups.items():
        prompts += v
        labels += [k] * len(v)
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
    out = []
    for i in range(0, len(prompts), 64):
        b = prompts[i:i + 64]
        t = tok(b, padding="max_length", max_length=77, truncation=True,
                return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            h = enc(t.input_ids).last_hidden_state
        e = t.input_ids.argmax(dim=-1)
        out.append(h[torch.arange(len(b), device=DEVICE), e].cpu().numpy())
    theta = np.concatenate(out).astype(np.float64)
    np.savez_compressed(CACHE, theta=theta, labels=np.array(labels),
                        prompts=np.array(prompts, dtype=object))
    return theta, np.array(labels), np.array(prompts, dtype=object)


def load():
    if not CACHE.exists():
        return build()
    z = np.load(CACHE, allow_pickle=True)
    return z["theta"].astype(np.float64), z["labels"], z["prompts"]


def bottleneck_n(D, n):
    """(n+1)-th heaviest MST edge of a dense distance matrix."""
    w = np.sort(minimum_spanning_tree(csr_array(D)).toarray().ravel())
    w = w[w > 0]
    return float(w[-(n + 1)]) if len(w) > n else 0.0


def score_orderinf_D(Dca, Daa, rows, n):
    """B^(n)(MST(anchors u {theta})) using precomputed distances."""
    M = Daa.shape[0]
    base = minimum_spanning_tree(csr_array(Daa)).toarray()
    base = np.maximum(base, base.T)
    G = np.zeros((M + 1, M + 1))
    G[:M, :M] = base
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        G[:M, M] = Dca[r]
        G[M, :M] = Dca[r]
        out[i] = bottleneck_n(G, n)
    return out


def features(theta, ref, metric, m=M_PCA):
    """Feature map in which distances are measured.

    `full`     the whitened Mahalanobis metric used everywhere else. The
               736-dimensional isotropic residual contributes ~25 units of
               distance against ~8 from the retained subspace, so it DROWNS the
               between-mode signal: Fisher separation falls from 28.7 (raw) to
               1.94. The modes become geometrically invisible.
    `subspace` the top-m whitened coordinates only. Preserves the separation
               (Fisher 27.6) because the between-mode direction IS PCA-1
               (alignment 0.991). Loses off-manifold sensitivity, which is the
               price -- see §11's truncation sweep.
    """
    B_ = Basis(theta[ref], m)
    A_, Bt_, cw, pw = B_.parts(theta)
    if metric == "raw":
        # the ONLY metric in which 2-means recovers the two modes (ARI 0.973);
        # whitening destroys them (ARI -0.003) because the between-mode
        # direction is PCA-1 and gets divided by the largest eigenvalue.
        return theta - theta[ref].mean(0), A_, Bt_
    if metric == "subspace":
        return cw, A_, Bt_
    return np.concatenate([cw, pw], axis=1), A_, Bt_


def main():
    theta, labels, prompts = load()
    a_idx = np.flatnonzero(labels == "mode_a_desert")
    b_idx = np.flatnonzero(labels == "mode_b_polar")
    tiers = {"gap": np.flatnonzero(labels == "gap")}
    tiers.update({k: np.flatnonzero(labels == f"other_{k}") for k in OTHERS})

    # ---------------------------------------------- 0. is it really bimodal --
    rng = np.random.default_rng(0)
    pa, pb = rng.permutation(len(a_idx)), rng.permutation(len(b_idx))
    ref = np.concatenate([a_idx[pa[:M_REF // 2]], b_idx[pb[:M_REF // 2]]])
    METRIC = os.environ.get("MM_METRIC", "subspace")
    print(f"metric = {METRIC}\n")
    W, A_, Bt_ = features(theta, ref, METRIC)

    Wa, Wb = W[a_idx], W[b_idx]
    within = (np.median(cdist(Wa, Wa)[np.triu_indices(len(Wa), 1)]),
              np.median(cdist(Wb, Wb)[np.triu_indices(len(Wb), 1)]))
    between = np.median(cdist(Wa, Wb))
    Dref = cdist(W[ref], W[ref])
    mst = minimum_spanning_tree(csr_array(Dref)).toarray()
    ii, jj = np.nonzero(mst)
    wts = mst[ii, jj]
    k = np.argmax(wts)
    is_a = np.isin(ref, a_idx)
    bridge_crosses = is_a[ii[k]] != is_a[jj[k]]

    print(f"{'='*92}\n0. IS THE SAFE SET ACTUALLY BIMODAL?\n{'='*92}")
    print(f"  median within-mode distance : desert {within[0]:.2f}   polar {within[1]:.2f}")
    print(f"  median between-mode distance: {between:.2f}   "
          f"(ratio {between / np.mean(within):.2f})")
    print(f"  heaviest MST edge = {wts[k]:.2f}; 2nd heaviest = {np.sort(wts)[-2]:.2f}")
    print(f"  does the heaviest edge CROSS the two modes? {bridge_crosses}")
    print(f"  -> the bridge is {wts[k] / np.sort(wts)[-2]:.2f}x the next edge")

    # ------------------------------------------- 1. the radius prediction ---
    print(f"\n{'='*92}\n1. THE PREDICTION: eps_0 ~ bridge, eps_1 ~ within-cluster\n{'='*92}")
    cal = np.concatenate([a_idx[pa[M_REF // 2:M_REF // 2 + K_CAL // 2]],
                          b_idx[pb[M_REF // 2:M_REF // 2 + K_CAL // 2]]])
    Dca = cdist(W, W[ref])
    Daa = Dca[ref]
    print(f"{'n_cuts':>8}{'eps':>10}{'c0 at eps':>12}{'region':>26}")
    print("-" * 58)
    for n in (0, 1, 2):
        e = conformal_radius(score_orderinf_D(Dca, Daa, cal, n), BETA)
        Adj = csr_array(((Daa <= e) & ~np.eye(len(Daa), dtype=bool)).astype(np.int8))
        c0, _ = connected_components(Adj, directed=False)
        q = c0 - n
        desc = ("EMPTY (must touch all components)" if q > 1 else
                "union of balls" if q == 1 else "whole space (vacuous)")
        print(f"{n:>8}{e:>10.3f}{c0:>12}{desc:>26}")

    # -------------------------------------------- 2. estimator comparison ---
    print(f"\n{'='*92}\n2. ESTIMATOR COMPARISON   {N_SPLITS} splits, "
          f"M={M_REF} K={K_CAL} test={N_TEST}, beta={BETA}\n{'='*92}")
    names = ["ECF (ellipsoid)", "s_perp", "s_(d,1) uniform",
             "s_(d,inf) n=0", "s_(d,inf) n=1", "s_(d,inf) n=2",
             "s~_(d,inf) K=2"]
    acc = {n: {k: [] for k in ["cov", "auroc", "gap"] + OTHERS} for n in names}

    for sp in range(N_SPLITS):
        r = np.random.default_rng(sp)
        pa, pb = r.permutation(len(a_idx)), r.permutation(len(b_idx))
        h = M_REF // 2
        ref = np.concatenate([a_idx[pa[:h]], b_idx[pb[:h]]])
        cal = np.concatenate([a_idx[pa[h:h + K_CAL // 2]], b_idx[pb[h:h + K_CAL // 2]]])
        tst = np.concatenate([a_idx[pa[h + K_CAL // 2:h + K_CAL // 2 + N_TEST // 2]],
                              b_idx[pb[h + K_CAL // 2:h + K_CAL // 2 + N_TEST // 2]]])
        W, A_, Bt_ = features(theta, ref, METRIC)
        Dca = cdist(W, W[ref])
        Daa = Dca[ref]
        mu = W[ref].mean(0)
        M = len(ref)
        lc = LowCardOrderInf.fit(W[ref], K=2, n_cuts=0)

        fns = {
            "ECF (ellipsoid)": lambda i: ((W[i] - mu) ** 2).sum(1),
            "s_perp": lambda i: np.sqrt(Bt_[i]),
            "s_(d,1) uniform": lambda i: Dca[i].sum(1) / (M * (M + 1)),
            "s_(d,inf) n=0": lambda i: score_orderinf_D(Dca, Daa, i, 0),
            "s_(d,inf) n=1": lambda i: score_orderinf_D(Dca, Daa, i, 1),
            "s_(d,inf) n=2": lambda i: score_orderinf_D(Dca, Daa, i, 2),
            "s~_(d,inf) K=2": lambda i: lc(W[i]),
        }
        for nm, f in fns.items():
            e = conformal_radius(f(cal), BETA)
            a = acc[nm]
            a["cov"].append(np.mean(f(tst) <= e))
            a["gap"].append(np.mean(f(tiers["gap"]) > e))
            for k in OTHERS:
                a[k].append(np.mean(f(tiers[k]) > e))
            a["auroc"].append(auroc(f(tst), f(tiers["gap"])))

    exact = np.ceil((K_CAL + 1) * (1 - BETA)) / (K_CAL + 1)
    print(f"target coverage {exact:.4f}")
    print(f"{'estimator':<18}{'cov':>7}{'GAP':>7}" +
          "".join(f"{k[:7]:>9}" for k in OTHERS) + f"{'AUROC':>8}")
    print("-" * 92)
    for nm in names:
        a = acc[nm]
        print(f"{nm:<18}{np.mean(a['cov']):>7.3f}{np.mean(a['gap']):>7.3f}"
              + "".join(f"{np.mean(a[k]):>9.3f}" for k in OTHERS)
              + f"{np.mean(a['auroc']):>8.3f}")
    print("-" * 92)
    base = np.array(acc["s_(d,inf) n=0"]["gap"])
    print("paired GAP blocking against n=0 (the cuts prediction):")
    for nm in names:
        if nm == "s_(d,inf) n=0":
            continue
        dd = np.array(acc[nm]["gap"]) - base
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)) + 1e-12)
        print(f"   {nm:<20}{dd.mean():+.3f}   (t = {t:+.1f})")

    # ------------------------------------------------------------- figure --
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))
    x = np.arange(len(names))
    ax[0].bar(x, [np.mean(acc[n]["gap"]) for n in names], color="tab:red")
    ax[0].plot(x, [np.mean(acc[n]["cov"]) for n in names], "kd--", label="safe coverage")
    ax[0].axhline(1 - BETA, ls=":", c="k")
    ax[0].set_xticks(x, [n.replace(" ", "\n") for n in names], fontsize=6.5)
    ax[0].set_ylabel("fraction blocked"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("GAP tier — between the two modes")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    for i, k in enumerate(OTHERS):
        ax[1].plot(x, [np.mean(acc[n][k]) for n in names], "o-", label=k, lw=1.3, ms=4)
    ax[1].set_xticks(x, [n.replace(" ", "\n") for n in names], fontsize=6.5)
    ax[1].set_ylim(0, 1.05); ax[1].set_title("other landscape types")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    # 2-D view of the two modes and the gap
    Wp = W - W[ref].mean(0)
    U, S, Vt = np.linalg.svd(Wp[ref], full_matrices=False)
    P = Wp @ Vt[:2].T
    for idx, c, l in [(a_idx, "tab:orange", "mode A desert"),
                      (b_idx, "tab:cyan", "mode B polar"),
                      (tiers["gap"], "tab:red", "gap"),
                      (tiers["forest"], "tab:green", "forest")]:
        ax[2].scatter(P[idx, 0], P[idx, 1], s=8, alpha=.6, color=c, label=l)
    ax[2].set_title("the safe set is bimodal"); ax[2].legend(fontsize=7)
    ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.suptitle("A bimodal safe set: do graph cuts finally pay off?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(OUT / f"s_multimodal_{METRIC}.png", dpi=150)
    print(f"\nfigure -> {OUT}/s_multimodal_{METRIC}.png")


if __name__ == "__main__":
    main()
