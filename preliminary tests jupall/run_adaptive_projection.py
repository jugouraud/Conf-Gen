"""Does ADAPTING the region move it away from the true support?

Hypothesis under test. The adaptive order-inf score g^(0,K) builds its region
from only the K anchors nearest in TASK space. If the task metric mis-ranks,
those K anchors are not the ones nearest in weight space, so

  * the region is centred on the wrong part of the support;
  * the calibrated radius has to inflate to keep covering safe prompts;
  * the metric projection lands on a shell further from real data than the
    non-adaptive projection would; and
  * repair_t = 1 lands on one of the K SELECTED anchors, which -- if selection
    was wrong -- is a semantically inappropriate safe prompt.

In other words: the adaptation may be creating the very gap that repair_t
exists to close.

The bimodal safe set of §17 makes "wrong selection" checkable, because every
anchor carries a ground-truth mode (desert or polar). Selecting anchors from the
wrong mode for a desert-ish query is a visible, countable error.

Swept: K in {2, 3, 5, 10, 25, 50, 100, 300}, where K = 300 = M is the
non-adaptive score. n_cuts in {0, 1}.

    python run_adaptive_projection.py            # ~8 min
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
import ot as pot

from run_multimodal_experiment import load, M_REF, K_CAL, BETA, bottleneck_n
from multimodal_projection import sequences
from wasserstein import whitening_map, conformal_radius
from conformal_regions import UnionOfBallsRegion

OUT = Path(__file__).parent / "outputs"
TASK_CACHE = Path(__file__).parent / "cache_mm_task.npz"
K_LIST = [2, 3, 5, 10, 25, 50, 100, 300]
N_TEST = 100
TIERS = ["gap", "other_forest", "other_coast", "other_alpine"]


# ------------------------------------------------------------ task metric --

def task_clouds(seq, eos):
    """Content-token clouds: the task descriptor tau of README §2.2."""
    return [seq[i, 1:max(int(eos[i]), 2)] for i in range(len(seq))]


def task_distances(clouds, cols, tok_wm):
    """W1 between whitened token clouds: rows = everything, cols = the anchors."""
    if TASK_CACHE.exists():
        z = np.load(TASK_CACHE)
        if z["cols"].shape[0] == len(cols) and np.array_equal(z["cols"], cols):
            return z["D"]
    cw = [tok_wm(c) for c in clouds]
    colc = [cw[j] for j in cols]
    D = np.empty((len(cw), len(cols)))
    for a, ci in enumerate(cw):
        ai = np.ones(len(ci)) / len(ci)
        for b, cj in enumerate(colc):
            C = np.ascontiguousarray(cdist(ci, cj))
            D[a, b] = pot.emd2(ai, np.ones(len(cj)) / len(cj), C, numThreads=1)
        if (a + 1) % 200 == 0:
            print(f"      task W1 {a + 1}/{len(cw)}")
    np.savez_compressed(TASK_CACHE, D=D, cols=cols)
    return D


def main():
    theta, labels, prompts = load()
    seq, eos = sequences(prompts)
    a_idx = np.flatnonzero(labels == "mode_a_desert")
    b_idx = np.flatnonzero(labels == "mode_b_polar")

    rng = np.random.default_rng(0)
    pa, pb = rng.permutation(len(a_idx)), rng.permutation(len(b_idx))
    h = M_REF // 2
    ref = np.concatenate([a_idx[pa[:h]], b_idx[pb[:h]]])
    cal = np.concatenate([a_idx[pa[h:h + K_CAL // 2]], b_idx[pb[h:h + K_CAL // 2]]])
    tst = np.concatenate([a_idx[pa[h + K_CAL // 2:h + K_CAL // 2 + N_TEST // 2]],
                          b_idx[pb[h + K_CAL // 2:h + K_CAL // 2 + N_TEST // 2]]])
    ref_is_a = np.isin(ref, a_idx)

    # weight space: flattened whitened sequences (the lift that repairs)
    L, d = seq.shape[1], seq.shape[2]
    flat = seq.reshape(len(seq), L * d).astype(np.float32)
    wm = whitening_map(flat[ref].astype(np.float64), m=32)
    Fw = wm(flat.astype(np.float64)).astype(np.float32)
    Dw = cdist(Fw, Fw[ref])                       # (N, M) weight-space distances
    Daa = Dw[ref]                                 # (M, M) anchor-anchor, ONCE.
    # Recomputing this inside the per-candidate loop costs ~5e9 flops per call
    # in 59k dimensions; precomputing turns the sweep from hours into seconds.

    # task space
    print("[1/4] task-space W1 between token clouds ...")
    tok_wm = whitening_map(np.vstack([seq[i, 1:max(int(eos[i]), 2)] for i in ref]), m=32)
    Dt = task_distances(task_clouds(seq, eos), ref, tok_wm)

    tiers = {t: np.flatnonzero(labels == t) for t in TIERS}
    probe = np.concatenate([tst] + [tiers[t] for t in TIERS])
    probe_tier = np.array(["safe"] * len(tst)
                          + sum([[t] * len(tiers[t]) for t in TIERS], []))

    # ------------------------------------------- 2. does selection go wrong? --
    print("\n" + "=" * 96)
    print("A. IS THE TASK-SPACE SELECTION PICKING THE RIGHT ANCHORS?")
    print("=" * 96)
    print(f"{'K':>5}{'overlap w/ weight-K':>21}{'d(sel)/d(all)':>16}"
          f"{'same-mode frac':>17}{'random baseline':>17}")
    print("-" * 96)
    ord_t = np.argsort(Dt, axis=1)
    ord_w = np.argsort(Dw, axis=1)
    sel_stats = {}
    for K in K_LIST:
        ov, ratio, same = [], [], []
        for i in probe:
            st, sw = set(ord_t[i, :K]), set(ord_w[i, :K])
            ov.append(len(st & sw) / K)
            ratio.append(Dw[i, ord_t[i, :K]].min() / Dw[i, ord_w[i, 0]])
        # mode agreement, safe prompts only (they have a ground-truth mode)
        for i in tst:
            true_a = i in a_idx
            same.append(np.mean(ref_is_a[ord_t[i, :K]] == true_a))
        sel_stats[K] = (np.mean(ov), np.mean(ratio), np.mean(same))
        print(f"{K:>5}{np.mean(ov):>21.3f}{np.mean(ratio):>16.3f}"
              f"{np.mean(same):>17.3f}{K / len(ref):>17.3f}")
    print("-" * 96)
    print("overlap  = |task-K ∩ weight-K| / K.  1.0 would mean the task metric")
    print("           reproduces the weight-space ranking exactly.")
    print("d(sel)/d(all) = distance to the nearest SELECTED anchor over distance to")
    print("           the nearest anchor overall. >1 means the region was moved away")
    print("           from the closest real data -- the hypothesis under test.")
    print("same-mode frac = fraction of selected anchors from the query's true mode;")
    print("           0.5 is chance on a balanced bimodal corpus.")

    # ------------------------------------ 3. radius, coverage, projection ----
    print("\n" + "=" * 96)
    print("B. CONFORMAL BEHAVIOUR AND PROJECTION QUALITY vs K")
    print("=" * 96)
    print(f"{'K':>5}{'n':>3}{'eps':>9}{'cov':>7}{'gap':>7}{'forest':>8}{'coast':>7}"
          f"{'alpine':>8}{'d(proj,real)':>14}{'repair mode ok':>16}")
    print("-" * 96)
    res = {}
    for n_cuts in (0, 1):
        for K in K_LIST:
            def score(idx):
                out = np.empty(len(idx))
                for j, i in enumerate(idx):
                    s = ord_t[i, :K]
                    n = len(s)
                    G = np.zeros((n + 1, n + 1))
                    G[:n, :n] = Daa[np.ix_(s, s)]
                    dd = Dw[i, s]
                    G[n, :n] = dd
                    G[:n, n] = dd
                    out[j] = bottleneck_n(G, n_cuts)
                return out

            eps = conformal_radius(score(cal), BETA)
            cov = np.mean(score(tst) <= eps)
            blk = {t: np.mean(score(tiers[t]) > eps) for t in TIERS}

            # projection / repair, using each candidate's OWN adaptive region
            dproj, mode_ok = [], []
            for i in np.concatenate([tiers[t] for t in TIERS]):
                s = ord_t[i, :K]
                A = Fw[ref[s]].astype(np.float64)
                reg = UnionOfBallsRegion(A, np.full(K, eps), support=A)
                z = Fw[i:i + 1].astype(np.float64)
                if reg.contains(z)[0]:
                    continue
                P = reg.project(z)[0]
                dproj.append(np.linalg.norm(P[None, :] - Fw[ref], axis=1).min())
                # repair_t=1 lands on a selected anchor: is it the weight-nearest?
                R = reg.repair(z, t=1.0)[0]
                landed = ref[s][int(np.argmin(np.linalg.norm(A - R[None, :], axis=1)))]
                mode_ok.append(landed == ref[ord_w[i, 0]])
            res[(K, n_cuts)] = dict(eps=eps, cov=cov, **blk,
                                    dproj=np.mean(dproj) if dproj else np.nan,
                                    mode=np.mean(mode_ok) if mode_ok else np.nan)
            r = res[(K, n_cuts)]
            print(f"{K:>5}{n_cuts:>3}{eps:>9.2f}{cov:>7.3f}{r['gap']:>7.3f}"
                  f"{r['other_forest']:>8.3f}{r['other_coast']:>7.3f}"
                  f"{r['other_alpine']:>8.3f}{r['dproj']:>14.2f}{r['mode']:>16.3f}")
        print("-" * 96)
    print("d(proj,real)   = distance from the projected point to the nearest REAL anchor")
    print("                 (over all M, not just the K selected). Lower is better.")
    print("repair mode ok = fraction of repairs that land on the anchor which was")
    print("                 actually nearest in weight space.")

    np.savez_compressed(OUT / "adaptive_projection_results.npz",
                        K=np.array(K_LIST),
                        eps0=np.array([res[(K, 0)]["eps"] for K in K_LIST]),
                        eps1=np.array([res[(K, 1)]["eps"] for K in K_LIST]),
                        dproj0=np.array([res[(K, 0)]["dproj"] for K in K_LIST]),
                        dproj1=np.array([res[(K, 1)]["dproj"] for K in K_LIST]),
                        mode0=np.array([res[(K, 0)]["mode"] for K in K_LIST]),
                        overlap=np.array([sel_stats[K][0] for K in K_LIST]),
                        ratio=np.array([sel_stats[K][1] for K in K_LIST]),
                        same=np.array([sel_stats[K][2] for K in K_LIST]))

    # ------------------------------------------------------------- figures --
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    Ks = np.array(K_LIST)
    ax[0].semilogx(Ks, [sel_stats[K][0] for K in K_LIST], "o-", label="overlap with weight-K")
    ax[0].semilogx(Ks, [sel_stats[K][2] for K in K_LIST], "s-", label="same-mode fraction")
    ax[0].axhline(.5, ls=":", c="k", lw=1, label="chance (mode)")
    ax[0].set_xlabel("K selected anchors"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("A. is the task selection right?"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[1].semilogx(Ks, [sel_stats[K][1] for K in K_LIST], "o-", color="tab:red")
    ax[1].axhline(1.0, ls=":", c="k", lw=1)
    ax[1].set_xlabel("K"); ax[1].set_ylabel("d(nearest selected) / d(nearest overall)")
    ax[1].set_title("B. how far the region is moved"); ax[1].grid(alpha=.3)
    for n_cuts, mk in [(0, "o-"), (1, "s--")]:
        ax[2].semilogx(Ks, [res[(K, n_cuts)]["eps"] for K in K_LIST], mk,
                       label=f"n_cuts={n_cuts}")
    ax[2].set_xlabel("K"); ax[2].set_ylabel("calibrated radius ε")
    ax[2].set_title("C. radius inflation"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.suptitle("Does adapting the region move it away from the true support?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(OUT / "s_adaptive_projection.png", dpi=150)
    print(f"\nfigure -> {OUT}/s_adaptive_projection.png")
    return res, sel_stats


if __name__ == "__main__":
    main()
