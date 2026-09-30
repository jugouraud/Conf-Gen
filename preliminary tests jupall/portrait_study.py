"""Where does the filter's boundary sit on the person <-> landscape continuum?

"Do these filters reject portraits?" is under-specified: the safe set contains
human ARTEFACTS (dock, boats, lighthouse, campsite, trail) but no human
FIGURES. So the question is really where the decision boundary falls as a
prompt moves from pure portrait to landscape-with-a-speck. Four equal-step
tiers walk that continuum:

    P1 studio                pure portrait, zero nature content
    P2 nature background     person dominant, nature blurred behind
    P3 environmental         person and setting both prominent
    P4 figure in landscape   landscape dominant, person incidental

Reported alongside the original tiers so the portrait gradient can be read
against the degenerate / subtle scale already established.

    python portrait_study.py
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
from wasserstein import conformal_radius, score_order1, score_orderinf, whitening_map
from ecf import build_safe_set, ecf_score
from sweep_reweighting import masses

OUT = Path(__file__).parent / "outputs"
EPS, M_REF, K_CAL, M_PCA, N_SPLITS = 0.05, 60, 40, 32, 150
K_INF, K_KNN = 5, 3

GRADIENT = ["P1_studio", "P2_nature_bg", "P3_environmental", "P4_figure_in_landscape"]
CONTEXT = ["degenerate", "subtle", "borderline"]
SHORT = {"P1_studio": "P1 studio", "P2_nature_bg": "P2 nature bg",
         "P3_environmental": "P3 environmental", "P4_figure_in_landscape": "P4 figure in land."}


def main():
    theta, labels, Dtask, prompts = dc.load()
    m = dc.masks(labels)
    safe_idx = np.flatnonzero(m["safe"])
    tiers = {t: np.flatnonzero(m[t]) for t in CONTEXT + GRADIENT}

    names = ["ECF", "s_(d,1)", "s_(d,inf)", "g_(d,inf) K=5", "g_(d,1) knn-3"]
    rate = {k: {t: [] for t in tiers} for k in names}
    cover = {k: [] for k in names}
    per_prompt = {k: np.zeros(len(theta)) for k in names}
    rng = np.random.default_rng(0)

    for _ in range(N_SPLITS):
        p = rng.permutation(len(safe_idx))
        ref_g = safe_idx[p[:M_REF]]
        cal_g = safe_idx[p[M_REF:M_REF + K_CAL]]
        tst_g = safe_idx[p[M_REF + K_CAL:]]

        wm = whitening_map(theta[ref_g], m=M_PCA)
        Rw = wm(theta[ref_g])
        ss = build_safe_set(torch.tensor(theta[ref_g]), torch.tensor(theta[cal_g]),
                            epsilon=EPS, m=M_PCA)

        rows = np.concatenate([cal_g, tst_g] + [tiers[t] for t in tiers])
        Zw = wm(theta[rows])
        Dt = Dtask[rows][:, p[:M_REF]]
        sel = np.argsort(Dt, axis=1)[:, :K_INF]
        Wk = masses(Dt, "knn", knn=K_KNN, rescaling_order=1.0)

        S = {
            "ECF": ecf_score(torch.tensor(theta[rows]), ss).numpy(),
            "s_(d,1)": score_order1(Zw, Rw),
            "s_(d,inf)": score_orderinf(Zw, Rw, 0),
            "g_(d,inf) K=5": np.array([score_orderinf(Zw[i:i + 1], Rw[sel[i]], 0)[0]
                                       for i in range(len(Zw))]),
            "g_(d,1) knn-3": (cdist(Zw, Rw) * Wk).sum(1) / (M_REF + 1),
        }
        nc, nt = len(cal_g), len(tst_g)
        for k, s in S.items():
            r = conformal_radius(s[:nc], EPS)
            cover[k].append(np.mean(s[nc:nc + nt] <= r))
            per_prompt[k][rows] += (s > r)
            off = nc + nt
            for t, idx in tiers.items():
                rate[k][t].append(np.mean(s[off:off + len(idx)] > r))
                off += len(idx)

    exact = np.ceil((K_CAL + 1) * (1 - EPS)) / (K_CAL + 1)
    print(f"\n{'='*94}\nBLOCK RATE ALONG THE PERSON <-> LANDSCAPE CONTINUUM"
          f"   ({N_SPLITS} splits, eps={EPS}, CP target {exact:.4f})\n{'='*94}")
    hdr = f"{'estimator':<16}{'safe cov':>9}" + "".join(f"{SHORT[t]:>18}" for t in GRADIENT)
    print(hdr)
    print("-" * 94)
    for k in names:
        print(f"{k:<16}{np.mean(cover[k]):>9.3f}"
              + "".join(f"{np.mean(rate[k][t]):>18.3f}" for t in GRADIENT))
    print("-" * 94)
    print(f"{'for reference:':<16}{'':>9}" + "".join(f"{t:>18}" for t in CONTEXT))
    for k in names:
        print(f"{k:<16}{'':>9}" + "".join(f"{np.mean(rate[k][t]):>18.3f}" for t in CONTEXT))

    # where does each estimator cross 0.5?
    print(f"\n{'='*94}\nBOUNDARY LOCATION (first tier along the gradient with block rate < 0.5)"
          f"\n{'='*94}")
    for k in names:
        vals = [np.mean(rate[k][t]) for t in GRADIENT]
        cross = next((SHORT[GRADIENT[i]] for i, v in enumerate(vals) if v < 0.5), "none — all blocked")
        print(f"  {k:<16} blocks P1..P4 at {['%.2f' % v for v in vals]}   -> lets through from: {cross}")

    # normalise per-prompt counts and show the hardest P3/P4 cases
    seen = np.zeros(len(theta))
    r2 = np.random.default_rng(0)
    for _ in range(N_SPLITS):
        q = r2.permutation(len(safe_idx))
        seen[safe_idx[q[M_REF:]]] += 1
    for idx in tiers.values():
        seen[idx] += N_SPLITS
    pr_rate = {k: np.divide(v, np.maximum(seen, 1)) for k, v in per_prompt.items()}

    best = "g_(d,inf) K=5"
    for t in ["P3_environmental", "P4_figure_in_landscape"]:
        idx = tiers[t]
        v = pr_rate[best][idx]
        o = np.argsort(v)
        print(f"\n{SHORT[t]} — per prompt under {best} (block rate):")
        for i in o:
            print(f"   {v[i]:>5.2f}  {prompts[idx][i][:64]}")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(GRADIENT))
    for k, mk in zip(names, ["o-", "s-", "^-", "D-", "v-"]):
        ax.plot(x, [np.mean(rate[k][t]) for t in GRADIENT], mk, label=k, lw=1.8, ms=6)
    ax.axhline(.5, ls=":", c="k", lw=1)
    ax.axhspan(0, .5, color="tab:red", alpha=.05)
    ax.text(3.02, .04, "mostly accepted", fontsize=8, color="tab:red", ha="right")
    ax.set_xticks(x, [SHORT[t] for t in GRADIENT])
    ax.set_xlabel("← person dominant                                     landscape dominant →")
    ax.set_ylabel("fraction blocked")
    ax.set_ylim(-.02, 1.05); ax.grid(alpha=.3); ax.legend(fontsize=9)
    ax.set_title(f"Where the boundary falls on the person↔landscape continuum (ε={EPS})")
    fig.tight_layout(); fig.savefig(OUT / "w_portrait_gradient.png", dpi=150); plt.close(fig)
    print(f"\nfigure -> {OUT}/w_portrait_gradient.png")


if __name__ == "__main__":
    main()
