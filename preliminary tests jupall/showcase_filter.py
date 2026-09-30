"""Per-prompt showcase: what each filter actually blocks, and where they differ.

The aggregate rates in run_wasserstein_experiment.py say the Wasserstein
estimators block ~3 points more of the SUBTLE tier than the ECF at equal
coverage. This script says WHICH prompts that is, by computing a per-prompt
block rate over many random calibration splits and diffing the estimators.

    python showcase_filter.py
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
from wasserstein import conformal_radius, score_orderinf, whitening_map, LowCardOrderInf
from ecf import build_safe_set, ecf_score
from sweep_reweighting import masses

OUT = Path(__file__).parent / "outputs"
EPS, M_REF, K_CAL, M_PCA, N_SPLITS = 0.05, 60, 40, 32, 150
K_INF, K_KNN = 5, 3


def main():
    theta, labels, Dtask, prompts = dc.load()
    m = dc.masks(labels)
    safe_idx = np.flatnonzero(m["safe"])
    tiers = {t: np.flatnonzero(m[t]) for t in ["degenerate", "borderline", "subtle"]}

    # per-prompt block counts, per estimator
    names = ["ECF", "s_(d,inf)", "g_(d,inf) K=5", "g_(d,1) knn-3"]
    blocked = {k: np.zeros(len(theta)) for k in names}
    covered = {k: [] for k in names}
    rng = np.random.default_rng(0)

    for _ in range(N_SPLITS):
        p = rng.permutation(len(safe_idx))
        ref_l, cal_l, tst_l = p[:M_REF], p[M_REF:M_REF + K_CAL], p[M_REF + K_CAL:]
        ref_g, cal_g, tst_g = safe_idx[ref_l], safe_idx[cal_l], safe_idx[tst_l]

        wm = whitening_map(theta[ref_g], m=M_PCA)
        Rw = wm(theta[ref_g])
        ss = build_safe_set(torch.tensor(theta[ref_g]), torch.tensor(theta[cal_g]),
                            epsilon=EPS, m=M_PCA)

        rows = np.concatenate([cal_g, tst_g] + [tiers[t] for t in tiers])
        Zw = wm(theta[rows])
        Dt = Dtask[rows][:, ref_l]
        sel = np.argsort(Dt, axis=1)[:, :K_INF]
        Wk = masses(Dt, "knn", knn=K_KNN, rescaling_order=1.0)

        S = {
            "ECF": ecf_score(torch.tensor(theta[rows]), ss).numpy(),
            "s_(d,inf)": score_orderinf(Zw, Rw, 0),
            "g_(d,inf) K=5": np.array([score_orderinf(Zw[i:i + 1], Rw[sel[i]], 0)[0]
                                       for i in range(len(Zw))]),
            "g_(d,1) knn-3": (cdist(Zw, Rw) * Wk).sum(1) / (M_REF + 1),
        }
        n_cal = len(cal_g)
        for k, s in S.items():
            r = conformal_radius(s[:n_cal], EPS)
            blocked[k][rows] += (s > r)
            covered[k].append(np.mean(s[n_cal:n_cal + len(tst_g)] <= r))

    # normalise: safe prompts appear as calibration OR test, tiers appear every split
    seen = np.zeros(len(theta))
    rng2 = np.random.default_rng(0)
    for _ in range(N_SPLITS):
        p = rng2.permutation(len(safe_idx))
        seen[safe_idx[p[M_REF:]]] += 1
    for t in tiers.values():
        seen[t] += N_SPLITS
    rate = {k: np.divide(v, np.maximum(seen, 1)) for k, v in blocked.items()}

    print(f"\n{'='*78}\nPER-TIER BLOCK RATE   ({N_SPLITS} splits, eps={EPS}, "
          f"exact CP target {np.ceil((K_CAL+1)*(1-EPS))/(K_CAL+1):.4f})\n{'='*78}")
    print(f"{'estimator':<16}{'safe cover':>11}{'degenerate':>12}{'borderline':>12}{'subtle':>9}")
    for k in names:
        print(f"{k:<16}{np.mean(covered[k]):>11.3f}"
              + "".join(f"{np.mean(rate[k][tiers[t]]):>12.3f}"
                        for t in ["degenerate", "borderline"])
              + f"{np.mean(rate[k][tiers['subtle']]):>9.3f}")

    # ---- which SUBTLE prompts, and where the estimators disagree ----
    sub = tiers["subtle"]
    base, best = rate["ECF"][sub], rate["g_(d,inf) K=5"][sub]
    order = np.argsort(-(best - base))

    print(f"\n{'='*78}\nSUBTLE TIER, per prompt "
          f"(block rate: 1.00 = always blocked, 0.00 = always slips through)\n{'='*78}")
    print(f"{'ECF':>6}{'g_inf':>7}{'delta':>7}  prompt")
    print("-" * 78)
    for i in order:
        d = best[i] - base[i]
        mark = " <<" if d >= 0.10 else (" >>" if d <= -0.10 else "")
        print(f"{base[i]:>6.2f}{best[i]:>7.2f}{d:>+7.2f}  {prompts[sub][i][:56]}{mark}")

    caught = (best >= 0.5).sum()
    print("-" * 78)
    print(f"blocked by g_(d,inf) at least half the time: {caught}/{len(sub)}"
          f"   (ECF: {(base >= 0.5).sum()}/{len(sub)})")
    print("<< gained by the Wasserstein estimator   >> lost relative to ECF")

    # what always slips through, for both
    never = [prompts[sub][i] for i in range(len(sub)) if best[i] < 0.1 and base[i] < 0.1]
    if never:
        print(f"\nNEVER blocked by either ({len(never)}):")
        for s in never:
            print(f"   - {s[:70]}")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9.5, 9))
    y = np.arange(len(sub))
    ax.barh(y - .2, base[order], .4, label="ECF", color="tab:blue")
    ax.barh(y + .2, best[order], .4, label=f"g_(d,inf) K={K_INF}", color="tab:red")
    ax.set_yticks(y, [prompts[sub][i][:52] for i in order], fontsize=6.5)
    ax.axvline(.5, ls=":", c="k", lw=1)
    ax.set_xlabel("fraction of splits in which the prompt is blocked")
    ax.set_title("Subtle prompts: ECF vs adaptive order-∞", fontsize=11)
    ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "w_subtle_per_prompt.png", dpi=150)
    plt.close(fig)
    print(f"\nfigure -> {OUT}/w_subtle_per_prompt.png")


if __name__ == "__main__":
    main()
