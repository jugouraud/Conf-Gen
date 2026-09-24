"""Blackbox meta-learning score vs the Wasserstein / ECF scores.

g_d^BB (adaptation.tex eq:bb_score) trains a small network psi_omega to predict
the pooled embedding from the token bag, and scores by the residual
|| theta - psi(tau) ||. This evaluates it on exactly the splits section 11 used,
so the numbers are directly comparable, and audits it with the section 12
machinery.

Reported for every estimator:

    cov       coverage on held-out prompts from the CALIBRATION corpus
    FRR-hw    false rejection of the 120 hand-written safe prompts -- the
              corpus-representativeness check
    AUROC*    AUROC(hand-written safe vs subtle): both hand-written prose, so
              the machine-generated corpus cancels. The comparison metric.

    python run_bb_experiment.py            # ~3 min
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

import cache_large as cl
from wasserstein import whitening_map, conformal_radius
from bb_model import fit_bb
from corpus_audit import audit_scores
from sweep_reweighting import auroc

OUT = Path(__file__).parent / "outputs"
BETA = 0.05
N_TRAIN, N_CAL, N_TEST = 500, 300, 200
N_SPLITS = 10
TIERS = ["safe_handwritten", "degenerate", "borderline", "subtle",
         "P1_studio", "P3_environmental", "P4_figure_in_landscape"]


def main():
    d = cl.load()
    ng, npool = d["n_geom"], d["n_pool"]
    theta, clouds, labels = d["theta"], d["clouds"], d["labels"]

    # metric: fitted on the geometry block, disjoint from every split
    th_wm = whitening_map(theta[d["geom"]], m=32)
    tok_wm = whitening_map(np.vstack([clouds[i] for i in d["geom"]]), m=32)
    Tw = th_wm(theta)

    pool = np.arange(ng, ng + npool)
    tiers = {t: np.flatnonzero(labels == t) for t in TIERS}

    # `residual only` is the null baseline that matters: no anchors, no network,
    # just how much of the whitened embedding lies OUTSIDE the safe corpus's
    # top-m PCA subspace. If the elaborate scores cannot beat it, they are not
    # earning their cost.
    names = ["residual only", "ECF (ellipsoid)", "s_(d,1) uniform",
             "g^(0,5)_(d,inf)", "g_d^BB (network)", "g_d^BB in-subspace"]
    acc = {n: {k: [] for k in ["cov", "auroc_hw", "eps"] + TIERS} for n in names}
    keep = {}

    for sp in range(N_SPLITS):
        rng = np.random.default_rng(sp)
        p = rng.permutation(npool)
        tr, cal, tst = pool[p[:N_TRAIN]], pool[p[N_TRAIN:N_TRAIN + N_CAL]], \
            pool[p[N_TRAIN + N_CAL:N_TRAIN + N_CAL + N_TEST]]

        print(f"[split {sp + 1}/{N_SPLITS}] training psi_omega on {len(tr)} safe prompts ...")
        t0 = time.time()
        bb = fit_bb(theta[tr], [clouds[i] for i in tr], tok_wm, th_wm,
                    seed=sp, verbose=(sp == 0))
        train_s = time.time() - t0

        # --- score functions, all on the same reference/training split ---
        A = Tw[tr]
        mu = A.mean(0)
        M = len(tr)
        Dt = cdist(Tw, A)
        sel = np.argsort(d["D_wht"][:, p[:N_TRAIN]], axis=1)     # task-nearest anchors

        from run_scaling_experiment import bottleneck

        Vth = th_wm.V

        def s_resid(idx):
            w = Tw[idx]
            c = w @ Vth
            return np.sqrt(np.maximum((w ** 2).sum(1) - (c ** 2).sum(1), 0.0))

        def s_ecf(idx):
            return 1.0 + ((Tw[idx] - mu) ** 2).sum(1)

        def s_w1(idx):
            return Dt[idx][:, :].sum(1) / (M * (M + 1)) if False else \
                Dt[np.ix_(idx, np.arange(len(A)))].sum(1) / (M * (M + 1))

        def s_winf(idx):
            out = np.empty(len(idx))
            for i, g in enumerate(idx):
                a = sel[g - ng, :5]
                sub = np.empty((6, 6))
                sub[:5, :5] = cdist(A[a], A[a])
                dd = np.linalg.norm(Tw[g] - A[a], axis=1)
                sub[5, :5] = dd
                sub[:5, 5] = dd
                sub[5, 5] = 0.0
                out[i] = bottleneck(sub)
            return out

        def s_bb(idx):
            return bb(theta[idx], [clouds[i] for i in idx])

        def s_bb_in(idx):
            return bb.in_subspace_error(theta[idx], [clouds[i] for i in idx])

        for nm, fn in zip(names, [s_resid, s_ecf, s_w1, s_winf, s_bb, s_bb_in]):
            eps = conformal_radius(fn(cal), BETA)
            s_t = fn(tst)
            a = acc[nm]
            a["eps"].append(eps)
            a["cov"].append(np.mean(s_t <= eps))
            for t in TIERS:
                a[t].append(np.mean(fn(tiers[t]) > eps))
            a["auroc_hw"].append(auroc(fn(tiers["safe_handwritten"]), fn(tiers["subtle"])))
            if sp == 0:
                keep[nm] = dict(cal=fn(cal), tst=s_t, eps=eps,
                                **{t: fn(tiers[t]) for t in TIERS})
        if sp == 0:
            keep["_meta"] = dict(params=bb.net.n_params, train_s=train_s)
            t0 = time.time()
            for _ in range(50):
                bb(theta[tst[:1]], [clouds[tst[0]]])
            keep["_meta"]["infer_ms"] = (time.time() - t0) / 50 * 1000

    # ------------------------------------------------------------ report --
    print(f"\n{'='*100}\nBLACKBOX SCORE vs THE REFERENCE-SET SCORES   "
          f"train={N_TRAIN} cal={N_CAL} test={N_TEST}, {N_SPLITS} splits, "
          f"beta={BETA}\n{'='*100}")
    print(f"{'estimator':<20}{'cov':>7}{'FRR-hw':>9}{'degen':>7}{'subtle':>8}"
          f"{'border':>8}{'P1':>7}{'P4':>7}{'AUROC*':>9}")
    print("-" * 100)
    for nm in names:
        a = acc[nm]
        print(f"{nm:<20}{np.mean(a['cov']):>7.3f}{np.mean(a['safe_handwritten']):>9.3f}"
              f"{np.mean(a['degenerate']):>7.3f}{np.mean(a['subtle']):>8.3f}"
              f"{np.mean(a['borderline']):>8.3f}{np.mean(a['P1_studio']):>7.3f}"
              f"{np.mean(a['P4_figure_in_landscape']):>7.3f}"
              f"{np.mean(a['auroc_hw']):>9.3f}")
    print("-" * 100)
    m = keep["_meta"]
    print(f"psi_omega: {m['params']} parameters ({100 * m['params'] / 859.5e6:.4f}% of the "
          f"U-Net), {m['train_s']:.1f}s to train, {m['infer_ms']:.2f} ms per prompt")
    print("no reference set is needed at inference: the corpus lives in the weights")

    # paired comparison against the best reference-set score
    base = np.array(acc["residual only"]["auroc_hw"])
    print("\npaired AUROC* against the `residual only` baseline:")
    for nm in names:
        if nm == "residual only":
            continue
        dd = np.array(acc[nm]["auroc_hw"]) - base
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)) + 1e-12)
        print(f"  {nm:<22} {dd.mean():+.4f}   (t = {t:+.1f})")

    # ------------------------------------------------------------- audit --
    print(f"\n{'='*100}\nCORPUS AUDIT of the blackbox score\n{'='*100}")
    r = audit_scores(keep["g_d^BB (network)"]["cal"],
                     keep["g_d^BB (network)"]["safe_handwritten"], beta=BETA)
    print(r)

    # ------------------------------------------------------------ figure --
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(names))
    w = 0.27
    for i, (t, c) in enumerate([("degenerate", "tab:green"),
                                ("subtle", "tab:red"), ("safe_handwritten", "tab:purple")]):
        lab = "hand-written FALSE rejection" if t == "safe_handwritten" else t
        axes[0].bar(x + (i - 1) * w, [np.mean(acc[n][t]) for n in names], w,
                    color=c, label=lab)
    axes[0].plot(x, [np.mean(acc[n]["cov"]) for n in names], "kd--", ms=6,
                 label="coverage (own corpus)")
    axes[0].axhline(1 - BETA, ls=":", c="k", lw=1)
    axes[0].set_xticks(x, names, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.08); axes[0].set_ylabel("rate")
    axes[0].set_title("detection and false rejection"); axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=.3, axis="y")

    vals = [np.mean(acc[n]["auroc_hw"]) for n in names]
    err = [np.std(acc[n]["auroc_hw"], ddof=1) / np.sqrt(N_SPLITS) for n in names]
    cols = ["tab:grey"] + ["tab:blue"] * 3 + ["tab:orange"] * 2
    axes[1].bar(x, vals, yerr=err, color=cols)
    axes[1].axhline(vals[0], ls="--", c="k", lw=1.2)
    axes[1].set_xticks(x, names, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylim(0.5, max(vals) * 1.08)
    axes[1].set_ylabel("AUROC(hand-written safe vs subtle)")
    axes[1].set_title("discrimination, free of the corpus artefact")
    axes[1].grid(alpha=.3, axis="y")
    fig.suptitle(f"Blackbox meta-model $g_d^{{BB}}$ ({m['params']:,} parameters) "
                 f"vs the reference-set scores", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(OUT / "s_blackbox.png", dpi=150)
    print(f"\nfigure -> {OUT}/s_blackbox.png")


if __name__ == "__main__":
    main()
