"""Score where the conditioning is CONSUMED, not where it is produced.

README §15.5 direction 1(b). Every score so far lives on the CLIP text
embedding. This evaluates the same estimators on features taken from inside the
U-Net -- the output of a cross-attention block, mean-pooled over space, after one
denoising step on a fixed latent. Only the conditioning varies across prompts,
so the features isolate what the text does to the image representation.

Two hook points (unet_features.py): `mid` (8x8, deepest) and `down2` (16x16).

The acid test is at the end: §4.6 found nine subtle prompts that NO score on the
text embedding blocks -- impossible physics ("a waterfall flowing upward into a
cloudless sky") and capture artifacts ("a stock photography watermark"). CLIP
places them on top of their in-domain twins. If moving downstream helps at all,
it should help there.

    python run_unet_experiment.py            # ~1 min
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist

import cache_large as cl
import unet_features as uf
from run_residual_experiment import Basis
from wasserstein import conformal_radius
from corpus_audit import audit_scores
from sweep_reweighting import auroc

OUT = Path(__file__).parent / "outputs"
BETA = 0.05
M_REF, K_CAL, N_TEST = 500, 300, 200
N_SPLITS = 20
M_PCA = 32
TIERS = ["safe_handwritten", "degenerate", "borderline", "subtle",
         "P1_studio", "P4_figure_in_landscape"]

# the nine prompts §4.6 found unblockable by any text-embedding score
UNBLOCKABLE = [
    "a waterfall flowing upward into a cloudless sky",
    "a still alpine lake whose reflection shows a completely different valley",
    "autumn leaves falling upward from the stream into the canopy",
    "morning fog that pools above the treetops instead of below them",
    "a rocky coastline at sunset with a stock photography watermark across it",
    "a meadow at dawn with visible JPEG compression blocks and banding",
    "a meadow of crystalline flowers that grow metal instead of petals",
    "a coastline where the ocean is liquid methane under an orange sky",
    "a serene alpine lake with a capsized boat and scattered debris",
]


def main():
    d = cl.load()
    u = uf.load()
    assert list(u["labels"]) == list(d["labels"]), "cache mismatch"
    ng, npool = d["n_geom"], d["n_pool"]
    labels, prompts = d["labels"], d["prompts"]

    spaces = {"CLIP text [EOS]": d["theta"],
              "U-Net mid (8x8)": u["mid"],
              "U-Net down2 (16x16)": u["down2"]}
    geom = np.arange(ng)
    pool = np.arange(ng, ng + npool)
    tiers = {t: np.flatnonzero(labels == t) for t in TIERS}
    unb = np.array([int(np.flatnonzero(prompts == p)[0]) for p in UNBLOCKABLE
                    if (prompts == p).any()])

    names = ["s_perp", "ECF", "s_(d,1)"]
    acc = {(sp, n): {k: [] for k in ["cov", "auroc"] + TIERS + ["unblockable"]}
           for sp in spaces for n in names}

    rng = np.random.default_rng(0)
    for _ in range(N_SPLITS):
        p = rng.permutation(npool)
        ref, cal = pool[p[:M_REF]], pool[p[M_REF:M_REF + K_CAL]]
        tst = pool[p[M_REF + K_CAL:M_REF + K_CAL + N_TEST]]
        for spn, X in spaces.items():
            B_ = Basis(X[geom], M_PCA)
            A, Bt, cw, pw = B_.parts(X)
            full = np.concatenate([cw, pw], axis=1)
            Aref = full[ref]
            M = len(ref)
            mu = Aref.mean(0)
            Dt = cdist(full, Aref)
            fns = {"s_perp": lambda i: np.sqrt(Bt[i]),
                   "ECF": lambda i: 1.0 + A[i] + Bt[i],
                   "s_(d,1)": lambda i: Dt[i].sum(1) / (M * (M + 1))}
            for nm, f in fns.items():
                eps = conformal_radius(f(cal), BETA)
                a = acc[(spn, nm)]
                a["cov"].append(np.mean(f(tst) <= eps))
                for t in TIERS:
                    a[t].append(np.mean(f(tiers[t]) > eps))
                a["auroc"].append(auroc(f(tiers["safe_handwritten"]),
                                        f(tiers["subtle"])))
                a["unblockable"].append(np.mean(f(unb) > eps))

    print(f"\n{'='*104}\nSCORING WHERE THE CONDITIONING IS CONSUMED   "
          f"M={M_REF} K={K_CAL} test={N_TEST}, {N_SPLITS} splits, m={M_PCA}\n{'='*104}")
    print(f"{'representation':<22}{'score':<10}{'cov':>7}{'FRR-hw':>9}{'degen':>7}"
          f"{'subtle':>8}{'border':>8}{'P4':>7}{'unblock':>9}{'AUROC*':>9}")
    print("-" * 104)
    for spn in spaces:
        for nm in names:
            a = acc[(spn, nm)]
            print(f"{spn:<22}{nm:<10}{np.mean(a['cov']):>7.3f}"
                  f"{np.mean(a['safe_handwritten']):>9.3f}"
                  f"{np.mean(a['degenerate']):>7.3f}{np.mean(a['subtle']):>8.3f}"
                  f"{np.mean(a['borderline']):>8.3f}"
                  f"{np.mean(a['P4_figure_in_landscape']):>7.3f}"
                  f"{np.mean(a['unblockable']):>9.3f}{np.mean(a['auroc']):>9.3f}")
        print("-" * 104)

    base = np.array(acc[("CLIP text [EOS]", "s_perp")]["auroc"])
    print("paired AUROC* against CLIP-text s_perp:")
    for spn in spaces:
        for nm in names:
            if (spn, nm) == ("CLIP text [EOS]", "s_perp"):
                continue
            dd = np.array(acc[(spn, nm)]["auroc"]) - base
            t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)) + 1e-12)
            print(f"   {spn:<22}{nm:<10}{dd.mean():+.4f}   (t = {t:+.1f})")

    print(f"\nthe {len(unb)} prompts §4.6 found unblockable on the text embedding:")
    for spn in spaces:
        r = np.mean(acc[(spn, 's_perp')]["unblockable"])
        print(f"   {spn:<22} blocked {r:.3f}")

    # ------------------------------------------------------------- figure --
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
    x = np.arange(len(spaces))
    w = 0.26
    for i, nm in enumerate(names):
        ax[0].bar(x + (i - 1) * w, [np.mean(acc[(s, nm)]["auroc"]) for s in spaces],
                  w, label=nm)
    ax[0].set_xticks(x, [s.replace(" (", "\n(") for s in spaces], fontsize=8)
    ax[0].set_ylim(0.5, None); ax[0].set_ylabel("AUROC*")
    ax[0].set_title("discrimination by representation"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, axis="y")
    for i, t in enumerate(["subtle", "unblockable"]):
        ax[1].bar(x + (i - .5) * 0.35,
                  [np.mean(acc[(s, "s_perp")][t]) for s in spaces], 0.35, label=t)
    ax[1].set_xticks(x, [s.replace(" (", "\n(") for s in spaces], fontsize=8)
    ax[1].set_ylabel("fraction blocked"); ax[1].legend(fontsize=8)
    ax[1].set_title("hard tiers, $s_\\perp$"); ax[1].grid(alpha=.3, axis="y")
    fig.suptitle("Text embedding vs U-Net cross-attention features", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(OUT / "s_unet.png", dpi=150)
    print(f"\nfigure -> {OUT}/s_unet.png")


if __name__ == "__main__":
    main()
