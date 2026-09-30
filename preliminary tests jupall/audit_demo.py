"""Run the corpus audit on the two real corpora, in both directions.

Three cases, all with the same score function, anchors and level:

  1  calibrate on the GENERATED corpus, audit against HAND-WRITTEN prompts
     -- the failure the audit exists to catch
  2  calibrate on GENERATED, audit against held-out GENERATED
     -- the control: the audit must stay silent when nothing is wrong
  3  calibrate on HAND-WRITTEN, audit against GENERATED
     -- the reverse direction: a broader corpus is safe for a narrower
        deployment, so this should pass

    python audit_demo.py
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
from wasserstein import whitening_map, conformal_radius
from corpus_audit import audit_scores

OUT = Path(__file__).parent / "outputs"
BETA = 0.05


def main():
    d = cl.load()
    ng, npool = d["n_geom"], d["n_pool"]
    wm = whitening_map(d["theta"][d["geom"]], m=32)
    Tw = wm(d["theta"])
    pool = np.arange(ng, ng + npool)
    hw = np.flatnonzero(d["labels"] == "safe_handwritten")

    rng = np.random.default_rng(0)
    p = rng.permutation(npool)
    gen_ref, gen_cal, gen_tst = pool[p[:500]], pool[p[500:800]], pool[p[800:920]]
    hw_p = rng.permutation(len(hw))
    hw_ref, hw_cal = hw[hw_p[:60]], hw[hw_p[60:]]

    def s_of(idx, ref):
        M = len(ref)
        return cdist(Tw[idx], Tw[ref]).sum(1) / (M * (M + 1))

    cases = [
        ("1. calibrate GENERATED\n   audit HAND-WRITTEN",
         s_of(gen_cal, gen_ref), s_of(hw, gen_ref), "the failure to catch"),
        ("2. calibrate GENERATED\n   audit GENERATED",
         s_of(gen_cal, gen_ref), s_of(gen_tst, gen_ref), "control: same source"),
        ("3. calibrate HAND-WRITTEN\n   audit GENERATED",
         s_of(hw_cal, hw_ref), s_of(gen_tst, hw_ref), "reverse direction"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (title, cal, aud, note) in zip(axes, cases):
        r = audit_scores(cal, aud, beta=BETA, n_perm=5000)
        print("=" * 78)
        print(f"{title.replace(chr(10), '  ')}   [{note}]")
        print("=" * 78)
        print(r)
        print()

        lo = min(cal.min(), aud.min())
        hi = max(np.percentile(cal, 99.5), np.percentile(aud, 99.5))
        bins = np.linspace(lo, hi, 45)
        ax.hist(cal, bins=bins, alpha=.65, color="tab:blue", density=True,
                label=f"calibration (n={len(cal)})")
        ax.hist(aud, bins=bins, alpha=.65, color="tab:red", density=True,
                label=f"external holdout (n={len(aud)})")
        ax.axvline(r.radius, color="k", ls="--", lw=1.6,
                   label=f"radius (β={BETA})")
        col = {"pass": "tab:green", "warn": "tab:orange", "fail": "tab:red"}[r.verdict]
        ax.set_title(f"{title}\n{r.verdict.upper()}   FRR={r.frr:.3f}   p={r.p_value:.2g}",
                     fontsize=10, color=col)
        ax.set_xlabel("nonconformity score"); ax.set_yticks([])
        ax.legend(fontsize=7.5); ax.grid(alpha=.3, axis="x")
    axes[0].set_ylabel("density")
    fig.suptitle("Corpus audit: coverage on held-out calibration data cannot see any of this",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(OUT / "s_corpus_audit.png", dpi=150)
    print(f"figure -> {OUT}/s_corpus_audit.png")


if __name__ == "__main__":
    main()
