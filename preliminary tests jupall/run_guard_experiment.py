"""Evaluate the conformal filter deployed as a PROJECTION step.

For every (score, lift) pair:

  validity     coverage on held-out safe prompts -- must match split CP's exact
               expectation ceil((K+1)(1-eps))/(K+1)
  reach        fraction of each tier that falls outside the safe set and is
               therefore moved by the projection
  displacement how far the projection has to move the conditioning, relative to
               its norm -- the cost the generator pays
  post-condition  after guard(), EVERYTHING is inside the safe set (this is the
               property a projection filter buys over an accept/reject gate)

    python run_guard_experiment.py
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

import encode_sequences as es
from conformal_guard import ConformalGuard, SCORES, LIFTS

OUT = Path(__file__).parent / "outputs"
EPS = 0.05
N_SAFE_REF = 60          # reference half of the safe prompts
TIERS = ["degenerate", "subtle", "borderline",
         "P1_studio", "P2_nature_bg", "P3_environmental", "P4_figure_in_landscape"]
SHOW = ["degenerate", "subtle", "P1_studio", "P3_environmental", "P4_figure_in_landscape"]


def evaluate(seq, eos, labels, score, lift, n_test=20, seed=0):
    safe = np.flatnonzero(labels == "safe")
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(safe))
    fit_i, tst_i = safe[p[:-n_test]], safe[p[-n_test:]]

    guard = ConformalGuard.fit(seq[fit_i], eos[fit_i], epsilon=EPS, score=score,
                               lift=lift, n_ref=N_SAFE_REF, seed=seed)
    reg = guard.region
    row = {"score": score, "lift": lift, "radius": guard.radius,
           "region": type(reg).__name__, "convex": reg.is_convex}

    def probe(idx):
        out, info = guard.guard(seq[idx], eos[idx])
        rel = info["displacement"] / np.linalg.norm(
            seq[idx].reshape(len(idx), -1), axis=1)
        after = np.atleast_1d(info["score_after"])
        return (np.asarray(info["was_outside"]), rel,
                bool(np.all(after <= reg.radius + 1e-6)))

    out_t, rel_t, ok_t = probe(tst_i)
    row["coverage"] = 1.0 - out_t.mean()
    row["post_ok"] = ok_t
    row["disp_safe"] = float(rel_t.mean())
    for t in TIERS:
        idx = np.flatnonzero(labels == t)
        o, rel, ok = probe(idx)
        row[t] = float(o.mean())
        row[t + "_disp"] = float(rel[o].mean()) if o.any() else 0.0
        row["post_ok"] = row["post_ok"] and ok
    return row


def main():
    seq, eos, labels, prompts = es.load()
    seq = seq.astype(np.float64)
    rows = []
    print(f"eps={EPS}  reference={N_SAFE_REF}  calibration={120 - 20 - N_SAFE_REF}  test=20")
    print("fitting guards ...")
    for score in SCORES:
        for lift in LIFTS:
            try:
                rows.append(evaluate(seq, eos, labels, score, lift))
                print(f"  ok  {score:<18} {lift}")
            except Exception as e:
                print(f"  --  {score:<18} {lift}: {type(e).__name__}: {e}")

    K = 120 - 20 - N_SAFE_REF
    exact = np.ceil((K + 1) * (1 - EPS)) / (K + 1)
    print(f"\n{'='*104}\nPROJECTION FILTER: fraction of each tier that is MOVED"
          f"   (split-CP coverage target {exact:.4f})\n{'='*104}")
    hdr = (f"{'score':<18}{'lift':<11}{'region':<22}{'cov':>7}{'post':>6}"
           + "".join(f"{t[:9]:>10}" for t in SHOW))
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['score']:<18}{r['lift']:<11}{r['region']:<22}{r['coverage']:>7.2f}"
              f"{'OK' if r['post_ok'] else 'FAIL':>6}"
              + "".join(f"{r[t]:>10.2f}" for t in SHOW))

    print(f"\n{'='*104}\nRELATIVE DISPLACEMENT ||proj - z|| / ||z||  (mean over the prompts "
          f"that were moved)\n{'='*104}")
    print(hdr[:58] + "".join(f"{t[:9]:>10}" for t in SHOW))
    for r in rows:
        print(f"{r['score']:<18}{r['lift']:<11}{r['region']:<22}{r['disp_safe']:>7.3f}"
              f"{'':>6}" + "".join(f"{r[t + '_disp']:>10.3f}" for t in SHOW))

    # ---- figure: reach vs displacement, per lift ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, lift in zip(axes, LIFTS):
        sub = [r for r in rows if r["lift"] == lift]
        if not sub:
            ax.set_title(f"{lift} (unavailable)"); continue
        x = np.arange(len(SHOW))
        for r in sub:
            ax.plot(x, [r[t] for t in SHOW], "o-", label=r["score"], lw=1.6, ms=5)
        ax.set_xticks(x, [t[:10] for t in SHOW], rotation=30, ha="right", fontsize=7)
        ax.set_title(f"lift = {lift}"); ax.grid(alpha=.3); ax.set_ylim(-.03, 1.05)
    axes[0].set_ylabel("fraction moved by the projection")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"Conformal projection filter: what each (score, lift) moves (ε={EPS})")
    fig.tight_layout(); fig.savefig(OUT / "w_guard_reach.png", dpi=150); plt.close(fig)
    print(f"\nfigure -> {OUT}/w_guard_reach.png")
    return rows


if __name__ == "__main__":
    main()
