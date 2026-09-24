"""Scaling study: 1000-prompt pool, more reweighting schemes, pooled vs sequence.

Four parts, all on the large cache (200 geometry / 1000 pool / 149 adversarial):

  A  REWEIGHTING     30 schemes for g_(d,1), at M = 500 anchors. Tests whether
                     standardised and ordinal schemes beat the manuscript's
                     inverse-power default on CLIP's narrow distance band.
  B  SCALING IN M    eps and detection as M goes 30 -> 500, against the
                     predicted eps ~ M^(-1/d_eff).
  C  POOLED vs SEQ   the same estimators with the pooled [EOS] vector replaced
                     by the token cloud and Euclidean distance replaced by W1
                     between clouds.
  D  HEAD-TO-HEAD    the survivors, at the best settings found in A-C.

Only estimators that scale are included: ECF, order-1 (uniform and adaptive),
order-inf adaptive at small K. The exact order-inf on all M anchors and the
n-cuts variants are excluded -- the first is O(M^2) per candidate with no
benefit at M = 500, the second was shown to destroy detection on a unimodal
safe set.

    python run_scaling_experiment.py            # ~12 min
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
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree

import cache_large as cl
from reweighting import SCHEMES, FAMILY, masses_by_name, effective_support
from wasserstein import conformal_radius, whitening_map
from sweep_reweighting import auroc

OUT = Path(__file__).parent / "outputs"
BETA = 0.05
M_REF, K_CAL, N_TEST = 500, 300, 200
N_SPLITS_FAST, N_SPLITS_MST = 100, 40
K_ADAPT = 5
# safe_handwritten is NOT an adversarial tier: it is the original hand-written
# safe corpus, genuinely in-domain. Its "blocked" rate is a FALSE-REJECTION rate
# and is the check on whether the generated corpus has narrowed the safe set.
TIERS = ["safe_handwritten", "degenerate", "borderline", "subtle",
         "P1_studio", "P3_environmental", "P4_figure_in_landscape"]


# --------------------------------------------------------------- machinery --

def bottleneck(Dsub):
    """MST bottleneck of a dense symmetric distance matrix (n_cuts = 0)."""
    mst = minimum_spanning_tree(csr_array(Dsub)).toarray()
    w = mst[mst > 0]
    return float(w.max()) if len(w) else 0.0


class Space:
    """One geometry: a candidate-to-anchor distance matrix plus anchor-anchor.

    `Dca[i, j]` is the distance from candidate row i to pool member j;
    `Daa[a, b]` between pool members. Both index the POOL by 0..999.
    """

    def __init__(self, name, Dca, Daa, ecf_center=None):
        self.name, self.Dca, self.Daa = name, Dca, Daa
        self.ecf_center = ecf_center

    def order1(self, rows, ref, w=None):
        D = self.Dca[np.ix_(rows, ref)]
        M = len(ref)
        if w is None:
            return D.sum(1) / (M * (M + 1))
        return (D * w).sum(1) / (M + 1)

    def orderinf_adaptive(self, rows, ref, sel_local, K):
        """g^(0,K): bottleneck on the K task-nearest anchors, per candidate."""
        out = np.empty(len(rows))
        for i, r in enumerate(rows):
            a = ref[sel_local[i, :K]]
            n = len(a)
            D = np.empty((n + 1, n + 1))
            D[:n, :n] = self.Daa[np.ix_(a, a)]
            d0 = self.Dca[r, a]
            D[n, :n] = d0
            D[:n, n] = d0
            D[n, n] = 0.0
            out[i] = bottleneck(D)
        return out


def build_spaces(d):
    """The pooled-vector space and the token-cloud space, on identical indexing."""
    ng, npool = d["n_geom"], d["n_pool"]
    theta = d["theta"]
    wm = whitening_map(theta[d["geom"]], m=32)       # fixed metric, geometry block
    Tw = wm(theta)                                    # (1349, 768)

    pool_g = np.arange(ng, ng + npool)                # global ids of the pool
    rows_g = np.concatenate([pool_g, np.arange(ng + npool, len(theta))])
    Dca_p = cdist(Tw[rows_g], Tw[pool_g])
    Daa_p = Dca_p[:npool]                             # pool-pool block

    pooled = Space("pooled [EOS]", Dca_p, Daa_p, ecf_center=Tw[pool_g])
    seq = Space("token cloud (W1)", d["D_wht"], d["D_wht"][:npool])
    return pooled, seq, rows_g


def row_index(d):
    """Map tier name -> row indices into the (1149, 1000) matrices."""
    ng, npool = d["n_geom"], d["n_pool"]
    lab = d["labels"]
    idx = {}
    for t in TIERS:
        g = np.flatnonzero(lab == t)
        idx[t] = g - ng                               # rows: pool then adv, offset ng
    return idx


def evaluate(space, ref, cal, tst, adv_rows, score_fn, beta=BETA):
    s_cal = score_fn(cal)
    eps = conformal_radius(s_cal, beta)
    s_tst = score_fn(tst)
    out = {"eps": eps, "cov": float(np.mean(s_tst <= eps))}
    for t, rows in adv_rows.items():
        s = score_fn(rows)
        out[t] = float(np.mean(s > eps))
        if t == "subtle":
            out["auroc"] = auroc(s_tst, s)
    return out


# ------------------------------------------------------- A. reweighting -----

def part_A(d, pooled, adv_rows, n_splits=N_SPLITS_FAST):
    print(f"\n{'='*104}\nA. REWEIGHTING SCHEMES for g_(d,1)   "
          f"M={M_REF} anchors, K={K_CAL} calibration, {N_TEST} test, "
          f"{n_splits} splits, beta={BETA}\n{'='*104}")
    D_task = d["D_wht"]
    rng = np.random.default_rng(0)
    acc = {n: {k: [] for k in ["cov", "auroc", "auroc_hw", "eff"] + TIERS} for n in SCHEMES}

    for _ in range(n_splits):
        p = rng.permutation(d["n_pool"])
        ref, cal, tst = p[:M_REF], p[M_REF:M_REF + K_CAL], p[M_REF + K_CAL:M_REF + K_CAL + N_TEST]
        blocks = {"cal": cal, "tst": tst, **adv_rows}
        rows = np.concatenate([blocks[k] for k in blocks])
        sizes = np.cumsum([len(blocks[k]) for k in blocks])[:-1]
        Dt = D_task[np.ix_(rows, ref)]
        Dw = pooled.Dca[np.ix_(rows, ref)]

        for name in SCHEMES:
            W = masses_by_name(Dt, name)
            sc = (Dw * W).sum(1) / (M_REF + 1)
            part = dict(zip(blocks.keys(), np.split(sc, sizes)))
            eps = conformal_radius(part["cal"], BETA)
            a = acc[name]
            a["cov"].append(np.mean(part["tst"] <= eps))
            for t in TIERS:
                a[t].append(np.mean(part[t] > eps))
            a["auroc"].append(auroc(part["tst"], part["subtle"]))
            # UNCONFOUNDED: both sides are hand-written prose, so the fact that
            # the reference corpus is machine-generated cancels out.
            a["auroc_hw"].append(auroc(part["safe_handwritten"], part["subtle"]))
            a["eff"].append(effective_support(W).mean())

    exact = np.ceil((K_CAL + 1) * (1 - BETA)) / (K_CAL + 1)
    base = np.array(acc["uniform"]["auroc_hw"])
    print(f"target coverage {exact:.4f}\n")
    print(f"{'scheme':<19}{'family':<16}{'cov':>7}{'FRR-hw':>8}{'degen':>7}{'subtle':>8}"
          f"{'AUROC*':>8}{'eff.sup':>9}{'vs unif':>9}")
    print("-" * 104)
    order = sorted(SCHEMES, key=lambda n: -np.mean(acc[n]["auroc_hw"]))
    for name in order:
        a = acc[name]
        v = np.array(a["auroc_hw"])
        dd = v - base
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)) + 1e-12)
        tag = "(base)" if name == "uniform" else f"{dd.mean():+.3f}{'*' if abs(t) > 2 else ' '}"
        print(f"{name:<19}{FAMILY[name]:<16}{np.mean(a['cov']):>7.3f}"
              f"{np.mean(a['safe_handwritten']):>8.3f}"
              f"{np.mean(a['degenerate']):>7.3f}{np.mean(a['subtle']):>8.3f}"
              f"{v.mean():>8.3f}{np.mean(a['eff']):>9.1f}{tag:>9}")
    print("-" * 104)
    print("AUROC* = AUROC(hand-written safe vs subtle). BOTH sides are hand-written")
    print("         prose, so the machine-generated reference corpus cancels out.")
    print("         This is the only column in the table free of the corpus artifact.")
    print("* paired difference in AUROC* vs uniform mass exceeds 2 s.e.")
    print("FRR-hw = fraction of the 120 HAND-WRITTEN safe prompts wrongly rejected.")
    print("         These are genuinely in-domain; anything above ~0.05 means the")
    print("         generated corpus has narrowed the safe set below what a person writes.")
    return acc, order


# ---------------------------------------------------------- B. scaling ------

def part_B(d, pooled, adv_rows, best_scheme, n_splits=30):
    print(f"\n{'='*104}\nB. SCALING IN M   does eps shrink as M^(-1/d_eff)?\n{'='*104}")
    D_task = d["D_wht"]
    Ms = [30, 60, 125, 250, 500]
    rng = np.random.default_rng(1)
    res = {M: {k: [] for k in ["eps", "cov", "spacing", "subtle", "degenerate",
                               "safe_handwritten", "P4_figure_in_landscape"]} for M in Ms}

    for _ in range(n_splits):
        p = rng.permutation(d["n_pool"])
        for M in Ms:
            ref = p[:M]
            cal = p[500:500 + K_CAL]
            tst = p[500 + K_CAL:500 + K_CAL + N_TEST]
            blocks = {"cal": cal, "tst": tst, **adv_rows}
            rows = np.concatenate([blocks[k] for k in blocks])
            sizes = np.cumsum([len(blocks[k]) for k in blocks])[:-1]
            W = masses_by_name(D_task[np.ix_(rows, ref)], best_scheme)
            sc = (pooled.Dca[np.ix_(rows, ref)] * W).sum(1) / (M + 1)
            part = dict(zip(blocks.keys(), np.split(sc, sizes)))
            eps = conformal_radius(part["cal"], BETA)
            Daa = pooled.Daa[np.ix_(ref, ref)].copy()
            np.fill_diagonal(Daa, np.inf)
            res[M]["eps"].append(eps)
            res[M]["spacing"].append(np.median(Daa.min(1)))
            res[M]["cov"].append(np.mean(part["tst"] <= eps))
            res[M]["subtle"].append(np.mean(part["subtle"] > eps))
            res[M]["degenerate"].append(np.mean(part["degenerate"] > eps))
            res[M]["safe_handwritten"].append(np.mean(part["safe_handwritten"] > eps))
            res[M]["P4_figure_in_landscape"].append(np.mean(part["P4_figure_in_landscape"] > eps))

    print(f"{'M':>6}{'eps*(M+1)':>11}{'spacing':>10}{'ratio':>8}{'cov':>7}"
          f"{'FRR-hw':>8}{'degen':>7}{'subtle':>8}{'P4':>7}")
    print("-" * 72)
    for M in Ms:
        r = res[M]
        print(f"{M:>6}{np.mean(r['eps']) * (M + 1):>11.3f}{np.mean(r['spacing']):>10.3f}"
              f"{np.mean(r['eps']) * (M + 1) / np.mean(r['spacing']):>8.2f}"
              f"{np.mean(r['cov']):>7.3f}{np.mean(r['safe_handwritten']):>8.3f}"
              f"{np.mean(r['degenerate']):>7.3f}"
              f"{np.mean(r['subtle']):>8.3f}{np.mean(r['P4_figure_in_landscape']):>7.3f}")
    e = np.array([np.mean(res[M]["eps"]) * (M + 1) for M in Ms])     # undo the 1/(M+1)
    slope = np.polyfit(np.log(Ms), np.log(e), 1)[0]
    print(f"\nlog-log slope of the un-normalised radius vs M: {slope:+.4f}"
          f"   => implied d_eff = {-1 / slope:.1f}" if slope < 0 else
          f"\nlog-log slope {slope:+.4f} (radius does not shrink)")
    return res, Ms


# ------------------------------------------------ C. pooled vs sequence -----

def part_C(d, pooled, seq, adv_rows, best_scheme, n_splits=N_SPLITS_MST):
    print(f"\n{'='*104}\nC. POOLED [EOS] VECTOR  vs  TOKEN SEQUENCE (W1 ground metric)"
          f"   {n_splits} splits\n{'='*104}")
    D_task = d["D_wht"]
    rng = np.random.default_rng(2)
    names = []
    for sp in (pooled, seq):
        names += [(sp, f"s_(d,1) uniform", "u"), (sp, f"g_(d,1) {best_scheme}", "a"),
                  (sp, f"g^(0,{K_ADAPT})_(d,inf)", "i")]
    acc = {(sp.name, n): {k: [] for k in ["cov", "auroc", "auroc_hw"] + TIERS} for sp, n, _ in names}

    for _ in range(n_splits):
        p = rng.permutation(d["n_pool"])
        ref, cal = p[:M_REF], p[M_REF:M_REF + K_CAL]
        tst = p[M_REF + K_CAL:M_REF + K_CAL + N_TEST]
        blocks = {"cal": cal, "tst": tst, **adv_rows}
        rows = np.concatenate([blocks[k] for k in blocks])
        sizes = np.cumsum([len(blocks[k]) for k in blocks])[:-1]
        Dt = D_task[np.ix_(rows, ref)]
        W = masses_by_name(Dt, best_scheme)
        sel = np.argsort(Dt, axis=1)

        for sp, nm, kind in names:
            if kind == "u":
                sc = sp.order1(rows, ref)
            elif kind == "a":
                sc = sp.order1(rows, ref, W)
            else:
                sc = sp.orderinf_adaptive(rows, ref, sel, K_ADAPT)
            part = dict(zip(blocks.keys(), np.split(sc, sizes)))
            eps = conformal_radius(part["cal"], BETA)
            a = acc[(sp.name, nm)]
            a["cov"].append(np.mean(part["tst"] <= eps))
            for t in TIERS:
                a[t].append(np.mean(part[t] > eps))
            a["auroc"].append(auroc(part["tst"], part["subtle"]))
            a["auroc_hw"].append(auroc(part["safe_handwritten"], part["subtle"]))

    print(f"{'representation':<20}{'estimator':<24}{'cov':>7}{'FRR-hw':>8}{'degen':>7}"
          f"{'subtle':>8}{'border':>8}{'P4':>7}{'AUROC*':>8}")
    print("-" * 96)
    for (spn, nm), a in acc.items():
        print(f"{spn:<20}{nm:<24}{np.mean(a['cov']):>7.3f}"
              f"{np.mean(a['safe_handwritten']):>8.3f}"
              f"{np.mean(a['degenerate']):>7.3f}{np.mean(a['subtle']):>8.3f}"
              f"{np.mean(a['borderline']):>8.3f}"
              f"{np.mean(a['P4_figure_in_landscape']):>7.3f}{np.mean(a['auroc_hw']):>8.3f}")
    return acc


# ---------------------------------------------------------- D. figures ------

def figures(accA, orderA, resB, Ms, accC):
    OUT.mkdir(exist_ok=True)

    # A: reweighting, grouped by family
    fam_col = {"uniform": "grey", "inv_power": "tab:red", "softmax": "tab:orange",
               "std_softmax": "tab:blue", "minmax_softmax": "tab:cyan",
               "rank_exp": "tab:green", "rank_power": "tab:olive",
               "knn": "tab:purple", "top1": "black", "gate_softmax": "tab:brown"}
    fig, ax = plt.subplots(figsize=(11, 6))
    names = orderA
    y = np.arange(len(names))
    vals = [np.mean(accA[n]["auroc_hw"]) for n in names]
    err = [np.std(accA[n]["auroc_hw"], ddof=1) / np.sqrt(len(accA[n]["auroc_hw"])) for n in names]
    ax.barh(y, vals, xerr=err, color=[fam_col[FAMILY[n]] for n in names], height=.75)
    ax.axvline(np.mean(accA["uniform"]["auroc_hw"]), ls="--", c="k", lw=1.2,
               label="uniform mass (non-adaptive)")
    ax.set_yticks(y, names, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlim(0.5, None)
    ax.set_xlabel("AUROC(hand-written safe vs subtle) — corpus-artifact free")
    ax.set_title(f"A. Reweighting schemes for $g_{{d,1}}$  (M={M_REF}, β={BETA})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in fam_col.items() if f != "uniform"]
    ax.legend([plt.Line2D([], [], ls="--", c="k")] + handles,
              ["uniform baseline"] + [f for f in fam_col if f != "uniform"],
              fontsize=7, loc="lower right", ncol=2)
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "s_reweighting.png", dpi=150); plt.close(fig)

    # B: scaling
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    e = np.array([np.mean(resB[M]["eps"]) * (M + 1) for M in Ms])
    sp = np.array([np.mean(resB[M]["spacing"]) for M in Ms])
    ax[0].loglog(Ms, e, "o-", label="conformal radius (un-normalised)")
    ax[0].loglog(Ms, sp, "s-", label="median anchor spacing")
    sl = np.polyfit(np.log(Ms), np.log(e), 1)[0]
    ax[0].loglog(Ms, e[0] * (np.array(Ms) / Ms[0]) ** sl, "k:",
                 label=f"slope {sl:+.3f} → $d_{{eff}}$≈{-1/sl:.0f}")
    ax[0].set_xlabel("M anchors"); ax[0].set_ylabel("whitened units")
    ax[0].set_title("B. radius vs corpus size"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which="both")
    ax[1].semilogx(Ms, [np.mean(resB[M]["eps"]) * (M + 1) / np.mean(resB[M]["spacing"]) for M in Ms],
                   "o-", color="tab:red")
    ax[1].axhline(1.0, ls=":", c="k"); ax[1].set_xlabel("M anchors")
    ax[1].set_ylabel("ε / spacing"); ax[1].set_title("tightness of the region"); ax[1].grid(alpha=.3)
    for t, c in [("subtle", "tab:red"), ("degenerate", "tab:green")]:
        ax[2].semilogx(Ms, [np.mean(resB[M][t]) for M in Ms], "o-", color=c, label=t)
    ax[2].semilogx(Ms, [np.mean(resB[M]["cov"]) for M in Ms], "s--", color="tab:blue",
                   label="safe coverage")
    ax[2].axhline(1 - BETA, ls=":", c="k"); ax[2].set_xlabel("M anchors")
    ax[2].set_ylim(0, 1.05); ax[2].set_title("detection vs corpus size")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "s_scaling.png", dpi=150); plt.close(fig)

    # C: pooled vs sequence
    fig, ax = plt.subplots(figsize=(10, 4.6))
    keys = list(accC.keys())
    labels = [f"{k[1]}\n[{k[0]}]" for k in keys]
    x = np.arange(len(keys)); w = .26
    for i, (t, c) in enumerate([("degenerate", "tab:green"),
                                ("borderline", "tab:orange"), ("subtle", "tab:red")]):
        ax.bar(x + (i - 1) * w, [np.mean(accC[k][t]) for k in keys], w, color=c, label=t)
    ax.plot(x, [np.mean(accC[k]["cov"]) for k in keys], "kd--", ms=6, label="safe coverage")
    ax.axhline(1 - BETA, ls=":", c="k", lw=1)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("fraction blocked"); ax.set_ylim(0, 1.08)
    ax.set_title("C. pooled [EOS] vector vs token sequence (W1 ground metric)")
    ax.legend(fontsize=8, ncol=4); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "s_pooled_vs_sequence.png", dpi=150); plt.close(fig)

    # D: the corpus artefact -- reweighting moves ALONG a trade-off, it does not lift it
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    fams = sorted({FAMILY[n] for n in accA})
    for f in fams:
        ns = [n for n in accA if FAMILY[n] == f]
        ax[0].scatter([np.mean(accA[n]["safe_handwritten"]) for n in ns],
                      [np.mean(accA[n]["subtle"]) for n in ns],
                      s=44, color=fam_col[f], label=f, zorder=3)
        ax[1].scatter([np.mean(accA[n]["eff"]) for n in ns],
                      [np.mean(accA[n]["auroc_hw"]) for n in ns],
                      s=44, color=fam_col[f], zorder=3)
    xs = np.array([np.mean(accA[n]["safe_handwritten"]) for n in accA])
    ys = np.array([np.mean(accA[n]["subtle"]) for n in accA])
    k = np.polyfit(xs, ys, 1)
    ax[0].plot(np.sort(xs), np.polyval(k, np.sort(xs)), "k--", lw=1.2,
               label=f"trade-off line (r={np.corrcoef(xs, ys)[0,1]:.2f})")
    ax[0].set_xlabel("FALSE rejection of hand-written safe prompts")
    ax[0].set_ylabel("subtle prompts blocked")
    ax[0].set_title("every scheme sits on one trade-off line:\nmore detection is bought with more false rejection")
    ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)
    ax[1].axhline(np.mean(accA["uniform"]["auroc_hw"]), ls="--", c="k", lw=1.2,
                  label="uniform mass")
    ax[1].set_xscale("log"); ax[1].set_xlabel("effective support  1/Σw²  (anchors carrying mass)")
    ax[1].set_ylabel("AUROC(hand-written safe vs subtle)")
    ax[1].set_title("discrimination does NOT improve with concentration")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "s_tradeoff.png", dpi=150); plt.close(fig)
    print(f"\nfigures -> {OUT}/s_*.png")


def main():
    d = cl.load()
    pooled, seq, rows_g = build_spaces(d)
    adv_rows = row_index(d)
    accA, orderA = part_A(d, pooled, adv_rows)
    best = orderA[0] if orderA[0] != "uniform" else orderA[1]
    print(f"\nbest reweighting scheme: {best}")
    resB, Ms = part_B(d, pooled, adv_rows, best)
    accC = part_C(d, pooled, seq, adv_rows, best)
    figures(accA, orderA, resB, Ms, accC)


if __name__ == "__main__":
    main()
