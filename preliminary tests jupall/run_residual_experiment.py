"""Distance to the CENTROID vs distance to the SUBSPACE.

The ECF score decomposes exactly as

    1 + d_M^2(theta)  =  1  +  A  +  B ,
    A = sum_{i<=m} c_i^2 / lambda_i      (in the safe corpus's top-m PCA subspace)
    B = ||p||^2 / gamma                  (orthogonal to it)

and the two terms disagree. Measured on the hand-written corpus, mean A is 22.7
for safe prompts but 7.2 for degenerate ones, while mean B goes the other way,
74 to 273. The top-m directions are the axes along which safe prompts VARY, so
having a large coordinate there is evidence of being in-domain -- yet
Mahalanobis distance, which measures displacement from the CENTROID, counts it
as anomalous. A is sign-flipped against the task.

    s_perp(theta) = || theta_w - V V' theta_w ||

is the same score with A deleted: distance to the SUBSPACE rather than to the
centroid. This script measures what that is worth, on both corpora, and asks
whether the idea generalises to the other estimators.

  1  HEAD-TO-HEAD   s_perp against every scalable estimator, on the hand-written
                    corpus (where §4's results were measured) and on the
                    generated corpus (§11).
  2  SPECTRAL BLEND s_lambda^2 = B + lambda A. lambda = 1 is the ECF, lambda = 0
                    is s_perp, lambda < 0 penalises low in-subspace energy.
  3  CUTOFF         how s_perp depends on m.
  4  RESIDUAL METRIC  the Wasserstein scores run in the metric that keeps only
                    the orthogonal complement.

    python run_residual_experiment.py            # ~2 min
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

from ecf import fit_regularized_covariance
from wasserstein import conformal_radius
from sweep_reweighting import auroc
from run_scaling_experiment import bottleneck

OUT = Path(__file__).parent / "outputs"
BETA = 0.05


# ------------------------------------------------------------- geometry ----

class Basis:
    """Whitened coordinates split into the top-m block and its complement."""

    def __init__(self, fit_theta, m):
        mean, V, er, ei, g = fit_regularized_covariance(
            torch.tensor(np.asarray(fit_theta, dtype=float)), m=m)
        self.mean = mean.numpy()
        self.V = V.numpy()
        self.lam = er.numpy()[:self.V.shape[1]]
        self.gamma = float(g)

    def parts(self, Z):
        """(A, B, whitened coords, whitened residual vectors)."""
        diff = np.atleast_2d(np.asarray(Z, dtype=float)) - self.mean
        c = diff @ self.V
        p = diff - c @ self.V.T
        A = (c ** 2 / self.lam).sum(1)
        B = (p ** 2).sum(1) / self.gamma
        return A, B, c / np.sqrt(self.lam), p / np.sqrt(self.gamma)


def make_scores(B_, theta, ref_idx, task_sel=None):
    """All score functions, sharing one basis and one anchor set."""
    A, Bt, cw, pw = B_.parts(theta)
    full = np.concatenate([cw, pw], axis=1)              # whitened, Mahalanobis
    Aref = full[ref_idx]
    M = len(ref_idx)
    mu = Aref.mean(0)
    Dfull = cdist(full, Aref)
    Dperp = cdist(pw, pw[ref_idx])                       # residual-only metric

    def s_perp(i):
        return np.sqrt(Bt[i])

    def s_ecf(i):
        return 1.0 + A[i] + Bt[i]

    def s_centroid(i):                                    # ECF's own geometry
        return ((full[i] - mu) ** 2).sum(1)

    def s_w1(i):
        return Dfull[i].sum(1) / (M * (M + 1))

    def s_w1_perp(i):
        return Dperp[i].sum(1) / (M * (M + 1))

    def s_winf(i):
        out = np.empty(len(i))
        for k, g in enumerate(i):
            a = task_sel[g][:5] if task_sel is not None else np.arange(5)
            sub = np.zeros((6, 6))
            sub[:5, :5] = cdist(full[ref_idx[a]], full[ref_idx[a]])
            dd = np.linalg.norm(full[g] - full[ref_idx[a]], axis=1)
            sub[5, :5] = dd
            sub[:5, 5] = dd
            out[k] = bottleneck(sub)
        return out

    return {"s_perp (subspace)": s_perp, "ECF (centroid)": s_ecf,
            "s_(d,1) uniform": s_w1, "g^(0,5)_(d,inf)": s_winf,
            "s_(d,1) in perp metric": s_w1_perp}, (A, Bt)


def evaluate(fns, cal, tst, tiers, beta=BETA, auroc_pos="subtle", auroc_neg=None):
    out = {}
    for nm, f in fns.items():
        eps = conformal_radius(f(cal), beta)
        st = f(tst)
        r = {"cov": np.mean(st <= eps)}
        for t, idx in tiers.items():
            r[t] = np.mean(f(idx) > eps)
        neg = f(tiers[auroc_neg]) if auroc_neg else st
        r["auroc"] = auroc(neg, f(tiers[auroc_pos]))
        out[nm] = r
    return out


# --------------------------------------------------------------- corpora ---

def corpus_handwritten():
    import data_cache as dc
    th, lab, _, _ = dc.load()
    safe = np.flatnonzero(lab == "safe")
    tiers = {t: np.flatnonzero(lab == t) for t in
             ["degenerate", "borderline", "subtle", "P1_studio",
              "P4_figure_in_landscape"]}
    return dict(name="hand-written (120)", theta=th, safe=safe, tiers=tiers,
                M=60, K=40, T=20, m=16, auroc_neg=None, splits=100)


def corpus_generated():
    import cache_large as cl
    d = cl.load()
    ng, npool = d["n_geom"], d["n_pool"]
    tiers = {t: np.flatnonzero(d["labels"] == t) for t in
             ["safe_handwritten", "degenerate", "borderline", "subtle",
              "P1_studio", "P4_figure_in_landscape"]}
    return dict(name="generated (1000)", theta=d["theta"],
                safe=np.arange(ng, ng + npool), tiers=tiers,
                M=500, K=300, T=200, m=32, auroc_neg="safe_handwritten",
                splits=30, geom=np.arange(ng))


def run_corpus(C, lam_grid, m_grid):
    rng = np.random.default_rng(0)
    acc, lam_acc, m_acc = {}, {l: [] for l in lam_grid}, {mm: [] for mm in m_grid}
    for sp in range(C["splits"]):
        p = rng.permutation(len(C["safe"]))
        ref = C["safe"][p[:C["M"]]]
        cal = C["safe"][p[C["M"]:C["M"] + C["K"]]]
        tst = C["safe"][p[C["M"] + C["K"]:C["M"] + C["K"] + C["T"]]]
        fit = C["theta"][C["geom"]] if "geom" in C else C["theta"][ref]

        B_ = Basis(fit, C["m"])
        ridx = np.arange(len(ref))
        theta_all = C["theta"]
        # anchor indices into theta_all, then a local index map
        full_ref = ref
        sel = {g: np.arange(5) for g in range(len(theta_all))}   # placeholder
        fns, (A, Bt) = make_scores(B_, theta_all, full_ref, task_sel=None)
        r = evaluate(fns, cal, tst, C["tiers"], auroc_neg=C["auroc_neg"])
        for nm, v in r.items():
            acc.setdefault(nm, []).append(v)

        # spectral blend  s^2 = B + lambda A
        neg_i = C["tiers"][C["auroc_neg"]] if C["auroc_neg"] else tst
        for l in lam_grid:
            s = lambda i, l=l: Bt[i] + l * A[i]
            lam_acc[l].append(auroc(s(neg_i), s(C["tiers"]["subtle"])))
        # cutoff sweep for s_perp
        for mm in m_grid:
            Bm = Basis(fit, mm)
            _, Bt2, _, _ = Bm.parts(theta_all)
            m_acc[mm].append(auroc(np.sqrt(Bt2[neg_i]),
                                   np.sqrt(Bt2[C["tiers"]["subtle"]])))
    return acc, lam_acc, m_acc


def main():
    lam_grid = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]
    m_grid = [4, 8, 16, 32, 48]
    results = {}
    for C in (corpus_handwritten(), corpus_generated()):
        acc, lam_acc, m_acc = run_corpus(C, lam_grid, [m for m in m_grid if m < C["M"] - 1])
        results[C["name"]] = (C, acc, lam_acc, m_acc)

        tag = ("AUROC*" if C["auroc_neg"] else "AUROC")
        print(f"\n{'='*98}\n1. HEAD-TO-HEAD on the {C['name']} corpus   "
              f"M={C['M']} K={C['K']} test={C['T']} m={C['m']}, {C['splits']} splits"
              f"\n{'='*98}")
        cols = [t for t in C["tiers"] if t != C["auroc_neg"]]
        print(f"{'score':<26}{'cov':>7}" + "".join(f"{c[:9]:>11}" for c in cols)
              + f"{tag:>9}")
        print("-" * 98)
        order = sorted(acc, key=lambda n: -np.mean([r["auroc"] for r in acc[n]]))
        for nm in order:
            rs = acc[nm]
            print(f"{nm:<26}{np.mean([r['cov'] for r in rs]):>7.3f}"
                  + "".join(f"{np.mean([r[c] for r in rs]):>11.3f}" for c in cols)
                  + f"{np.mean([r['auroc'] for r in rs]):>9.3f}")
        base = np.array([r["auroc"] for r in acc["s_perp (subspace)"]])
        print("-" * 98)
        print(f"paired {tag} against s_perp:")
        for nm in order:
            if nm == "s_perp (subspace)":
                continue
            dd = np.array([r["auroc"] for r in acc[nm]]) - base
            t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)) + 1e-12)
            print(f"   {nm:<26}{dd.mean():+.4f}   (t = {t:+.1f})")

        print(f"\n2. SPECTRAL BLEND  s^2 = B + lambda*A   "
              f"(lambda=1 is the ECF, lambda=0 is s_perp)")
        print("   " + "".join(f"{l:>9.2f}" for l in lam_grid))
        print("   " + "".join(f"{np.mean(lam_acc[l]):>9.3f}" for l in lam_grid))

        print(f"\n3. CUTOFF m for s_perp")
        ms = sorted(m_acc)
        print("   " + "".join(f"{m:>9d}" for m in ms))
        print("   " + "".join(f"{np.mean(m_acc[m]):>9.3f}" for m in ms))

    # ------------------------------------------------------------ figure --
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for (name, (C, acc, lam_acc, m_acc)), mk in zip(results.items(), ["o-", "s-"]):
        order = sorted(acc, key=lambda n: -np.mean([r["auroc"] for r in acc[n]]))
        axes[1].plot(lam_grid, [np.mean(lam_acc[l]) for l in lam_grid], mk, label=name)
        ms = sorted(m_acc)
        axes[2].plot(ms, [np.mean(m_acc[m]) for m in ms], mk, label=name)
    names = list(results["hand-written (120)"][1])
    x = np.arange(len(names))
    w = 0.38
    for i, (name, (C, acc, _, _)) in enumerate(results.items()):
        axes[0].bar(x + (i - .5) * w, [np.mean([r["auroc"] for r in acc[n]]) for n in names],
                    w, label=name)
    axes[0].set_xticks(x, [n.replace(" ", "\n") for n in names], fontsize=7)
    axes[0].set_ylim(0.5, None); axes[0].set_ylabel("AUROC (safe vs subtle)")
    axes[0].set_title("1. distance to the subspace vs the alternatives")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3, axis="y")
    axes[1].axvline(0, ls=":", c="k"); axes[1].axvline(1, ls="--", c="tab:red")
    axes[1].text(1.02, axes[1].get_ylim()[0], "ECF", color="tab:red", fontsize=8)
    axes[1].set_xlabel("λ  (weight on the in-subspace term A)")
    axes[1].set_ylabel("AUROC"); axes[1].set_title("2. spectral blend  $s^2 = B + λA$")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    axes[2].set_xlabel("m  (PCA cutoff)"); axes[2].set_ylabel("AUROC")
    axes[2].set_title("3. where to cut the spectrum"); axes[2].legend(fontsize=8)
    axes[2].grid(alpha=.3)
    fig.suptitle("Distance to the safe SUBSPACE beats distance to its CENTROID", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(OUT / "s_residual.png", dpi=150)
    print(f"\nfigure -> {OUT}/s_residual.png")


if __name__ == "__main__":
    main()
