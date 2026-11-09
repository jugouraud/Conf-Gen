"""Standard flow matching against reflow on the ring of the toy study, with the EXACT marginal velocity (Bayes' formula
for a Gaussian-mixture target), so that the figure shows the couplings and not a fitting error.
safe_prompt_fields.tex, Section 3 (Definition, Lemma and Remark on couplings and reflow).

  (a) independent pairs (x0, x1): standard flow matching;
  (b) trajectories of the exact 1-rectified field v*_p from the same noises;
  (c) reflow pairs (x0, T(x0)), T the flow map of v*_p (Euler, 1000 steps);
  (d) minimal flow-matching loss E||u - E[u | x_t]||^2 against t, u = x1 - x0: exact for the independent coupling,
      k-nearest-neighbour estimate for both couplings (the estimate is checked against the exact curve).
Also checks the identity of Liu et al. (2023) for one rectification with c = |.|^2:
  E|x1 - x0|^2 (independent) - E|T(x0) - x0|^2 = crossing of the independent coupling + curvature of the 1-rectified flow.
Writes ../figures/fig_reflow.{pdf,png} and ../tables/reflow_numbers.tex.     Usage: python make_reflow_figure.py
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import re
import numpy as np
from scipy.spatial import cKDTree
import e1_figures as F                                   # style, colours and save() of the E1 figures
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "..", "tables")
MU = 2.0 * np.c_[np.cos(np.arange(4) * np.pi / 2), np.sin(np.arange(4) * np.pi / 2)]   # the ring of Section 8, phi = 0
SD = 0.3
N, STEPS, KNN = 20000, 1000, 25
TS = (np.arange(48) + 0.5) / 48                          # midpoint rule on (0, 1)


def vstar(x, t):
    """Exact marginal velocity of the independent coupling (Bayes' formula, Lemma 2.10(ii)) for the Gaussian mixture."""
    t = np.broadcast_to(np.asarray(t, float), (len(x),))
    s2 = (1 - t) ** 2 + (t * SD) ** 2                    # x_t given component j: N(t mu_j, s2 I)
    logw = -((x[:, None, :] - t[:, None, None] * MU[None]) ** 2).sum(-1) / (2 * s2[:, None])
    w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    mj = MU[None] + (t * SD ** 2 / s2)[:, None, None] * (x[:, None, :] - t[:, None, None] * MU[None])
    return ((w[..., None] * mj).sum(1) - x) / (1 - t)[:, None]


def flow(x0, keep=False, target=None):
    """Euler integration of v*_p; optionally the path, or the curvature int_0^1 E|target - x0 - v*(X_t, t)|^2 dt."""
    X = x0.copy(); path = [X.copy()]; curv = 0.0
    for k in range(STEPS):
        V = vstar(X, k / STEPS)
        if target is not None:
            curv += np.mean(((target - x0 - V) ** 2).sum(1)) / STEPS
        X = X + V / STEPS
        if keep:
            path.append(X.copy())
    return (X, np.stack(path)) if keep else (X, curv)


def knn_residual(X, U):
    """Leave-one-out k-nearest-neighbour estimate of E|U - E[U | X]|^2."""
    _, idx = cKDTree(X).query(X, KNN + 1)
    return float(np.mean(((U - U[idx[:, 1:]].mean(1)) ** 2).sum(1)))


def main():
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((N, 2))
    x1 = MU[rng.integers(0, 4, N)] + SD * rng.standard_normal((N, 2))      # independent of x0
    T, _ = flow(x0)
    _, curv = flow(x0, target=T)                                           # curvature S of the 1-rectified flow
    exact, est_ind, est_ref = [], [], []
    for t in TS:
        xt, u = (1 - t) * x0 + t * x1, x1 - x0
        exact.append(np.mean((u ** 2).sum(1)) - np.mean((vstar(xt, t) ** 2).sum(1)))   # Pythagoras, Lemma 2.10(i)
        est_ind.append(knn_residual(xt, u))
        xr, ur = (1 - t) * x0 + t * T, T - x0
        est_ref.append(knn_residual(xr, ur))
    exact, est_ind, est_ref = map(np.array, (exact, est_ind, est_ref))
    cost_ind = 2.0 + 4.0 + 2 * SD ** 2                                     # E|x0|^2 + E|x1|^2, exact
    cost_ref = float(np.mean(((T - x0) ** 2).sum(1)))
    V_ind, V_ref = float(exact.mean()), float(est_ref.mean())
    print(f"crossing: independent {V_ind:.3f} (kNN {est_ind.mean():.3f}), reflow {V_ref:.4f} (kNN, an upper estimate)")
    print(f"E|x1-x0|^2: independent {cost_ind:.3f}, reflow {cost_ref:.3f}; curvature of the 1-rectified flow {curv:.3f}")
    print(f"identity: decrease {cost_ind - cost_ref:.3f} = crossing + curvature {V_ind + curv:.3f}")
    print(f"one-step outputs of v*_p: mean distance to the target mean {np.mean(np.linalg.norm(x0 + vstar(x0, 0.0), axis=1)):.2e}")

    # ------------------------------------------------------------------ figure
    sel = np.random.default_rng(4).choice(N, 60, replace=False)
    _, P = flow(x0[sel], keep=True)
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 1.95), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35]))

    def frame(ax, title):
        ax.set_xlim(-3.1, 3.1); ax.set_ylim(-3.1, 3.1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(F.GRID); sp.set_linewidth(0.6)
        ax.set_title(title, loc="left", fontsize=6.3)

    ax = axs[0]
    for i in sel:
        ax.plot([x0[i, 0], x1[i, 0]], [x0[i, 1], x1[i, 1]], lw=0.45, color=F.INK2, alpha=0.55)
    ax.scatter(x0[sel, 0], x0[sel, 1], s=3, color=F.MUTED, linewidths=0, zorder=3)
    ax.scatter(x1[sel, 0], x1[sel, 1], s=3, color=F.BLUE, linewidths=0, zorder=3)
    frame(ax, "(a) standard: random pairs")
    ax = axs[1]
    for j in range(len(sel)):
        ax.plot(P[:, j, 0], P[:, j, 1], lw=0.6, color=F.BLUE, alpha=0.85, solid_capstyle="round")
    ax.scatter(x0[sel, 0], x0[sel, 1], s=3, color=F.MUTED, linewidths=0, zorder=3)
    ax.scatter(P[-1, :, 0], P[-1, :, 1], s=3, color=F.BLUE, linewidths=0, zorder=3)
    frame(ax, "(b) its field: curved paths")
    ax = axs[2]
    for i in sel:
        ax.plot([x0[i, 0], T[i, 0]], [x0[i, 1], T[i, 1]], lw=0.6, color=F.ORANGE, alpha=0.9)
    ax.scatter(x0[sel, 0], x0[sel, 1], s=3, color=F.MUTED, linewidths=0, zorder=3)
    ax.scatter(T[sel, 0], T[sel, 1], s=3, color=F.BLUE, linewidths=0, zorder=3)
    frame(ax, "(c) reflow: noise to its output")
    ax = axs[3]
    ax.plot(TS, exact, color=F.BLUE, lw=1.5, label="standard (exact)")
    ax.plot(TS, est_ind, color=F.BLUE, lw=1.0, ls=(0, (2, 2)), label="standard (estimate)")
    ax.plot(TS, est_ref, color=F.ORANGE, lw=1.5, label="reflow (estimate)")
    ax.set_xlim(0, 1); ax.set_ylim(-0.25, 7); ax.grid(True, axis="y")
    ax.text(0.5, 0.3, "reflow pairs: about %.3f" % float(np.mean(est_ref)), fontsize=5.8, color=F.INK2, ha="center")
    ax.set_xlabel("time $t$", fontsize=6.5)
    ax.set_ylabel("$E\\|\\mathbf{u}-E[\\mathbf{u}\\,|\\,\\mathbf{x}_t]\\|^2$", fontsize=6.5)
    ax.set_title("(d) how much the pairs cross", loc="left", fontsize=6.3)
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.6)
    fig.tight_layout(w_pad=0.4)
    F.save(fig, "fig_reflow")

    # ------------------------------------------------------------------ numbers
    log = open(os.path.join(HERE, "..", "results", "e1_fit.log")).read()
    m = re.search(r"dropped ([0-9.]+)% of the pairs", log)
    mx = float(re.search(r"max \|x1\| = ([0-9.e+]+)\)", log).group(1)); ex = int(np.floor(np.log10(mx)))
    macros = dict(rfVind=f"{V_ind:.1f}", rfVrefl=f"{V_ref:.3f}", rfCostInd=f"{cost_ind:.2f}", rfCostRefl=f"{cost_ref:.2f}",
                  rfCurv=f"{curv:.2f}", rfDecrease=f"{cost_ind - cost_ref:.2f}", rfSum=f"{V_ind + curv:.2f}",
                  rfVmin=f"{exact.min():.1f}", rfVzero=f"{exact[0]:.1f}", rfEoneDropped=f"{float(m.group(1)):.2f}",
                  rfEoneMax=f"{mx / 10 ** ex:.1f}\\times10^{{{ex}}}")
    with open(os.path.join(TAB, "reflow_numbers.tex"), "w") as f:
        f.write("% written by code/make_reflow_figure.py\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("macros:", macros)


if __name__ == "__main__":
    main()
