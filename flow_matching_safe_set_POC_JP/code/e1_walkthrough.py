"""Step-by-step walkthrough of the E1 test (safe_prompt_fields.tex, Section 9): figures and quoted numbers.

Running example: the first base prompt of the unsafe pool, (phi, rho) = (10.1 deg, 1.81), with unsafe mass
m = 0.01, 0.03, 0.1, 0.3 (pool indices 300, 400, 500, 600); its safe version (m = 0) is encoded on the fly.
Everything uses split 0 and the 1-rectified model with 50 Euler steps.
Heavy parts (the slice of field space, the replay of the white-box attack) are cached in
../results/e1/walkthrough.npz; --recompute rebuilds the cache.
Writes ../figures/fig_e1_{task,fields,time,coupling,scores,slice,certificate,attack}.{pdf,png} and
../tables/e1_walk_numbers.tex.          Usage: python e1_walkthrough.py [--recompute]
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time
import numpy as np
from joblib import Parallel, delayed
import e1_phase2 as P
from e1_eval import enc
from e1_common import Features, fields, simulate, one_step, sample_target, r_A
import e1_figures as F                                   # style, colours and save() shared with the other E1 figures
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

BLUE, ORANGE, AQUA, YELLOW, INK, INK2, MUTED, GRID, AXIS, CRIT = (F.BLUE, F.ORANGE, F.AQUA, F.YELLOW, F.INK, F.INK2,
                                                                 F.MUTED, F.GRID, F.AXIS, F.CRIT)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]     # sequential blue ramp
UNSAFE_CMAP = LinearSegmentedColormap.from_list("unsafe", ["#fcfcfb", "#f3c4c0", "#e7867f", CRIT, "#8f2222"])
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(P.RES, "walkthrough.npz")
TAB = os.path.join(HERE, "..", "tables")
CFGM, J = "rect1", 50                                    # model/sampler of the walkthrough
NS, Z, E = P.NS, P.Z, P.E
WST = P.WST[CFGM]
J_EX = {0.01: NS, 0.03: NS + 100, 0.1: NS + 200, 0.3: NS + 300}
Z_EX = Z[J_EX[0.3]]
Z_SAFE = np.r_[Z_EX[:2], 0.0]
E_SAFE = enc(Z_SAFE[None])[0]
ANC, CAL, TE = P.split(1000)
GRID_T = np.linspace(0.0, 1.0, J + 1)
LABEL = {"U-L2": "as written (uniform, $L^2$)", "kNN1-tr1": "improved (nearest, truncated $L^1$)",
         "E-U": "prompt space (uniform)", "E-kNN1": "prompt space (nearest)", "kNN1-L1": "nearest, $L^1$",
         "kNN1-tl.5": "nearest, tail score"}
NUM = {}                                                  # macros for tables/e1_walk_numbers.tex


def mac(name, val):
    NUM[name] = val


def W_of(e):
    return fields(WST, np.atleast_2d(e))[0]


def trajectories(W, x0):
    """Euler positions at every grid time, (J + 1, n, 2), for one field."""
    feat = P.feat
    Y = x0.copy(); out = [Y.copy()]
    for k in range(J):
        Y = Y + (GRID_T[k + 1] - GRID_T[k]) * feat(Y, GRID_T[k]) @ W.T
        out.append(Y.copy())
    return np.stack(out)


def velocity(W, X, t):
    return P.feat(X, t) @ W.T


def _sim(Ws, x0):
    return simulate(Features(P.CFG["F"], seed=P.CFG["feat_seed"]), Ws, x0, J)


def sim_many(Ws, x0, n_jobs=12):
    """Outputs of many fields on shared noise, in parallel: (P, n, 2)."""
    parts = Parallel(n_jobs=n_jobs)(delayed(_sim)(Ws[c], x0) for c in np.array_split(np.arange(len(Ws)), 4 * n_jobs) if len(c))
    return np.concatenate(parts, 0)


def sim_points(W, X, n_jobs=12):
    """Outputs of one field on many noise points, in parallel: (n, 2)."""
    parts = Parallel(n_jobs=n_jobs)(delayed(_sim)(W[None], X[c]) for c in np.array_split(np.arange(len(X)), n_jobs))
    return np.concatenate([p[0] for p in parts], 0)


def in_A(Y):
    return np.linalg.norm(Y, axis=-1) <= r_A


def pA(Y):
    return float(np.mean(in_A(Y)))


def blob_label(Y, z):
    """Index of the nearest of the four blob centres of prompt z, or 4 if the point is in A."""
    ang = z[0] + np.arange(4) * np.pi / 2
    C = z[1] * np.c_[np.cos(ang), np.sin(ang)]
    lab = np.argmin(((Y[:, None] - C[None]) ** 2).sum(-1), 1)
    lab[in_A(Y)] = 4
    return lab


def plane_box(ax, lim=3.2, ticks=True):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    if ticks:
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    else:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():                     # a light full frame: these panels are pieces of the plane
            sp.set_visible(True); sp.set_color(GRID); sp.set_linewidth(0.6)


def draw_A(ax, plus=None, lw=0.7):
    ax.add_patch(Circle((0, 0), r_A, facecolor=CRIT, alpha=0.08, edgecolor=CRIT, linewidth=lw, zorder=1))
    if plus is not None:
        ax.add_patch(Circle((0, 0), r_A + plus, facecolor="none", edgecolor=CRIT, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1))


def scatter_out(ax, Y, s=0.8, alpha=0.45):
    m = in_A(Y)
    ax.scatter(Y[~m, 0], Y[~m, 1], s=s, color=BLUE, alpha=alpha, linewidths=0, rasterized=True)
    ax.scatter(Y[m, 0], Y[m, 1], s=s * 1.5, color=CRIT, alpha=min(1, alpha + 0.35), linewidths=0, rasterized=True)


def corner(ax, txt, loc="ll", size=6.5):
    x, ha = (0.03, "left") if loc[1] == "l" else (0.97, "right")
    y, va = (0.03, "bottom") if loc[0] == "l" else (0.97, "top")
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=size, color=INK2, ha=ha, va=va)


# ====================================================================================================================
# cached computations
# ====================================================================================================================
def enforce(name, j):
    """Back-off of pool prompt j for one method on split 0, exactly as in e1_phase2 (returns field, kappa, target)."""
    reg = P.Region(name, CFGM, ANC, CAL)
    pool = np.r_[ANC, CAL]; feas = pool[reg.scores_pool(pool) <= reg.eps]
    Wf, kap, s0, c = P.enforce_one(reg, int(j), fields(WST, E[feas]), E[feas], P.D[f"{CFGM}_Tp"][feas].astype(float))
    return Wf, kap, int(feas[c]), reg


def attack_replay(name, s=0, n_starts=12, iters=300, seed=0):
    """The white-box attack of e1_phase2.attack, step for step, keeping the final embedding of every run."""
    anc, cal, te = P.split(1000 + s)
    reg = P.Region(name, CFGM, anc, cal)
    rng = np.random.default_rng(seed); x_att = P.x_eval[:600]
    sd = E.std(0)
    pool_feas = np.r_[anc, cal]; feas = pool_feas[reg.scores_pool(pool_feas) <= reg.eps]
    feas_W, feas_E, feas_T = fields(WST, E[feas]), E[feas], P.D[f"{CFGM}_Tp"][feas].astype(float)
    starts = rng.choice(np.arange(NS, P.NP)[Z[NS:, 2] == 0.3], n_starts, replace=False)
    runs = []
    for j in starts:
        Wf, kap, _, c = P.enforce_one(reg, int(j), feas_W, feas_E, feas_T)
        e = (1 - kap) * E[j] + kap * feas_E[c]
        W = fields(WST, e[None])[0]
        if reg.score_field(W, e) > reg.eps:
            W = Wf
        cur = np.mean(np.linalg.norm(P.outputs(W, CFGM, x_att), axis=1) <= r_A); step = 0.3
        for it in range(iters):
            e2 = e + step * sd * rng.standard_normal(len(e)); W2 = fields(WST, e2[None])[0]
            if reg.score_field(W2, e2) > reg.eps:
                step *= 0.97; continue
            val = np.mean(np.linalg.norm(P.outputs(W2, CFGM, x_att), axis=1) <= r_A)
            if val >= cur:
                e, W, cur = e2, W2, val; step *= 1.05
        Yf = P.outputs(W, CFGM, P.x_eval[600:])
        runs.append(dict(start=int(j), pA_fresh=float(np.mean(np.linalg.norm(Yf, axis=1) <= r_A)), W=W, e=e))
    return runs


def build_cache():
    t0 = time.time()
    out = {}
    # --- slice of field space through the unsafe request u, the common back-off target c and a rotated safe anchor a
    ju = J_EX[0.3]
    _, kU, cU, _ = enforce("U-L2", ju)
    _, kK, cK, _ = enforce("kNN1-tr1", ju)
    assert cU == cK, "the two methods back off towards different safe fields; the slice would not contain both paths"
    a = int(ANC[np.argmin(Z[ANC, 0])])                                   # the anchor with the most negative rotation
    Wu, Wc, Wa = W_of(E[ju]), W_of(E[cU]), W_of(E[a])
    al = np.linspace(-0.3, 1.25, 63); ga = np.linspace(-0.3, 1.3, 65)
    AA, GG = np.meshgrid(al, ga)
    Wg = Wc[None] + AA.ravel()[:, None, None] * (Wu - Wc)[None] + GG.ravel()[:, None, None] * (Wa - Wc)[None]
    Tp = sim_many(Wg, P.x_probe); Te = sim_many(Wg, P.x_eval[:1000])
    out.update(sl_al=al, sl_ga=ga, sl_pA=in_A(Te).mean(-1).reshape(AA.shape), sl_c=cU, sl_a=a, sl_kU=kU, sl_kK=kK)
    for name in ["U-L2", "kNN1-tr1"]:
        reg = P.Region(name, CFGM, ANC, CAL)
        out[f"sl_S_{name}"] = (P.agg(reg.dist_T(Tp), reg.how) / reg.eps).reshape(AA.shape)
    print(f"[{time.time()-t0:.0f}s] slice", flush=True)
    # --- replay of the attack for the prompt-space region, the region as written and the improved region
    names = ["E-U", "U-L2", "kNN1-tr1"]
    res = Parallel(n_jobs=3)(delayed(attack_replay)(n) for n in names)
    stored = json.load(open(os.path.join(P.RES, "phase2.json")))["attack"]
    for n, runs in zip(names, res):
        ref = [a_ for a_ in stored if a_["cfg"] == CFGM and a_["method"] == n][0]["runs"]
        assert np.allclose([r["pA_fresh"] for r in runs], [r["pA_fresh"] for r in ref]), f"attack replay differs for {n}"
        out[f"at_pA_{n}"] = np.array([r["pA_fresh"] for r in runs])
        k = int(np.argmax(out[f"at_pA_{n}"]))
        out[f"at_W_{n}"], out[f"at_e_{n}"] = runs[k]["W"], runs[k]["e"]
    print(f"[{time.time()-t0:.0f}s] attack replay (matches phase2.json)", flush=True)
    np.savez_compressed(CACHE, **out)
    return dict(np.load(CACHE))


# ====================================================================================================================
# figures
# ====================================================================================================================
def fig_task():
    """Step 1: what safe and unsafe prompts ask for."""
    rng = np.random.default_rng(31)
    fig, axs = plt.subplots(1, 5, figsize=(6.5, 1.55))
    ax = axs[0]
    for k, phi in enumerate(np.linspace(-np.pi / 8, np.pi / 8, 5)):
        Y = sample_target(np.r_[phi, 2.0, 0.0], 250, rng)
        ax.scatter(Y[:, 0], Y[:, 1], s=0.7, color=SEQ[k + 1], alpha=0.8, linewidths=0, rasterized=True)
    draw_A(ax); plane_box(ax, ticks=False)
    ax.set_title("(a) safe prompts", loc="left", fontsize=6.5)
    for ax, m in zip(axs[1:], [0.0, 0.03, 0.1, 0.3]):
        z = np.r_[Z_SAFE[:2], m]
        Y = sample_target(z, 1000, rng)
        scatter_out(ax, Y, s=0.7, alpha=0.55); draw_A(ax); plane_box(ax, ticks=False)
        pA_true = float(D_TRUE[m])
        ax.set_title(("(b) running example, safe" if m == 0 else f"({'bcde'[[0.0, 0.03, 0.1, 0.3].index(m)]}) unsafe, $m={m:g}$"),
                     loc="left", fontsize=6.5)
        corner(ax, f"$P(A)={pA_true:.3f}$", loc="ur")
    axs[1].annotate("unsafe set $A$", xy=(-0.42, 0.42), xytext=(-2.95, 2.1), fontsize=6, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))
    fig.tight_layout(w_pad=0.3)
    F.save(fig, "fig_e1_task")


def fig_fields():
    """Step 2: the vector fields of the safe version and of the unsafe request, and their difference, over time."""
    x0 = P.x_eval[:1500]
    Ws, Wu = W_of(E_SAFE), W_of(E[J_EX[0.3]])
    TR = {"s": trajectories(Ws, x0), "u": trajectories(Wu, x0)}
    endA = {k: in_A(v[-1]) for k, v in TR.items()}
    ks = [0, 25, 40]
    sub = np.random.default_rng(3).choice(len(x0), 110, replace=False)
    g = np.linspace(-2.8, 2.8, 15); GX, GY = np.meshgrid(g, g); PG = np.c_[GX.ravel(), GY.ravel()]
    fig, axs = plt.subplots(3, 4, figsize=(6.5, 5.15))
    rows = [("s", Ws, "safe version ($m=0$)"), ("u", Wu, "unsafe request ($m=0.3$)")]
    for r, (key, W, lab) in enumerate(rows):
        for c, k in enumerate(ks):
            ax = axs[r, c]; Y = TR[key][k]
            m = endA[key]
            ax.scatter(Y[~m, 0], Y[~m, 1], s=0.5, color=BLUE, alpha=0.35, linewidths=0, rasterized=True)
            ax.scatter(Y[m, 0], Y[m, 1], s=0.8, color=CRIT, alpha=0.75, linewidths=0, rasterized=True)
            V = velocity(W, Y[sub], GRID_T[k])
            ax.quiver(Y[sub, 0], Y[sub, 1], V[:, 0], V[:, 1], angles="xy", scale_units="xy", scale=1 / 0.15,
                      width=0.006, headwidth=3.5, headlength=4, color=INK, alpha=0.75, zorder=3)
            draw_A(ax); plane_box(ax, ticks=False)
        ax = axs[r, 3]; scatter_out(ax, TR[key][-1], s=0.6); draw_A(ax); plane_box(ax, ticks=False)
        corner(ax, f"$P(A)={pA(TR[key][-1]):.3f}$")
        axs[r, 0].set_ylabel(lab, fontsize=6.5)
    # difference of the two fields where the particles are, and the displacement of each noise point
    for c, k in enumerate(ks):
        ax = axs[2, c]
        both = np.r_[TR["s"][k], TR["u"][k]]
        H, _, _ = np.histogram2d(both[:, 0], both[:, 1], bins=[np.r_[g - 0.2, g[-1] + 0.2]] * 2)
        keep = H.T.ravel() >= 4
        dV = velocity(Wu - Ws, PG[keep], GRID_T[k])
        ax.scatter(TR["u"][k][endA["u"], 0], TR["u"][k][endA["u"], 1], s=0.6, color=CRIT, alpha=0.25, linewidths=0, rasterized=True)
        ax.quiver(PG[keep, 0], PG[keep, 1], dV[:, 0], dV[:, 1], angles="xy", scale_units="xy", scale=1 / 0.15,
                  width=0.006, headwidth=3.5, headlength=4, color=INK, alpha=0.85, zorder=3)
        draw_A(ax); plane_box(ax, ticks=False)
    ax = axs[2, 3]
    Ys, Yu = TR["s"][-1], TR["u"][-1]
    for i in range(0, len(x0), 2):
        ax.plot([Ys[i, 0], Yu[i, 0]], [Ys[i, 1], Yu[i, 1]], lw=0.35, color=CRIT if endA["u"][i] else MUTED,
                alpha=0.6 if endA["u"][i] else 0.35, rasterized=True)
    draw_A(ax); plane_box(ax, ticks=False)
    axs[2, 0].set_ylabel("difference (unsafe $-$ safe)", fontsize=6.5)
    for c, title in enumerate(["$t=0$", "$t=0.5$", "$t=0.8$", "outputs ($t=1$)"]):
        axs[0, c].set_title(title, fontsize=7)
    corner(axs[2, 3], "each segment: one noise,\nsafe output to unsafe output", loc="ul", size=5.3)
    fig.tight_layout(w_pad=0.2, h_pad=0.4)
    F.save(fig, "fig_e1_fields")
    disp = np.linalg.norm(Yu - Ys, axis=1)
    mac("ewMassA", f"{pA(Yu):.2f}")
    mac("ewDispA", f"{np.median(disp[endA['u']]):.1f}")
    mac("ewDispNotA", f"{np.median(disp[~endA['u']]):.2f}")


def fig_time():
    """Step 3: a curved (1-rectified) and a straightened (2-rectified) field, and their one-step maps."""
    x0 = P.x_eval[:1000]
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.35))
    for r, (name, Wst) in enumerate([("1-rectified", P.WST["rect1"]), ("2-rectified (reflow)", P.WST["rect2"])]):
        W = fields(Wst, E_SAFE[None])[0]
        TR = trajectories(W, x0); T1 = one_step(P.feat, W[None], x0)[0]
        zeta = float(np.sqrt(((TR[-1] - T1) ** 2).sum(-1).mean()))
        ax = axs[r, 0]; idx = np.arange(0, 1000, 8)
        V0 = velocity(W, x0[idx], 0.0)
        ax.scatter(x0[idx, 0], x0[idx, 1], s=2, color=MUTED, linewidths=0, zorder=2)
        ax.quiver(x0[idx, 0], x0[idx, 1], V0[:, 0], V0[:, 1], angles="xy", scale_units="xy", scale=1, width=0.004,
                  headwidth=3, headlength=3.5, color=INK2, alpha=0.6, zorder=3)
        draw_A(ax); plane_box(ax)
        ax = axs[r, 1]
        for i in range(0, 1000, 14):
            ax.plot(TR[:, i, 0], TR[:, i, 1], lw=0.55, color=BLUE, alpha=0.75, solid_capstyle="round")
        ax.scatter(TR[0, ::14, 0], TR[0, ::14, 1], s=2, color=MUTED, linewidths=0, zorder=3)
        draw_A(ax); plane_box(ax); corner(ax, f"defect $\\zeta={zeta:.2f}$")
        ax = axs[r, 2]
        ax.scatter(TR[-1, :, 0], TR[-1, :, 1], s=0.7, color=BLUE, alpha=0.5, linewidths=0, rasterized=True, label="50 Euler steps")
        ax.scatter(T1[:, 0], T1[:, 1], s=0.7, color=ORANGE, alpha=0.6, linewidths=0, rasterized=True, label="one step")
        draw_A(ax); plane_box(ax); corner(ax, f"one step: {100 * pA(T1):.0f}% in $A$")
        axs[r, 0].set_ylabel(name, fontsize=7)
        key = "One" if r == 0 else "Two"
        mac(f"ewZeta{key}", f"{zeta:.2f}"); mac(f"ewOneStepA{key}", f"{100 * pA(T1):.0f}")
    axs[0, 0].set_title("(a) one-step displacement $v(\\mathbf{x},0)$", loc="left", fontsize=6.5)
    axs[0, 1].set_title("(b) trajectories of the sampler", loc="left", fontsize=6.5)
    axs[0, 2].set_title("(c) outputs: 50 steps and one step", loc="left", fontsize=6.5)
    h, l = axs[0, 2].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.99, 1.04), ncol=2, markerscale=6, handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    F.save(fig, "fig_e1_time")


def fig_coupling():
    """Step 4: how two fields are compared, through the outputs they give to the same noise."""
    g = np.linspace(-3, 3, 241); GX, GY = np.meshgrid(g, g); X0 = np.c_[GX.ravel(), GY.ravel()]
    a = int(ANC[np.argmin(Z[ANC, 0])])
    Tp = P.D[f"{CFGM}_Tp"].astype(float)
    Ts_probe = P.outputs(W_of(E_SAFE), CFGM, P.x_probe)
    near = int(ANC[np.argmin(np.minimum(np.linalg.norm(Ts_probe[None] - Tp[ANC], axis=-1), 1.0).mean(-1))])
    fz = {"s": (W_of(E_SAFE), Z_SAFE), "u": (W_of(E[J_EX[0.3]]), Z_EX), "a": (W_of(E[a]), Z[a]), "n": (W_of(E[near]), Z[near])}
    lab = {k: blob_label(sim_points(W, X0), z).reshape(GX.shape) for k, (W, z) in fz.items()}
    sect = ListedColormap(["#eceae4", "#d6d4cc", "#eceae4", "#d6d4cc", CRIT])
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 1.95), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
    titles = ["(a) safe version: noise $\\to$ blob", "(b) unsafe request ($m=0.3$)",
              "(c) safe anchor $a$ ($%.0f^\\circ$)" % np.degrees(Z[a, 0])]
    for ax, key, title in zip(axs[:3], ["s", "u", "a"], titles):
        L = lab[key].astype(float)
        if key == "a":                                                   # noise whose blob changes w.r.t. the safe version
            sw = (lab["a"] != lab["s"]) & (lab["a"] < 4) & (lab["s"] < 4)
            L = np.where(sw, np.nan, L)
        ax.imshow(L, origin="lower", extent=[-3, 3, -3, 3], cmap=sect, vmin=-0.5, vmax=4.5, interpolation="nearest")
        if key == "a":
            ax.imshow(np.where(sw, 1.0, np.nan), origin="lower", extent=[-3, 3, -3, 3], cmap=ListedColormap([ORANGE]),
                      interpolation="nearest", alpha=0.9)
        ax.contour(GX, GY, lab["s"], levels=[0.5, 1.5, 2.5, 3.5], colors=INK, linewidths=0.4,
                   linestyles="solid" if key != "a" else "dashed")
        for rr in [1, 2]:
            ax.add_patch(Circle((0, 0), rr, fill=False, edgecolor=BLUE, linewidth=0.5, linestyle=(0, (1, 1.5))))
        plane_box(ax, lim=3.0, ticks=False); ax.set_title(title, loc="left", fontsize=6.3)
    # displacement distributions on the evaluation noise, against the safe version
    Ys = P.outputs(W_of(E_SAFE), CFGM, P.x_eval)
    pairs = [("n", "nearest safe anchor", BLUE), ("a", "safe anchor $a$", INK2), ("u", "unsafe request", CRIT)]
    dl = np.linspace(0, 3.2, 321)
    ax = axs[3]
    rows = {}
    for key, name, col in pairs:
        Yo = P.outputs(fz[key][0], CFGM, P.x_eval)
        d = np.linalg.norm(Yo - Ys, axis=1)
        S = np.array([np.mean(d >= x) for x in dl])
        ax.plot(dl, S, color=col, lw=1.3, label=name, solid_capstyle="round")
        rows[key] = dict(L2=np.sqrt(np.mean(d ** 2)), L1=np.mean(d), tr=np.mean(np.minimum(d, 1.0)), tail=np.mean(d >= 0.5))
    ax.axvline(1.0, color=AXIS, lw=0.7, zorder=0); ax.axvline(0.5, color=AXIS, lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax.text(1.0, 1.02, "$r=1$", fontsize=5.5, color=MUTED, ha="center", va="bottom")
    ax.text(0.5, 1.02, "$0.5$", fontsize=5.5, color=MUTED, ha="center", va="bottom")
    ax.set_xlim(0, 3.2); ax.set_ylim(0, 1.0); ax.grid(True, axis="y")
    ax.set_xlabel("displacement $\\delta$", fontsize=6.5); ax.set_ylabel("$P(\\|T(\\mathbf{x})-T'(\\mathbf{x})\\|\\geq\\delta)$", fontsize=6.5)
    ax.set_title("(d) how far the same noise lands", loc="left", fontsize=6.3, pad=9)
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.2, bbox_to_anchor=(1.02, 0.78))
    fig.tight_layout(w_pad=0.3)
    F.save(fig, "fig_e1_coupling")
    # numbers: masses on the evaluation noise, and the four distances for the three pairs
    lab_e = {k: blob_label(P.outputs(fz[k][0], CFGM, P.x_eval), fz[k][1]) for k in ["s", "a"]}
    sw_e = (lab_e["a"] != lab_e["s"]) & (lab_e["a"] < 4) & (lab_e["s"] < 4)
    mac("ewSwitch", f"{100 * np.mean(sw_e):.0f}")
    mac("ewRotA", f"{np.degrees(Z_SAFE[0] - Z[a, 0]):.0f}")
    mac("ewAngA", f"{np.degrees(Z[a, 0]):.0f}")
    mac("ewRotNear", f"{abs(np.degrees(Z_SAFE[0] - Z[near, 0])):.1f}")
    for key, tag in [("n", "Near"), ("a", "Rot"), ("u", "Uns")]:
        for q, qt in [("L2", "Ltwo"), ("L1", "Lone"), ("tr", "Tr"), ("tail", "Tail")]:
            mac(f"ewD{qt}{tag}", f"{rows[key][q]:.2f}")


def fig_scores():
    """Step 5: the conformal region as a threshold on a score; why the uniform score cannot see small unsafe modes."""
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.2), gridspec_kw=dict(height_ratios=[1.15, 1]))
    groups = [("calibration", CAL, False), ("safe test", TE, False)] + \
             [(f"$m={m:g}$", np.arange(NS, P.NP)[Z[NS:, 2] == m], True) for m in [0.01, 0.03, 0.1, 0.3]]
    for col, name in enumerate(["U-L2", "kNN1-tr1"]):
        reg = P.Region(name, CFGM, ANC, CAL)
        ax = axs[0, col]
        for k, (lab, idx, unsafe) in enumerate(groups):
            s = reg.scores_pool(idx) / reg.eps
            y = -k + 0.32 * (np.random.default_rng(k).random(len(s)) - 0.5)
            ax.scatter(s, y, s=3, color=CRIT if unsafe else BLUE, alpha=0.55, linewidths=0, rasterized=True)
            out = np.mean(s > 1)
            ax.text(1.02, -k, f"{100 * out:.0f}%", transform=ax.get_yaxis_transform(), fontsize=6, va="center", color=INK2)
            if k == 1:
                mac(f"ewCov{'U' if col == 0 else 'K'}", f"{100 * (1 - out):.0f}")
                if col == 0:                                             # spread of the safe test scores, as written
                    mac("ewSafeLoU", f"{np.quantile(s, 0.05):.2f}"); mac("ewSafeHiU", f"{np.quantile(s, 0.95):.2f}")
            if unsafe:
                mac(f"ewEx{'U' if col == 0 else 'K'}{['a', 'b', 'c', 'd'][k - 2]}", f"{100 * out:.0f}")
        for m, mk in [(0.1, "o"), (0.3, "o")]:
            j = J_EX[m]; sj = reg.scores_pool(np.array([j]))[0] / reg.eps
            kk = 2 + [0.01, 0.03, 0.1, 0.3].index(m)
            ax.scatter([sj], [-kk], s=22, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=4)
        ax.axvline(1.0, color=INK, lw=0.9)
        ax.set_yticks(-np.arange(len(groups))); ax.set_yticklabels([g_[0] for g_ in groups], fontsize=6.5)
        ax.set_xscale("log"); ax.set_xlim((0.55, 1.9) if col == 0 else (0.02, 6))
        tk = [0.6, 0.8, 1, 1.5] if col == 0 else [0.03, 0.1, 0.3, 1, 3]
        ax.set_xticks(tk); ax.set_xticklabels([f"{v:g}" for v in tk]); ax.minorticks_off()
        ax.set_xlabel("score / threshold $\\varepsilon$", fontsize=6.5)
        ax.text(1.02, 0.6, "outside", transform=ax.get_yaxis_transform(), fontsize=6, color=INK2)
        ax.set_title(("(a) " if col == 0 else "(b) ") + LABEL[name], loc="left", fontsize=6.8)
        ax.grid(True, axis="x")
        # distance profiles of the running example (m = 0 and m = 0.1) to the 30 safe anchors
        ax = axs[1, col]
        Wpair = np.stack([W_of(E_SAFE), W_of(E[J_EX[0.1]])])
        Tq = np.stack([P.outputs(W, CFGM, P.x_probe) for W in Wpair])
        dmat = reg.dist_T(Tq)                                           # (2, 30)
        dphi = np.degrees(np.abs(Z[ANC, 0] - Z_SAFE[0])); o = np.argsort(dphi)
        for r_, (col_, lab_) in enumerate([(BLUE, "running example, $m=0$"), (CRIT, "running example, $m=0.1$")]):
            ax.plot(dphi[o], dmat[r_][o], color=col_, lw=0.8, alpha=0.6)
            ax.scatter(dphi[o], dmat[r_][o], s=6, color=col_, label=lab_, zorder=3, linewidths=0)
            sc = P.agg(dmat[r_:r_ + 1], reg.how)[0]
            if name == "U-L2":
                ax.axhline(sc, color=col_, lw=0.8, ls=(0, (3, 2)))
            else:
                i0 = int(np.argmin(dmat[r_]))
                ax.scatter([dphi[i0]], [dmat[r_][i0]], s=40, facecolors="none", edgecolors=col_, linewidths=1.0, zorder=4)
            mac(f"ewSc{'U' if col == 0 else 'K'}{'zero' if r_ == 0 else 'one'}", f"{sc / reg.eps:.2f}")
        ax.axhline(reg.eps, color=INK, lw=0.9)
        ax.text(0.47, reg.eps, "threshold", transform=ax.get_yaxis_transform(), fontsize=6, ha="center", va="bottom", color=INK)
        ax.set_xlabel("rotation difference to the safe anchor (degrees)", fontsize=6.5)
        ax.set_ylabel("distance to the anchor", fontsize=6.5)
        ax.set_title("(c) score = mean of these distances (dashed)" if col == 0 else "(d) score = smallest distance (circled)",
                     loc="left", fontsize=6.8)
        ax.grid(True, axis="y")
        if col == 0:
            ax.legend(loc="lower right", fontsize=5.8, handletextpad=0.3, markerscale=1.5)
        mac(f"ewEps{'U' if col == 0 else 'K'}", f"{reg.eps:.2f}")
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    F.save(fig, "fig_e1_scores")


def fig_slice(C):
    """Step 6: a two-dimensional slice of field space, the two regions, the unsafe mass, and the enforcement path."""
    al, ga, pAg = C["sl_al"], C["sl_ga"], C["sl_pA"]
    kU, kK, c, a, ju = float(C["sl_kU"]), float(C["sl_kK"]), int(C["sl_c"]), int(C["sl_a"]), J_EX[0.3]
    Wu, Wc, Wa = W_of(E[ju]), W_of(E[c]), W_of(E[a])
    fig = plt.figure(figsize=(6.5, 3.25))
    gs = fig.add_gridspec(2, 6, width_ratios=[2.2, 0.07, 0.36, 1, 1, 1], wspace=0.08, hspace=0.32)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.pcolormesh(al, ga, pAg, cmap=UNSAFE_CMAP, vmin=0, vmax=0.3, shading="auto", rasterized=True)
    for name, col in [("U-L2", BLUE), ("kNN1-tr1", AQUA)]:
        ax.contour(al, ga, C[f"sl_S_{name}"], levels=[1.0], colors=col, linewidths=1.6)
    ax.annotate("", xy=(0.03, 0), xytext=(0.97, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=0,
                                                                 linestyle=(0, (3, 2))))
    pts = {1: (1.0, 0.0), 2: (1 - kU, 0.0), 3: (1 - kK, 0.0), 4: (0.0, 0.0), 5: (0.0, 0.5), 6: (0.0, 1.0)}
    for k, (x, y) in pts.items():
        ax.scatter([x], [y], s=46, color="white", edgecolors=INK, linewidths=0.8, zorder=5)
        ax.text(x, y, str(k), fontsize=6, ha="center", va="center", color=INK, zorder=6)
    ax.text(0.97, -0.17, "unsafe request $u$", fontsize=6, ha="center", color=INK)
    ax.text(0.05, -0.17, "safe field $c$", fontsize=6, ha="center", color=INK)
    ax.text(0.07, 1.08, "safe anchor $a$", fontsize=6, ha="left", color=INK)
    ax.text(0.86, 1.17, "region as written", fontsize=6.5, color=BLUE, ha="right")
    ax.text(0.36, 0.13, "improved region", fontsize=6.5, color=AQUA, ha="left", va="center")
    ax.set_xlim(al[0], al[-1]); ax.set_ylim(ga[0], ga[-1])
    ax.set_xlabel("$\\alpha$: weight on $u-c$", fontsize=6.5); ax.set_ylabel("$\\gamma$: weight on $a-c$", fontsize=6.5)
    ax.set_title("(a) fields $c+\\alpha(u-c)+\\gamma(a-c)$", loc="left", fontsize=6.8)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[:, 1])); cb.ax.tick_params(labelsize=6)
    cb.set_label("$P(A)$ of the outputs", fontsize=6.5); cb.outline.set_linewidth(0.4)
    # what six points of the plane generate
    Wpts = [Wc + x * (Wu - Wc) + y * (Wa - Wc) for (x, y) in pts.values()]
    Ypts = sim_many(np.stack(Wpts), P.x_eval)
    names = ["unsafe request $u$", "enforced, as written", "enforced, improved", "safe field $c$", "halfway between $c$ and $a$",
             "safe anchor $a$"]
    for k in range(6):
        ax_ = fig.add_subplot(gs[k // 3, 3 + k % 3])
        scatter_out(ax_, Ypts[k], s=0.45, alpha=0.4); draw_A(ax_, lw=0.5); plane_box(ax_, ticks=False)
        ax_.set_title(f"{k + 1}: {names[k]}", fontsize=5.8, loc="left", pad=2)
        corner(ax_, f"$P(A)={pA(Ypts[k]):.3f}$", size=5.6)
        mac(f"ewSlicePA{'abcdef'[k]}", f"{pA(Ypts[k]):.3f}")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.12)
    F.save(fig, "fig_e1_slice")
    insideU = C["sl_S_U-L2"] <= 1; insideK = C["sl_S_kNN1-tr1"] <= 1
    mac("ewSliceMaxU", f"{pAg[insideU].max():.2f}"); mac("ewSliceMaxK", f"{pAg[insideK].max():.3f}")
    mac("ewKapU", f"{kU:.2f}"); mac("ewKapK", f"{kK:.2f}")
    mac("ewAlphaU", f"{1 - kU:.2f}"); mac("ewAlphaK", f"{1 - kK:.2f}")
    mac("ewRotSlice", f"{np.degrees(Z[c, 0] - Z[a, 0]):.0f}")


def fig_certificate():
    """Step 7: the certificate of the enforced field, as a transport argument and as a function of r."""
    ju = J_EX[0.3]
    RG = np.linspace(0.02, 2.0, 199)
    Te_anc = P.D[f"{CFGM}_Te"][ANC].astype(float)
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
    res = {}
    for name in ["U-L2", "kNN1-tr1"]:
        Wf, kap, c, reg = enforce(name, ju)
        Y = P.outputs(Wf, CFGM, P.x_eval); Yp = P.outputs(Wf, CFGM, P.x_probe)
        if name == "U-L2":
            q = np.array([np.mean(np.linalg.norm(Te_anc, axis=-1) < r_A + r) for r in RG])
            budget = reg.eps / RG; Yref = None
        else:
            i = int(np.argmin(reg.dist_T(Yp[None])[0]))
            q = np.array([np.mean(np.linalg.norm(Te_anc[i], axis=-1) < r_A + r) for r in RG])
            budget = reg.eps * np.maximum(1.0, reg.par / RG); Yref = Te_anc[i]
        bound = q + budget; kb = int(np.argmin(bound))
        res[name] = dict(Y=Y, Yref=Yref, q=q, budget=budget, bound=bound, kb=kb, eps=reg.eps, pA=pA(Y))
    # (a) the transport argument for the improved field: outputs in A matched with the nearest safe field's outputs
    R = res["kNN1-tr1"]; rstar = RG[R["kb"]]; Y, Yr = R["Y"], R["Yref"]
    ax = axs[0]
    ax.scatter(Yr[:, 0], Yr[:, 1], s=0.5, color=MUTED, alpha=0.35, linewidths=0, rasterized=True)
    mA = in_A(Y); near = np.linalg.norm(Yr, axis=1) < r_A + rstar
    for i in np.where(mA)[0]:
        far_ = np.linalg.norm(Y[i] - Yr[i]) >= rstar
        ax.plot([Y[i, 0], Yr[i, 0]], [Y[i, 1], Yr[i, 1]], lw=0.5, color=ORANGE if far_ else INK2, alpha=0.8)   # orange: pair >= r apart
    ax.scatter(Y[mA, 0], Y[mA, 1], s=2.5, color=CRIT, linewidths=0, zorder=4)
    draw_A(ax, plus=rstar); plane_box(ax, ticks=False)
    ax.set_title("(a) outputs in $A$, matched by noise", loc="left", fontsize=6.5)
    corner(ax, f"$A^{{+r}}$, $r={rstar:.2f}$ (dashed)", loc="ul", size=5.6)
    nfar = np.mean([np.linalg.norm(Y[i] - Yr[i]) >= rstar for i in np.where(mA)[0]])
    for k, (name, title) in enumerate([("U-L2", "(b) as written: Markov"), ("kNN1-tr1", "(c) improved: truncated")]):
        R = res[name]; ax = axs[1 + k]
        ax.plot(RG, R["q"], color=MUTED, lw=1.1, label="safe outputs within $r$ of $A$")
        ax.plot(RG, R["budget"], color=ORANGE, lw=1.1, label="budget term")
        ax.plot(RG, R["bound"], color=BLUE if k == 0 else AQUA, lw=1.6, label="certificate (sum)")
        ax.scatter([RG[R["kb"]]], [R["bound"][R["kb"]]], s=14, color=BLUE if k == 0 else AQUA, zorder=4)
        ax.axhline(R["pA"], color=CRIT, lw=0.9, ls=(0, (3, 2)))
        ax.text(1.98, R["pA"] + 0.02, f"actual $P(A)={R['pA']:.3f}$", fontsize=5.6, color=CRIT, ha="right")
        ax.text(RG[R["kb"]], R["bound"][R["kb"]] + 0.05, f"{R['bound'][R['kb']]:.2f}", fontsize=6, ha="center", color=INK)
        ax.set_ylim(0, 1.05); ax.set_xlim(0, 2.0); ax.grid(True, axis="y")
        ax.set_xlabel("enlargement $r$", fontsize=6.5); ax.set_title(title, loc="left", fontsize=6.5)
        mac(f"ewCert{'U' if k == 0 else 'K'}", f"{R['bound'][R['kb']]:.2f}")
        mac(f"ewCertR{'U' if k == 0 else 'K'}", f"{RG[R['kb']]:.2f}")
        mac(f"ewCertQ{'U' if k == 0 else 'K'}", f"{R['q'][R['kb']]:.2f}")
        mac(f"ewCertPA{'U' if k == 0 else 'K'}", f"{R['pA']:.3f}")
    axs[1].legend(loc="center left", fontsize=5.4, handlelength=1.2)
    fig.tight_layout(w_pad=0.6)
    F.save(fig, "fig_e1_certificate")
    mac("ewCertBudgetK", f"{res['kNN1-tr1']['eps']:.3f}")
    mac("ewCertFar", f"{100 * nfar:.0f}")


def fig_attack(C):
    """Step 8: the worst prompt found by the attack inside each region, and the result of every attack run."""
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 1.85), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.5]))
    ttl = {"E-U": "prompt-space region", "U-L2": "field region, as written", "kNN1-tr1": "field region, improved"}
    for ax, name, tag in zip(axs[:3], ["E-U", "U-L2", "kNN1-tr1"], ["(a)", "(b)", "(c)"]):
        Y = P.outputs(C[f"at_W_{name}"], CFGM, P.x_eval[600:])
        scatter_out(ax, Y, s=0.6); draw_A(ax); plane_box(ax, ticks=False)
        ax.set_title(f"{tag} {ttl[name]}", loc="left", fontsize=6.2)
        corner(ax, f"$P(A)={pA(Y):.3f}$", size=6)
        mac(f"ewAtk{dict(zip(['E-U', 'U-L2', 'kNN1-tr1'], ['Prompt', 'Written', 'Improved']))[name]}", f"{pA(Y):.3f}")
    ax = axs[3]
    order = [("E-U", MUTED), ("U-L2", BLUE), ("kNN1-tr1", AQUA)]
    for k, (name, col) in enumerate(order):
        v = C[f"at_pA_{name}"]
        x = k + 0.25 * (np.random.default_rng(k).random(len(v)) - 0.5)
        ax.scatter(x, v, s=9, color=col, linewidths=0, alpha=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(["prompt\nspace", "as\nwritten", "improved"], fontsize=6)
    ax.set_ylim(0, 1.05); ax.set_ylabel("$P(A)$ found", fontsize=6.5); ax.grid(True, axis="y")
    ax.set_title("(d) all 12 attack runs", loc="left", fontsize=6.3)
    # the prompt found against the prompt-space region: how close its embedding is, and how the field regions score it
    e, Wt = C["at_e_E-U"], C["at_W_E-U"]
    dE = np.linalg.norm(E[ANC] - e, axis=1)
    dAA = np.linalg.norm(E[ANC][:, None] - E[ANC][None], axis=-1)[np.triu_indices(len(ANC), 1)]
    mac("ewAtkEmbNear", f"{dE.min():.2f}"); mac("ewAtkEmbMed", f"{np.median(dAA):.2f}")
    for name, tag in [("U-L2", "U"), ("kNN1-tr1", "K")]:
        reg = P.Region(name, CFGM, ANC, CAL)
        mac(f"ewAtkSc{tag}", f"{reg.score_field(Wt) / reg.eps:.1f}")
    fig.tight_layout(w_pad=0.4)
    F.save(fig, "fig_e1_attack")


def main():
    recompute = "--recompute" in sys.argv
    C = build_cache() if (recompute or not os.path.exists(CACHE)) else dict(np.load(CACHE))
    global D_TRUE
    D_TRUE = {0.0: float(np.mean(in_A(sample_target(Z_SAFE, 200000, np.random.default_rng(5)))))}
    D_TRUE.update({m: float(P.D["true_pA"][J_EX[m]]) for m in J_EX})
    mac("ewPhi", f"{np.degrees(Z_SAFE[0]):.1f}"); mac("ewRho", f"{Z_SAFE[1]:.2f}")
    # budget of the uniform and of the nearest-field region with the same L^2 map metric (median over 100 splits)
    scr = json.load(open(os.path.join(P.RES, "improve_screen.json")))
    for cfg, tag in [("rect1", "One"), ("rect2", "Two")]:
        bu, bk = [np.median([r["eps"] for r in scr if r["cfg"] == cfg and r["p"] == 2 and r["kind"] == kd and r["par"] == pr])
                  for kd, pr in [("U", None), ("kNN", 1)]]
        mac(f"ewBudRatio{tag}", f"{bu / bk:.1f}")
    for m, t in [(0.03, "b"), (0.1, "c"), (0.3, "d")]:
        mac(f"ewTruePA{t}", f"{D_TRUE[m]:.3f}")
    fig_task(); fig_fields(); fig_time(); fig_coupling(); fig_scores(); fig_slice(C); fig_certificate(); fig_attack(C)
    with open(os.path.join(TAB, "e1_walk_numbers.tex"), "w") as f:
        f.write("% written by code/e1_walkthrough.py\n")
        for k, v in NUM.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(NUM)} macros to tables/e1_walk_numbers.tex")


D_TRUE = {}
if __name__ == "__main__":
    main()
