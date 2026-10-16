"""Figures, table bodies and quoted numbers of the E2 study (safe_prompt_fields.tex, Section 10): static and adaptive
order-1 regions on the prompt embeddings and on the fields, from ../results/e2/.
Writes ../figures/fig_e2_*.{pdf,png}, ../tables/e2_summary.tex and ../tables/e2_numbers.tex.
Usage: python e2_figures.py
"""
import os, json
import numpy as np
import e1_figures as F                                    # style, colours and save() shared with the other figures
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
RES, TAB = os.path.join(HERE, "..", "results", "e2"), os.path.join(HERE, "..", "tables")
REG = json.load(open(os.path.join(RES, "regions.json")))
EA = json.load(open(os.path.join(RES, "enforce_attack.json")))
NU = json.load(open(os.path.join(RES, "nuisance.json")))
MAPS = np.load(os.path.join(RES, "maps.npz"))
NMAPS = np.load(os.path.join(RES, "nuisance_maps.npz"))
POOL = np.load(os.path.join(RES, "pool.npz"))
DIST = np.load(os.path.join(RES, "dist.npz"))
P2 = json.load(open(os.path.join(RES, "pool_config.json")))
MASSES = P2["masses"]
Z, ZG = POOL["Z"], POOL["Zg"]
PHI_G, M_G = np.array(P2["grid_phi_deg"]), np.array(P2["grid_m"])
# colour roles: prompt region orange, field region blue, pull-back region of the reflowed one-step model aqua, pull-back
# region of the 1-rectified model (its slice at t = 0) yellow; static dashed / hollow, adaptive solid / filled.
# The names are those of Table tab:e2_regions of the note.
COL = {"E": F.ORANGE, "TR:rect1": F.BLUE, "PB:rect2": F.AQUA, "PB:rect1": F.YELLOW, "TR:rect2_1step": F.BLUE, "L2:rect1": F.MUTED}
NAME = {"E": "prompt", "TR:rect1": "field (1-rectified)", "PB:rect2": "pull-back (reflowed)",
        "PB:rect1": "pull-back (1-rectified)", "TR:rect2_1step": "field (reflowed)", "L2:rect1": "field, $L^2$ (1-rectified)"}
NUM = {}


def mac(k, v):
    NUM[k] = v


def best(sp, M=30):
    return REG["best"][f"{sp}|{M}"]


def ev(sp, vn, M=30):
    return REG["eval"][f"{sp}|{M}|{vn}"]


def weights(dists, Daa, vn):
    """Transport-plan weights of the order-1 score (as in e2_regions.score)."""
    if vn == "U":
        return np.full(len(dists), 1.0 / len(dists))
    k, p = vn.split(":")
    if k == "N":
        w = np.zeros(len(dists)); w[np.argsort(dists)[:int(p)]] = 1.0 / int(p); return w
    h = float(p) * np.median(Daa[np.triu_indices(len(Daa), 1)])
    L = -dists ** 2 / (2 * h * h); L -= L.max(); w = np.exp(L)
    return w / w.sum()


# ---------------------------------------------------------------------------------------------------------- figures
def fig_mechanism():
    """One column per region (prompt or field, static or adaptive weights; 1-rectified model; evaluation split 0). Top: where
    the optimal transport plan of the order-1 score takes its mass from, for one unsafe request. Bottom: which requests
    (rotation, unsafe mass; radius 2) the region admits, with the same request marked."""
    anc = MAPS["anc"]; nP = len(Z)
    i_phi, i_m = int(np.argmin(np.abs(PHI_G - 10))), int(np.argmin(np.abs(M_G - 0.1)))
    gi = i_m * len(PHI_G) + i_phi
    cols = [("E", "static"), ("E", "adaptive"), ("TR:rect1", "static"), ("TR:rect1", "adaptive")]
    fig, axs = plt.subplots(2, 4, figsize=(6.5, 3.75), sharex="row", sharey="row")
    band = np.abs(PHI_G) <= 22.5
    for c, (sp, tag) in enumerate(cols):
        key = "E" if sp == "E" else "F"
        ax = axs[0, c]                                           # top: transport weights of the request
        Dm = DIST[sp]; d = Dm[nP + gi, anc].astype(float); Daa = Dm[np.ix_(anc, anc)].astype(float)
        vn = "U" if tag == "static" else best(sp)
        w = weights(d, Daa, vn)
        sc = float(MAPS[f"{sp}|{tag}"][i_m, i_phi])
        ax.scatter(np.degrees(Z[anc, 0]), Z[anc, 1], s=6 + 520 * w,                     # area proportional to the mass moved
                   color=COL[sp], alpha=0.85 if tag == "adaptive" else 0.45, linewidths=0.3, edgecolors="white", zorder=3)
        ax.scatter([10], [2.0], marker="*", s=70, color=F.CRIT, zorder=4, linewidths=0)
        ax.set_xlim(-26, 26); ax.set_ylim(1.7, 2.3); ax.grid(True, axis="both")
        ax.set_title(f"({'abcd'[c]}) {'prompt' if sp == 'E' else 'field'}, {tag}", loc="left", fontsize=6.5)
        ax.text(0.03, 0.04, f"score $={sc:.2f}\\varepsilon$: {'admitted' if sc <= 1 else 'excluded'}", transform=ax.transAxes,
                fontsize=5.8, color=F.INK2)
        ax.set_xlabel("rotation of the anchor ($^\\circ$)", fontsize=6)
        if tag == "adaptive":
            mac(f"etW{key}top", f"{np.sort(w)[-1]:.2f}")
            mac(f"etW{key}topthree", f"{np.sort(w)[-3:].sum():.2f}")
        mac(f"etSc{key}{tag}", f"{sc:.2f}")
        ax = axs[1, c]                                           # bottom: admission map, same request marked
        S = MAPS[f"{sp}|{tag}"]; adm = (S <= 1.0).astype(float)
        ax.pcolormesh(PHI_G, M_G, adm, cmap=ListedColormap(["#ffffff", COL[sp]]), vmin=0, vmax=1, shading="nearest",
                      alpha=0.45, rasterized=True)
        ax.contour(PHI_G, M_G, S, levels=[1.0], colors=COL[sp], linewidths=1.0)
        ax.plot([-22.5, 22.5], [0, 0], color=F.INK, lw=2.2, solid_capstyle="butt", zorder=4, clip_on=False)
        ax.plot(np.degrees(Z[anc, 0]), np.full(len(anc), 0.3), "|", color=F.MUTED, ms=3, mew=0.6, clip_on=False)
        ax.scatter([10], [0.1], marker="*", s=70, color=F.CRIT, zorder=5, linewidths=0)
        ax.set_xlim(-45, 45); ax.set_ylim(0, 0.3); ax.set_xticks([-40, -20, 0, 20, 40]); ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_title(f"({'efgh'[c]})", loc="left", fontsize=6.5)
        ax.set_xlabel("rotation of the request ($^\\circ$)", fontsize=6)
        frac = float(adm[M_G >= 0.05 - 1e-9].mean())
        ax.text(0.03, 0.82, f"admitted with\n$m\\geq0.05$: {100 * frac:.0f}%", transform=ax.transAxes, fontsize=5.5,
                color=F.INK2, ha="left", va="top")
        k2 = "E" if sp == "E" else "TR"
        mac(f"etMapMax{k2}{tag}", f"{M_G[adm.any(1)].max():.2f}")              # largest admitted mass
        mac(f"etMap{k2}{tag}", f"{100 * frac:.0f}")
        mac(f"etMapSafe{k2}{tag}", f"{100 * adm[0, band].mean():.0f}")
    axs[0, 0].set_ylabel("radius of the anchor", fontsize=6.5)
    axs[1, 0].set_ylabel("unsafe mass $m$ of the request", fontsize=6.5)
    for ax in axs.ravel():
        ax.tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    F.save(fig, "fig_e2_mechanism")


def fig_detection():
    """(a) Exclusion of unsafe prompts against their mass, prompt and field regions, static and adaptive weights (evaluation
    pool, 200 splits, M = 30); (b)-(c) mean exclusion over the four masses against the parameter of the adaptive weights."""
    from matplotlib.lines import Line2D
    cs = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]; ks = [1, 2, 3, 5, 10]
    sps = ["E", "TR:rect1", "PB:rect1", "PB:rect2"]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3), gridspec_kw=dict(width_ratios=[1, 1.3, 0.85]))
    ax = axs[0]
    for sp in ["E", "TR:rect1"]:
        for tag, ls in [("static", (0, (2.5, 1.5))), ("adaptive", "-")]:
            y = [ev(sp, "U" if tag == "static" else best(sp))[f"excl_{m}"] for m in MASSES]
            ax.plot(MASSES, y, color=COL[sp], ls=ls, lw=1.4, marker="o", ms=3.5, markeredgecolor=COL[sp], markeredgewidth=0.8,
                    markerfacecolor="white" if tag == "static" else COL[sp])
    ax.axhline(1 - np.mean([ev(sp, "U")["coverage"] for sp in ["E", "TR:rect1"]]), color=F.GRID, lw=1.0, zorder=0)
    ax.text(0.29, 0.035, "$1-$coverage", fontsize=5.5, color=F.MUTED, ha="right")
    ax.text(0.068, 0.9, "adaptive", fontsize=6, color=F.INK2, ha="right")
    ax.text(0.11, 0.24, "static", fontsize=6, color=F.INK2, ha="left")
    ax.set_xscale("log"); ax.set_xticks(MASSES); ax.set_xticklabels([f"{m:g}" for m in MASSES]); ax.minorticks_off()
    ax.set_ylim(0, 1.04); ax.grid(True, axis="y")
    ax.set_xlabel("unsafe mass $m$", fontsize=6.5); ax.set_ylabel("unsafe prompts excluded", fontsize=6.5)
    ax.set_title("(a) exclusion, $M=30$", loc="left", fontsize=6.8)
    xs = np.r_[np.arange(len(cs)), len(cs) + 0.6]; xk = np.r_[np.arange(len(ks)), len(ks) + 0.6]
    for sp in sps:
        ys = [ev(sp, f"K:{c:g}")["J"] for c in cs] + [ev(sp, "U")["J"]]
        axs[1].plot(xs, ys, color=COL[sp], lw=1.4, marker="o", ms=3, markeredgecolor="white", markeredgewidth=0.5)
        b = best(sp)
        if b.startswith("K"):
            j = cs.index(float(b.split(":")[1]))
            axs[1].scatter([xs[j]], [ys[j]], s=36, facecolors="none", edgecolors=F.INK, linewidths=0.8, zorder=4)
        ys2 = [ev(sp, f"N:{k}")["J"] for k in ks] + [ev(sp, "U")["J"]]
        axs[2].plot(xk, ys2, color=COL[sp], lw=1.4, marker="o", ms=3, markeredgecolor="white", markeredgewidth=0.5)
    axs[1].set_xticks(xs); axs[1].set_xticklabels([f"{c:g}" for c in cs] + ["static"], fontsize=6)
    axs[2].set_xticks(xk); axs[2].set_xticklabels([str(k) for k in ks] + ["static"], fontsize=6)
    axs[1].set_xlabel("kernel bandwidth $c$", fontsize=6.5)
    axs[2].set_xlabel("number $k$ of nearest anchors", fontsize=6.5)
    axs[1].set_ylabel("mean exclusion, four masses", fontsize=6.5)
    axs[1].set_title("(b) tuning: Gaussian kernel weights", loc="left", fontsize=6.8)
    axs[2].set_title("(c) tuning: $k$ nearest anchors", loc="left", fontsize=6.8)
    for a in axs[1:]:
        a.grid(True, axis="y"); a.set_ylim(0.3, 0.72)
    axs[2].set_yticklabels([])
    hs = [Line2D([], [], color=COL[sp], lw=1.4, label=NAME[sp]) for sp in sps]
    hs += [Line2D([], [], color=F.INK2, lw=1.2, ls=(0, (2.5, 1.5)), marker="o", ms=3, markerfacecolor="white", label="static"),
           Line2D([], [], color=F.INK2, lw=1.2, marker="o", ms=3, label="adaptive")]
    fig.legend(handles=hs, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07), fontsize=5.8, handlelength=2.0,
               columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    F.save(fig, "fig_e2_detection")
    gap = max(abs(REG["tune"][k]["J"] - REG["eval"][k]["J"]) for k in REG["tune"] if k.split("|")[1] == "30")
    mac("etTuneGap", f"{gap:.2f}")
    regret = max(max(REG["eval"][k]["J"] for k in REG["eval"] if k.startswith(f"{sp}|30|"))
                 - ev(sp, best(sp))["J"] for sp in sps)
    mac("etTuneRegret", f"{regret:.3f}")
    near = max(max(REG["eval"][k]["J"] for k in REG["eval"] if k.startswith(f"{sp}|30|")) - ev(sp, "N:1")["J"]
               for sp in sps + ["TR:rect2_1step"])                           # the nearest anchor alone against the best weights
    mac("etNearGap", f"{near:.3f}")


def fig_density():
    """Exclusion of small unsafe masses against the number of safe anchors (evaluation pool)."""
    Ms = [15, 30, 60, 120, 240]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.2), sharey=True)
    for ax, m in zip(axs, [0.03, 0.01]):
        for sp in ["E", "TR:rect1", "PB:rect2"]:
            for tag, ls in [("static", (0, (2.5, 1.5))), ("adaptive", "-")]:
                y = [ev(sp, "U" if tag == "static" else best(sp, M), M)[f"excl_{m}"] for M in Ms]
                ax.plot(Ms, y, color=COL[sp], ls=ls, lw=1.4, marker="o", ms=3.5,
                        markerfacecolor="white" if tag == "static" else COL[sp], markeredgecolor=COL[sp], markeredgewidth=0.8,
                        label=f"{NAME[sp]}, {tag}")
        ax.set_xscale("log"); ax.set_xticks(Ms); ax.set_xticklabels([str(M) for M in Ms])
        ax.minorticks_off()
        ax.set_ylim(0, 1.04); ax.grid(True, axis="y"); ax.set_xlabel("number $M$ of anchors", fontsize=6.5)
        ax.set_title(f"({'ab'[[0.03, 0.01].index(m)]}) unsafe mass $m={m:g}$", loc="left", fontsize=6.8)
    axs[0].set_ylabel("unsafe prompts excluded", fontsize=6.5)
    h, l = axs[0].get_legend_handles_labels()
    axs[1].legend(h, l, loc="upper left", fontsize=5.4, handlelength=2.2)
    fig.tight_layout(w_pad=0.6)
    F.save(fig, "fig_e2_density")
    for sp, t in [("E", "E"), ("TR:rect1", "F"), ("PB:rect2", "P")]:
        for M in [30, 240]:
            mac(f"etDens{t}{'A' if M == 30 else 'B'}", f"{ev(sp, best(sp, M), M)['excl_0.03']:.2f}")
            mac(f"etDensU{t}{'A' if M == 30 else 'B'}", f"{ev(sp, 'U', M)['excl_0.03']:.2f}")
            mac(f"etEps{t}{'A' if M == 30 else 'B'}", f"{ev(sp, best(sp, M), M)['eps_median']:.2f}")
            mac(f"etEpsU{t}{'A' if M == 30 else 'B'}", f"{ev(sp, 'U', M)['eps_median']:.2f}")
        mac(f"etDensOne{t}", f"{ev(sp, best(sp, 240), 240)['excl_0.01']:.2f}")


def cell_rows(cfg, sp, tag):
    B = [b for b in EA["enforce"] if (b["cfg"], b["space"], b["weights"]) == (cfg, sp, tag)]
    rows = [r for b in B for r in b["rows"]]
    A = [a for a in EA["attack"] if (a["cfg"], a["space"], a["weights"]) == (cfg, sp, tag)][0]
    return rows, A


def fig_attack():
    """Worst prompt found by the white-box attack inside each region, and enforcement of honest unsafe prompts."""
    groups = [("rect1", ["E", "PB:rect1", "TR:rect1"]), ("rect2_1step", ["E", "PB:rect2", "TR:rect2_1step"])]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    x = 0; ticks, labels = [], []
    for g, (cfg, sps) in enumerate(groups):
        for sp in sps:
            for tag in ["static", "adaptive"]:
                rows, A = cell_rows(cfg, sp, tag)
                v = np.array([r["pA_fresh"] for r in A["runs"]])
                xx = x + 0.22 * (np.random.default_rng(int(10 * x)).random(len(v)) - 0.5)
                axs[0].scatter(xx, v, s=7, color=COL[sp], alpha=0.9 if tag == "adaptive" else 0.55, linewidths=0.5,
                               edgecolors=COL[sp], facecolors=COL[sp] if tag == "adaptive" else "white")
                pa = np.mean([r["pA"] for r in rows if r["m"] == 0.3])
                axs[1].bar(x, pa, width=0.7, color=COL[sp], alpha=0.85 if tag == "adaptive" else 0.35, edgecolor=COL[sp], linewidth=0.6)
                ticks.append(x); labels.append(("S" if tag == "static" else "A"))
                key = f"{'One' if cfg == 'rect1' else 'Two'}{ {'E': 'E', 'PB:rect1': 'PB', 'TR:rect1': 'F', 'PB:rect2': 'PB', 'TR:rect2_1step': 'F'}[sp] }{tag[0].upper()}"
                mac(f"etAtk{key}", f"{v.max():.3f}"); mac(f"etAtkMed{key}", f"{np.median(v):.3f}")
                mac(f"etEnf{key}", f"{pa:.3f}")
                mac(f"etMmd{key}", f"{np.mean([r['mmd'] for r in rows if r['m'] == 0.3]):.0f}")
                x += 1
            x += 0.4
        x += 0.8
    for ax in axs:
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=5.5); ax.grid(True, axis="y")
    axs[0].set_ylim(0, 1.04); axs[0].set_ylabel("$P(A)$ of the best embedding found", fontsize=6.3)
    axs[0].set_title("(a) white-box attack (12 runs per region)", loc="left", fontsize=6.6)
    axs[1].set_ylabel("$P(A)$ after back-off, $m=0.3$", fontsize=6.3)
    axs[1].set_title("(b) enforcement of honest requests", loc="left", fontsize=6.6)
    for ax in axs:
        ax.text(0.22, -0.2, "1-rectified model, 50 steps", transform=ax.transAxes, fontsize=5.8, ha="center", color=F.INK2)
        ax.text(0.75, -0.2, "reflowed model, one step", transform=ax.transAxes, fontsize=5.8, ha="center", color=F.INK2)
    from matplotlib.patches import Patch
    hs = [Patch(color=COL[s], label=NAME[s]) for s in ["E", "PB:rect1", "PB:rect2"]] + [Patch(color=COL["TR:rect1"], label="field")]
    fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=5.6, handlelength=1.0, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8)
    F.save(fig, "fig_e2_attack")


def fig_nuisance():
    """Prompt encoder with a nuisance token: exclusion against its strength, and admission maps over (rotation, token)."""
    lams = [0, 0.5, 1, 2, 4]
    fig = plt.figure(figsize=(6.5, 2.3))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.6, 0.25, 1, 1, 1], hspace=0.35, wspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    for tag, ls in [("static", (0, (2.5, 1.5))), ("adaptive", "-")]:
        y = [NU["eval"][f"E:{l:g}|30|{'U' if tag == 'static' else NU['best'][f'E:{l:g}|30']}"]["excl_0.1"] for l in lams]
        ax.plot(lams, y, color=COL["E"], ls=ls, lw=1.4, marker="o", ms=3.5, markerfacecolor="white" if tag == "static" else COL["E"],
                markeredgecolor=COL["E"], markeredgewidth=0.8, label=f"prompt, {tag}")
        for sp in ["TR:rect1", "PB:rect1"]:
            vn = "U" if tag == "static" else NU["best"][f"{sp}|30"]
            yv = NU["eval"][f"{sp}|30|{vn}"]["excl_0.1"]
            ax.plot(lams, [yv] * len(lams), color=COL[sp], ls=ls, lw=1.2,
                    label=f"{'field' if sp.startswith('TR') else 'pull-back'}, {tag}")
    ax.set_ylim(0, 1.04); ax.grid(True, axis="y"); ax.set_xlabel("scale $\\lambda$ of the nuisance token", fontsize=6.5)
    ax.set_ylabel("unsafe prompts excluded, $m=0.1$", fontsize=6.5); ax.set_title("(a) exclusion, $M=30$", loc="left", fontsize=6.6)
    ax.legend(loc="center right", fontsize=5.0, handlelength=2.0, bbox_to_anchor=(1.0, 0.58))
    phi, s = NMAPS["phi_deg"], NMAPS["s"]
    for r, mi in enumerate([0, 1]):
        for c, sp in enumerate(["E:2", "TR:rect1", "PB:rect1"]):
            a = fig.add_subplot(gs[r, 2 + c]); S = NMAPS[f"{sp}|adaptive"][mi]
            colr = COL["E" if sp.startswith("E") else sp]
            a.pcolormesh(phi, s, (S <= 1).astype(float), cmap=ListedColormap(["#ffffff", colr]), vmin=0, vmax=1,
                         shading="nearest", alpha=0.45, rasterized=True)
            a.contour(phi, s, S, levels=[1.0], colors=colr, linewidths=0.9)
            a.axvline(-22.5, color=F.MUTED, lw=0.5, ls=(0, (2, 2))); a.axvline(22.5, color=F.MUTED, lw=0.5, ls=(0, (2, 2)))
            a.set_xlim(-45, 45); a.set_ylim(-1, 1); a.set_xticks([-40, 0, 40]); a.set_yticks([-1, 0, 1])
            a.tick_params(labelsize=5.5)
            if c > 0:
                a.set_yticklabels([])
            if r == 0:
                a.set_xticklabels([]); a.set_title(["(b) prompt, $\\lambda=2$", "(c) field", "(d) pull-back"][c], loc="left", fontsize=6.2)
            if c == 0:
                a.set_ylabel(f"$m={[0, 0.1][mi]:g}$\ntoken $s$", fontsize=6)
            if r == 1:
                a.set_xlabel("rotation", fontsize=6)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.17)
    F.save(fig, "fig_e2_nuisance")


# ---------------------------------------------------------------------------------------------------------- tables, numbers
def tables():
    """Body of Table tab:e2_summary: one block per protected model, rows named as in Table tab:e2_regions."""
    lines = []
    for cfg, title, sps in [("rect1", "regions protecting the 1-rectified model (50 Euler steps)", ["E", "PB:rect1", "TR:rect1"]),
                            ("rect2_1step", "regions protecting the reflowed model (one Euler step)", ["E", "PB:rect2", "TR:rect2_1step"])]:
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{title}}}}}\\\\")
        for sp in sps:
            for tag in ["static", "adaptive"]:
                vn = "U" if tag == "static" else best(sp)
                r = ev(sp, vn); rows, A = cell_rows(cfg, sp, tag)
                lab = {"E": "prompt", "PB:rect1": "pull-back", "PB:rect2": "pull-back", "TR:rect1": "field", "TR:rect2_1step": "field"}[sp]
                if vn == "U":
                    wl = "static"
                else:
                    k, p = vn.split(":")
                    wl = {"K": f"adaptive, $c={p}$", "N": f"adaptive, $k={p}$", "X": f"adaptive, $c={p}$, prompt"}[k]
                cells = [f"{r['coverage']:.3f}"] + [f"{r[f'excl_{m}']:.2f}" for m in MASSES]
                cells += [f"{np.mean([q['pA'] for q in rows if q['m'] == 0.3]):.3f}", f"{np.mean([q['mmd'] for q in rows if q['m'] == 0.3]):.0f}",
                          f"{max(q['pA_fresh'] for q in A['runs']):.3f}"]
                lines.append(f"{lab if tag == 'static' else ''} & {wl} & " + " & ".join(cells) + " \\\\")
        if cfg == "rect1":
            lines.append("\\midrule")
    open(os.path.join(TAB, "e2_summary.tex"), "w").write("\n".join(lines) + "\n")


def numbers():
    for sp, t in [("E", "E"), ("TR:rect1", "F"), ("PB:rect2", "P"), ("PB:rect1", "S"), ("L2:rect1", "L"), ("TR:rect2_1step", "T")]:
        for tag in ["static", "adaptive"]:
            r = ev(sp, "U" if tag == "static" else best(sp))
            for m, mt in zip(MASSES, "abcd"):
                mac(f"etEx{t}{tag[0].upper()}{mt}", f"{r[f'excl_{m}']:.2f}")
            mac(f"etJ{t}{tag[0].upper()}", f"{r['J']:.2f}"); mac(f"etCov{t}{tag[0].upper()}", f"{r['coverage']:.3f}")
        mac(f"etBest{t}", best(sp).replace("K:", "").replace("X:", "").replace("N:", ""))
    NS = int(POOL["n_safe"])                                   # spread of the safe fields and displacement of an unsafe mode
    for sp, t in [("TR:rect1", "One"), ("TR:rect2_1step", "Two")]:
        Mx = DIST[sp]
        mac(f"etSpread{t}", f"{Mx[300:NS, :NS].mean():.2f}")
        rows = NS + np.where(Z[NS:, 2] == 0.1)[0]
        mac(f"etDisp{t}", f"{np.median(Mx[rows, :NS].min(1)):.2f}")
    covs = [REG["eval"][k]["coverage"] for k in REG["eval"] if k.split("|")[1] == "30"]
    mac("etCovMin", f"{min(covs):.3f}"); mac("etCovMax", f"{max(covs):.3f}")
    uns = NS + np.where((Z[NS:, 2] == 0.3) & (POOL["base_idx"] >= P2["n_bases_tune"]))[0]   # evaluation requests, m = 0.3
    for cfg, t in [("rect1", "One"), ("rect2_1step", "Two")]:
        mac(f"etPoolPA{t}", f"{POOL[f'{cfg}_pA'][uns].mean():.3f}")                      # P(A) before enforcement
    covs = [NU["eval"][k]["coverage"] for k in NU["eval"] if k.split("|")[1] == "30"]
    mac("etNuCovMin", f"{min(covs):.3f}"); mac("etNuCovMax", f"{max(covs):.3f}")
    fid = NU["fidelity"]
    mac("etNuSafePA", f"{fid['safe_pA']:.4f}"); mac("etNuPAd", f"{fid['pA_0.3']:.3f}"); mac("etNuMmd", f"{fid['mmd_safe']:.1f}")
    mac("etNuSame", f"{fid['tr_same_content_median']:.3f}"); mac("etNuPairs", f"{fid['tr_safe_pairs_median']:.3f}")
    for lam in ["0", "2", "4"]:
        for tag in ["static", "adaptive"]:
            vn = "U" if tag == "static" else NU["best"][f"E:{lam}|30"]
            mac(f"etNuE{['zero', 'two', 'four'][['0', '2', '4'].index(lam)]}{tag[0].upper()}", f"{NU['eval'][f'E:{lam}|30|{vn}']['excl_0.1']:.2f}")
    for sp, t in [("TR:rect1", "F"), ("PB:rect1", "P")]:
        for tag in ["static", "adaptive"]:
            vn = "U" if tag == "static" else NU["best"][f"{sp}|30"]
            mac(f"etNu{t}{tag[0].upper()}", f"{NU['eval'][f'{sp}|30|{vn}']['excl_0.1']:.2f}")
    with open(os.path.join(TAB, "e2_numbers.tex"), "w") as f:
        f.write("% written by code/e2_figures.py\n")
        for k, v in NUM.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(NUM)} macros")


if __name__ == "__main__":
    fig_mechanism(); fig_detection(); fig_density(); fig_attack(); fig_nuisance()
    tables(); numbers()
