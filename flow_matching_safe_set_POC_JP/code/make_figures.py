"""All figures of the note (PDF for LaTeX + PNG previews) from the cached results in ../results.
Colour roles (fixed across figures): fixed-reference / forward cost = blue, on-policy / reverse cost = orange,
reverse sqrt-KL (pilot A only) = aqua, baselines = greys, unsafe set A = status red (always labelled)."""
import os, json, glob, numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CRIT = "#d03b3b"
GREY_LIGHT = "#b9b8b1"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "lines.linewidth": 1.5, "legend.frameon": False, "savefig.dpi": 300,
    "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
r_A = 0.6

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)

def unsafe_disk(ax, label=False):
    ax.add_patch(Circle((0, 0), r_A, facecolor=CRIT, alpha=0.10, edgecolor=CRIT, linewidth=0.8, zorder=1))
    if label:
        ax.annotate("unsafe set $A$", xy=(0.42, -0.42), xytext=(1.0, -2.75), fontsize=7, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6))

def square(ax, lim=3.2):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])

# ------------------------------------------------------------------ Fig 1: task family
tp = np.load(os.path.join(RES, "time_profiles.npz"))
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.35))
rng = np.random.default_rng(0)
for i, col in enumerate([BLUE, ORANGE, AQUA]):
    mu = tp["show_means"][i]; comp = rng.integers(0, 4, 700)
    x = mu[comp] + 0.3 * rng.standard_normal((700, 2))
    axs[0].scatter(x[:, 0], x[:, 1], s=1.2, color=col, alpha=0.55, linewidths=0, rasterized=True, label=f"task {i + 1}")
unsafe_disk(axs[0], label=True)
axs[0].set_title("(a) three tasks of the family")
lg = axs[0].legend(loc="upper left", markerscale=5, handletextpad=0.2, borderaxespad=0.0, fontsize=6.5)
x = tp["chain_samples"][0]
axs[1].scatter(x[:, 0], x[:, 1], s=1.0, color=BLUE, alpha=0.45, linewidths=0, rasterized=True)
unsafe_disk(axs[1]); axs[1].set_title("(b) exact-score chain, task 1")
x = tp["chain_samples_bad"]; ins = np.linalg.norm(x, axis=1) <= r_A
axs[2].scatter(x[~ins, 0], x[~ins, 1], s=1.0, color=INK2, alpha=0.45, linewidths=0, rasterized=True)
axs[2].scatter(x[ins, 0], x[ins, 1], s=1.0, color=CRIT, alpha=0.8, linewidths=0, rasterized=True)
unsafe_disk(axs[2]); axs[2].set_title(f"(c) contaminated, $m=0.1$: $P(A)={ins.mean():.3f}$")
for ax in axs: square(ax)
fig.tight_layout(w_pad=0.6)
save(fig, "fig_task_family")

# ------------------------------------------------------------------ Fig 2: where in time the KL accumulates
ts = tp["ts"][:-1]
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), sharey=True)
for ax, (name, title) in zip(axs, [("clean", "held-out clean field"), ("m=0.03", "contaminated, $m=0.03$"), ("m=0.1", "contaminated, $m=0.1$")]):
    for key, col, lab in [("fwd", BLUE, "forward KL"), ("rev", ORANGE, "reverse KL")]:
        c = np.cumsum(tp[f"{key}_{name}"])
        ax.plot(ts, c, color=col, label=lab)
    cf, cr = np.cumsum(tp[f"fwd_{name}"])[-1], np.cumsum(tp[f"rev_{name}"])[-1]
    ends = [(cf, cr)] if abs(cf - cr) < 0.03 else [(cf, cf), (cr, cr)]
    for val, y in ends:
        ax.annotate(f"{val:.2f}", xy=(ts[-1], y), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=INK2, va="center")
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_title(title)
    ax.grid(True, axis="y"); ax.set_xlabel("diffusion time $t$")
    ax.set_xlim(7, 0.007)
axs[1].annotate("", xy=(0.7, 0.88), xytext=(0.3, 0.88), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.8))
axs[1].text(0.5, 0.91, "generation", transform=axs[1].transAxes, ha="center", fontsize=6.5, color=MUTED)
axs[0].set_ylabel("cumulative KL, averaged\nover the 20 historical fields")
axs[0].legend(loc="upper left")
fig.tight_layout(w_pad=0.8)
save(fig, "fig_time_profiles")

# ------------------------------------------------------------------ Fig 3: exclusion of contaminated fields
A = json.load(open(os.path.join(RES, "part_a_splits.json")))
C = np.load(os.path.join(RES, "part_c_state.npz"))
Dd = json.load(open(os.path.join(RES, "part_d_splits.json")))
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
mA = np.array(A["masses"])
panels = [
    (axs[0], "(a) diffusion, exact scores", mA,
     [("F", BLUE, "forward $\\sqrt{2\\mathrm{KL}}$"), ("R", AQUA, "reverse $\\sqrt{2\\mathrm{KL}}$"), ("K", ORANGE, "reverse KL")],
     lambda k: [np.mean(A["exclusion"][k][str(m)]) for m in mA], [np.mean(A["coverage"][k]) for k in ["F", "R", "K"]]),
    (axs[1], "(b) linear FM, ODE sampler", np.array(C["masses"]),
     [("ref", BLUE, "fixed Gram $G$"), ("on", ORANGE, "on-policy Gram")],
     lambda k: list(C["excl_ref"] if k == "ref" else C["excl_on"]), [np.mean(C["cov_ref"]), np.mean(C["cov_on"])]),
    (axs[2], "(c) linear FM, SDE sampler", np.array(C["masses"]),
     [("fwd", BLUE, "forward KL"), ("rev", ORANGE, "reverse KL")],
     lambda k: [np.mean(Dd[k]["excl"][str(m)]) for m in np.array(C["masses"])], [np.mean(Dd[k]["cov"]) for k in ["fwd", "rev"]]),
]
for ax, title, ms, series, get, covs in panels:
    ax.axhline(1 - np.mean(covs), color=MUTED, linewidth=0.8)
    for k, col, lab in series:
        ax.plot(ms, get(k), color=col, marker="o", markersize=3.5, label=lab)
    ax.set_xscale("log"); ax.set_ylim(0, 1.04); ax.set_title(title); ax.grid(True, axis="y")
    ax.set_xlabel("unsafe mass $m$ of the field")
axs[0].set_ylabel("fraction of contaminated\nfields outside the region")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=BLUE, marker="o", markersize=3.5, label="fixed reference / forward cost"),
           Line2D([], [], color=ORANGE, marker="o", markersize=3.5, label="on-policy / reverse cost"),
           Line2D([], [], color=AQUA, marker="o", markersize=3.5, label="reverse $\\sqrt{2\\,\\mathrm{KL}}$ (panel a)"),
           Line2D([], [], color=MUTED, linewidth=0.8, label="safe fields excluded (1 $-$ coverage)")]
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=0.6)
save(fig, "fig_exclusion")

# ------------------------------------------------------------------ Fig 4: certificates
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7))
for ax, title, pairs in [
    (axs[0], "(a) diffusion, exact scores", [(A["inside_max"]["F_tv"], A["bounds"]["F_tv"], BLUE, "forward region, TV bound"),
                                             (A["inside_max"]["K_kl"], A["bounds"]["K_kl"], ORANGE, "reverse-KL region, kl bound")]),
    (axs[1], "(b) linear FM, SDE sampler", [(Dd["fwd"]["inside"], Dd["fwd"]["bound"], BLUE, "forward region, TV bound"),
                                            (Dd["rev"]["inside"], Dd["rev"]["bound"], ORANGE, "reverse-KL region, kl bound")])]:
    lo, hi = 2e-3, 1.0
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.8)
    ax.annotate("bound = actual", xy=(0.25, 0.25), xytext=(0.30, 0.17), fontsize=6.5, color=MUTED, rotation=33)
    for x, y, col, lab in pairs:
        ax.scatter(np.maximum(x, lo), y, s=9, color=col, alpha=0.55, linewidths=0, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("largest actual $P(A)$ among fields inside the region")
    ax.set_title(title); ax.grid(True, which="major")
    ax.legend(loc="lower right", fontsize=6.5, markerscale=1.5)
axs[0].set_ylabel("guaranteed upper bound on $P(A)$")
fig.tight_layout(w_pad=1.0)
save(fig, "fig_certificates")

# ------------------------------------------------------------------ Fig 5: time-dependent velocity fields
E = np.load(os.path.join(RES, "example_fits.npz"))
from lfm_common import Features
feat = Features(256, seed=0)                     # same features as the fits (part_c_pre.py)
g = np.linspace(-3.1, 3.1, 63); GX, GY = np.meshgrid(g, g); pts = np.c_[GX.ravel(), GY.ravel()]
times = E["times"]
rows = [("centre", "centre $\\theta_c$"), ("U_lam0.001", "least squares"), ("C_ref", "conformal, fixed $G$"), ("C_on", "conformal, on-policy")]
fig, axs = plt.subplots(len(rows), len(times), figsize=(6.5, 6.6))
data = E["data_3"]; bad = np.linalg.norm(data, axis=1) <= 1.0
for i, (k, lab) in enumerate(rows):
    W = E[f"W_{k}_3"]
    for jx, t in enumerate(times):
        ax = axs[i, jx]
        V = feat(pts, t) @ W.T
        ax.streamplot(GX, GY, V[:, 0].reshape(GX.shape), V[:, 1].reshape(GX.shape), color=INK2, linewidth=0.45,
                      density=0.9, arrowsize=0.45, arrowstyle="->")
        unsafe_disk(ax)
        ax.scatter(data[~bad, 0], data[~bad, 1], s=7, color=INK, linewidths=0, zorder=3)
        ax.scatter(data[bad, 0], data[bad, 1], s=14, color=CRIT, marker="x", linewidths=0.9, zorder=3)
        square(ax, 3.1)
        if i == 0: ax.set_title(f"$t = {t:g}$")
        if jx == 0: ax.set_ylabel(lab)
        if i < len(rows) - 1: ax.set_xticklabels([])
        if jx > 0: ax.set_yticklabels([])
fig.tight_layout(h_pad=0.3, w_pad=0.2)
save(fig, "fig_fields")

# ------------------------------------------------------------------ Fig 6: generated samples
cols = [("truth", "task + data"), ("U_lam0.001", "least squares"), ("C_ref", "conformal, fixed $G$"),
        ("C_on", "conformal, on-policy"), ("S_gam10", "shrinkage, $\\gamma=10$"), ("oracle", "ample-data fit")]
fig, axs = plt.subplots(2, len(cols), figsize=(6.5, 2.75))
for i, nb in enumerate([0, 3]):
    data = E[f"data_{nb}"]; bad = np.linalg.norm(data, axis=1) <= 1.0
    for jx, (k, lab) in enumerate(cols):
        ax = axs[i, jx]
        if k == "truth":
            x = E["true_samples"]; ax.scatter(x[:, 0], x[:, 1], s=0.6, color=GREY_LIGHT, alpha=0.6, linewidths=0, rasterized=True)
            ax.scatter(data[~bad, 0], data[~bad, 1], s=6, color=INK, linewidths=0, zorder=3)
            ax.scatter(data[bad, 0], data[bad, 1], s=12, color=CRIT, marker="x", linewidths=0.9, zorder=3)
            sub = "12 clean" + (" + 3 unsafe" if nb else "")
        else:
            x = E[f"samples_{k}_{nb}"]; ins = np.linalg.norm(x, axis=1) <= r_A
            col = {"C_ref": BLUE, "C_on": ORANGE}.get(k, INK2)
            ax.scatter(x[~ins, 0], x[~ins, 1], s=0.6, color=col, alpha=0.45, linewidths=0, rasterized=True)
            ax.scatter(x[ins, 0], x[ins, 1], s=0.8, color=CRIT, alpha=0.9, linewidths=0, rasterized=True)
            sub = f"$P(A)={float(E[f'pA_{k}_{nb}']):.3f}$"
        unsafe_disk(ax); square(ax, 3.1); ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.02, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color=INK2)
        if i == 0: ax.set_title(lab, fontsize=7)
        if jx == 0: ax.set_ylabel("clean data" if nb == 0 else "contaminated", fontsize=7)
        for s in ax.spines.values(): s.set_visible(True); s.set_color(AXIS); s.set_linewidth(0.5)
fig.tight_layout(h_pad=0.3, w_pad=0.25)
save(fig, "fig_samples")

# ------------------------------------------------------------------ Fig 7: ODE trajectories
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.35))
tcmap = LinearSegmentedColormap.from_list("t", ["#cde2fb", "#3987e5", "#0d366b"])
for ax, (k, lab) in zip(axs, [("U_lam0.001", "least squares"), ("C_ref", "conformal, fixed $G$"), ("C_on", "conformal, on-policy")]):
    tr_ = E[f"traj_{k}_3"]                                   # (K+1, n, 2)
    tt = np.linspace(0, 1, tr_.shape[0])
    for n in range(tr_.shape[1]):
        pts = tr_[:, n]
        for s in range(0, len(tt) - 1, 2):
            ax.plot(pts[s:s + 3, 0], pts[s:s + 3, 1], color=tcmap(tt[s]), linewidth=0.7, solid_capstyle="round")
    end = tr_[-1]; ins = np.linalg.norm(end, axis=1) <= r_A
    ax.scatter(end[~ins, 0], end[~ins, 1], s=6, color=INK, linewidths=0, zorder=3)
    ax.scatter(end[ins, 0], end[ins, 1], s=10, color=CRIT, linewidths=0, zorder=3)
    unsafe_disk(ax); square(ax, 3.1)
    ax.set_title(f"{lab}: {ins.sum()}/{len(ins)} paths end in $A$")
cax = fig.add_axes([0.3, -0.03, 0.4, 0.025])
cb = mpl.colorbar.ColorbarBase(cax, cmap=tcmap, norm=mpl.colors.Normalize(0, 1), orientation="horizontal")
cb.set_label("flow time $t$ (noise at $t=0$, data at $t=1$)", fontsize=7); cb.outline.set_linewidth(0.4)
fig.tight_layout(w_pad=0.6)
save(fig, "fig_trajectories")

# ------------------------------------------------------------------ Fig 8: few-shot summary over 32 tasks
rows_ft = [r for f in sorted(glob.glob(os.path.join(RES, "part_c_ft_*.json"))) for r in json.load(open(f))]
meths = [("U_lam0.001", "least\nsquares", INK2), ("C_ref", "conformal\nfixed $G$", BLUE), ("C_on", "conformal\non-policy", ORANGE),
         ("S_gam10", "shrinkage\n$\\gamma=10$", MUTED), ("centre", "centre\n$\\theta_c$", MUTED), ("oracle", "ample-data\nfit", MUTED)]
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.5))
for ax, nb, field, title, scale in [(axs[0], 0, "mmd", "(a) MMD$^2$, clean data", 1e3), (axs[1], 3, "mmd", "(b) MMD$^2$, contaminated", 1e3),
                                    (axs[2], 3, "pA", "(c) $P(A)$, contaminated", 1.0)]:
    vals = [np.array([r["res"][k][field] for r in rows_ft if r["n_bad"] == nb]) * scale for k, _, _ in meths]
    bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, linewidth=1.0), whiskerprops=dict(color=INK2, linewidth=0.7),
                    capprops=dict(color=INK2, linewidth=0.7), flierprops=dict(marker="o", markersize=2, markerfacecolor=INK2, markeredgewidth=0))
    for patch, (_, _, col) in zip(bp["boxes"], meths):
        patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col); patch.set_linewidth(0.8)
    ax.set_yscale("log"); ax.set_xticks(range(1, len(meths) + 1)); ax.set_xticklabels([m[1].replace("\n", " ") for m in meths], fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_title(title); ax.grid(True, axis="y", which="major")
    ax.tick_params(axis="x", length=0)
axs[0].set_ylabel("MMD$^2$ to the clean task ($\\times 10^{-3}$)")
hist_pA = float(np.mean(C["pA_all"]))
axs[2].axhline(hist_pA, color=CRIT, linewidth=0.8)
axs[2].annotate("historical safe fits (mean)", xy=(0.6, hist_pA), xytext=(0, -9), textcoords="offset points", fontsize=6.3, color=INK2, ha="left")
axs[2].set_ylabel("$P(A)$ of the fitted sampler")
fig.tight_layout(w_pad=0.6)
save(fig, "fig_fewshot")

# ------------------------------------------------------------------ Fig 9: score distributions (pilot A, one split)
cache = np.load(os.path.join(RES, "part_a_cache.npz"))
G_LL, G_BL, G_LB, B_base, B_mass = cache["G_LL"], cache["G_BL"], cache["G_LB"], cache["B_base"], cache["B_mass"]
P = G_LL.shape[0]; M, Kc = 20, 40; norm = 1 / (M * (M + 1)); q = int(np.ceil((Kc + 1) * 0.9))
perm = np.random.default_rng(7).permutation(P); tr, ca, te = perm[:M], perm[M:M + Kc], perm[M + Kc:]
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.3))
for ax, name, legit, badf in [
        (axs[0], "(a) forward cost $\\sqrt{2\\,\\mathrm{KL}(P^{(i)}\\|P^{v})}$",
         lambda A_: np.sqrt(G_LL[np.ix_(tr, A_)]).sum(0) * norm, lambda idx: np.sqrt(G_LB[np.ix_(tr, idx)]).sum(0) * norm),
        (axs[1], "(b) reverse cost $\\mathrm{KL}(P^{v}\\|P^{(i)})$",
         lambda A_: (G_LL[np.ix_(A_, tr)] / 2).sum(1) * norm, lambda idx: (G_BL[np.ix_(idx, tr)] / 2).sum(1) * norm)]:
    eps = np.sort(legit(ca))[q - 1]
    s_te = legit(te) / eps
    sel = np.where(np.isin(B_base, te) & (B_mass == 0.03))[0]
    s_b = badf(sel) / eps
    bins = np.linspace(0, 1.05 * np.max(np.r_[s_te, s_b]), 40)
    ax.hist(s_te, bins=bins, color=INK2, alpha=0.45, label="held-out safe fields")
    ax.hist(s_b, bins=bins, color=ORANGE if "reverse" in name else BLUE, alpha=0.55, label="same fields, 3% unsafe mass")
    ax.axvline(1.0, color=INK, linewidth=0.9)
    ax.annotate("$\\varepsilon$", xy=(1.0, ax.get_ylim()[1] * 0.92), xytext=(3, 0), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("order-1 score $S(v;\\mathcal{D}) / \\varepsilon$"); ax.set_title(name)
    ax.legend(loc="upper right", fontsize=6.5)
axs[0].set_ylabel("number of fields")
fig.tight_layout(w_pad=1.0)
save(fig, "fig_scores")
