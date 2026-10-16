"""Exclusion figure of the E1 test (PDF for LaTeX + PNG preview), from ../results/e1/; the other E1 figures are made by
e1_walkthrough.py.  Style and colour roles as in
make_figures.py: categorical slots in the documented order (blue, orange, aqua, yellow), baselines grey, unsafe set
in the status red, always labelled.
Usage: python e1_figures.py"""
import os, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "..", "results", "e1"), os.path.join(HERE, "..", "figures")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, CRIT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#d03b3b"
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
MASSES = [0.01, 0.03, 0.1, 0.3]
r_A = 0.6


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig); print("wrote", name)


def fig_detection():
    """Exclusion of unsafe prompts against their unsafe mass: the method as written, its localisations, the prompt baseline."""
    scr = json.load(open(os.path.join(RES, "improve_screen.json")))
    ph1 = json.load(open(os.path.join(RES, "eval_splits.json")))

    def curve(rows):
        return [np.mean([r[f"excl_{m}"] for r in rows]) for m in MASSES], 1 - np.mean([r["coverage"] for r in rows])
    series = [  # (label, colour, selector)
        ("uniform, $L^2$ map metric (as written)", BLUE, lambda c: [r for r in scr if r["cfg"] == c and r["p"] == 2 and r["kind"] == "U"]),
        ("nearest safe field, $L^2$", ORANGE, lambda c: [r for r in scr if r["cfg"] == c and r["p"] == 2 and r["kind"] == "kNN" and r["par"] == 1]),
        ("nearest safe field, truncated $L^1$", AQUA, lambda c: [r for r in scr if r["cfg"] == c and r["p"] == "trunc1.0" and r["kind"] == "kNN" and r["par"] == 1]),
        ("nearest safe field, tail score", YELLOW, lambda c: [r for r in scr if r["cfg"] == c and r["p"] == "tail0.5" and r["kind"] == "kNN" and r["par"] == 1]),
        ("uniform, prompt embeddings (baseline)", MUTED, lambda c: [r for r in ph1 if r["cfg"] == c and r["metric"] == "E"]),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True)
    for ax, cfg, title in [(axs[0], "rect1", "(a) 1-rectified model, 50 Euler steps"), (axs[1], "rect2", "(b) 2-rectified (reflowed) model")]:
        fa = []
        for lab, col, sel in series:
            y, f = curve(sel(cfg)); fa.append(f)
            ax.plot(MASSES, y, color=col, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8,
                    label=lab, solid_capstyle="round", zorder=3)
        ax.axhline(np.mean(fa), color=GRID, linewidth=1.0, zorder=1)
        ax.text(0.11, np.mean(fa) - 0.06, "safe prompts excluded", fontsize=6.5, color=MUTED)
        ax.set_xscale("log"); ax.set_xticks(MASSES); ax.set_xticklabels([f"{m:g}" for m in MASSES])
        ax.set_ylim(0, 1.04); ax.set_title(title, loc="left"); ax.grid(True, axis="y")
        ax.set_xlabel("unsafe mass $m$ of the prompt")
    axs[0].set_ylabel("unsafe prompts outside the region")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_e1_detection")


if __name__ == "__main__":
    fig_detection()
