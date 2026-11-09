"""Figures, tables and numbers of Section 12 of safe_prompt_fields.tex (a safe and an unsafe set), from ../results/e3/
regions_unsafe.json (e3u_regions.py), enforce_unsafe.json (e3u_enforce.py) and segment_unsafe.json (e3u_segment.py), with the
records of Sections 11.6 and 11.8 (enforce.json, enforce_combined.json) as references.
  fig_e3u_cells        per kind of request: the fractions in each of the four cells (safe set only, unsafe set only, both, neither)
  fig_e3u_detection    unsafe requests excluded against safe unfamiliar requests excluded: the safe set alone, then both sets
  fig_e3u_sizes        against the number of anchors of each class: what each set contains of the other class, and two exclusions
  fig_e3u_segment      safe and unsafe scores along the repair segment of a few requests
  fig_e3u_enforce      per method: deployed images judged unsafe and their scene similarity
  fig_e3u_depth        per repaired request: mixing weight and P(winter), previous stop against the two-set stop
  fig_e3u_gallery_*    what the methods deploy for the requests of fig_e3c_gallery_*
Tables: ../tables/e3u_sets.tex, e3u_detection.tex, e3u_enforce.tex, e3u_unfamiliar.tex (row bodies), e3u_numbers.tex (macros).
Usage: python e3u_figures.py [--no-gallery]
"""
import os, sys, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch
import e1_figures as F
import e3_common as C
import e3_figures as G
import e3_figures_combined as H

TAB = G.TAB
RU = json.load(open(os.path.join(C.RES, "regions_unsafe.json")))
ROWS = G.ROWS + json.load(open(os.path.join(C.RES, "prompts_unsafe.json")))["prompts"]
POOL = G.POOL
SPACES = ["E", "PB", "L2"]
SLAB = {"E": "prompt", "PB": "source slice", "L2": "field"}
SMAC = {"E": "P", "PB": "S", "L2": "F"}
TMAC = {"static": "S", "adaptive": "A"}
RCOL = {"E": F.ORANGE, "PB": F.AQUA, "L2": F.BLUE}
GROUPS = [("test_safe", "approved (test)", "App"), ("test_unsafe", "unsafe history (test)", "Hist"), ("summer", "summer", "Sum"),
          ("neutral", "season not mentioned", "Neu"), ("novel_summer", "novel summer", "NovS"), ("disguised", "disguised", "Dis"),
          ("paraphrase", "paraphrase", "Par"), ("novel_winter", "novel winter", "NovW")]
GLAB = {g: l for g, l, _ in GROUPS}
GMAC = {g: m for g, _, m in GROUPS}
CELLS = [("c_safe", "safe set only (admitted)", F.BLUE), ("c_ambig", "both sets (ambiguous)", F.VIOLET),
         ("c_unsafe", "unsafe set only", F.CRIT), ("c_novel", "neither set (novel)", F.AXIS)]
N_UNF = {"paraphrase": 16, "novel_winter": 12}
ENF_W = {"E": "XN:1", "PB": "U", "L2": "XN:1"}       # weights of the sets used for enforcement (M = 140)
NUM = {}


def mac(k, v):
    NUM[k] = v


f2 = G.f2


def ev(sp, tag):
    return RU["eval"][f"{sp}|{tag}"]


def false_alarms(e, rule):
    """Fraction of the safe unfamiliar requests (paraphrases and novel winter scenes) that a rule excludes."""
    key = {"safe": "adm_safe", "unsafe": "adm_unsafe", "both": "adm_both", "cap": "adm_both_cap"}[rule]
    return sum(N_UNF[g] * (1 - e[g][key]) for g in N_UNF) / sum(N_UNF.values())


def excl(e, g, rule):
    key = {"safe": "adm_safe", "unsafe": "adm_unsafe", "both": "adm_both", "cap": "adm_both_cap"}[rule]
    return 1 - e[g][key]


# ------------------------------------------------------------------ the sets and detection
def sets_table():
    """One row per space and weighting: the tuned weights, each set's coverage of its own class and the fraction of the other class
    that it contains (its size where it matters), and the fraction of approved prompts that the two-set rule admits."""
    lines = []
    for sp in SPACES:
        for tag in ["static", "adaptive"]:
            e = ev(sp, tag)
            w = e["weights"]
            wl = "uniform" if tag == "static" else f"{w['safe']} / {w['unsafe']}"
            cells = [SLAB[sp] if tag == "static" else "", tag, wl, f"{e['test_safe']['in_safe']:.3f}", f2(e["test_unsafe"]["in_safe"]),
                     f"{e['test_unsafe']['in_unsafe']:.3f}", f2(e["test_safe"]["in_unsafe"]), f"{e['test_safe']['adm_both']:.3f}",
                     f2(e["cap_binds"])]
            lines.append(" & ".join(cells) + " \\\\")
        if sp != SPACES[-1]:
            lines.append("\\midrule")
    open(os.path.join(TAB, "e3u_sets.tex"), "w").write("\n".join(lines) + "\n")


def detection_table():
    """One row per space and weighting; each cell: the fraction excluded by the safe set alone -> by the two-set rule. First block: the
    protocol of Section 11 (M = 30 anchors per class, weights tuned on the tuning pools); second block: M = 140 anchors per class,
    the weights used for enforcement (anchor sweep, 200 splits of the pooled prompts)."""
    lines = []
    def row(e, lab, tag):
        cells = [lab, tag, f"{e['test_safe']['adm_safe']:.2f}$\\to${e['test_safe']['adm_both']:.2f}"]
        for g in ["summer", "neutral", "novel_summer", "paraphrase", "novel_winter", "disguised"]:
            cells.append(f"{f2(excl(e, g, 'safe'))}$\\to${f2(excl(e, g, 'both'))}")
        return " & ".join(cells) + " \\\\"
    lines.append("\\multicolumn{9}{l}{\\emph{$M=30$ anchors of each class, weights tuned on the tuning pools}}\\\\")
    for sp in SPACES:
        for tag in ["static", "adaptive"]:
            lines.append(row(ev(sp, tag), SLAB[sp] if tag == "static" else "", tag + ("" if tag == "static" else
                                                                                       f" ({ev(sp, tag)['weights']['safe']}, {ev(sp, tag)['weights']['unsafe']})")))
    lines.append("\\midrule")
    lines.append("\\multicolumn{9}{l}{\\emph{$M=140$ anchors of each class, weights of the repairs (Section~\\ref{sec:e3u_repair})}}\\\\")
    for sp in SPACES:
        w = ENF_W[sp]
        lines.append(row(RU["anchors"][f"{sp}|{w}|140"], SLAB[sp], "static" if w == "U" else "nearest prompt"))
    open(os.path.join(TAB, "e3u_detection.tex"), "w").write("\n".join(lines) + "\n")


def detection_macros():
    mac("ethuNhistTune", len([r for r in ROWS if r["kind"] == "summer_hist" and r["pool"] == "tune"]))
    mac("ethuNhistEval", len([r for r in ROWS if r["kind"] == "summer_hist" and r["pool"] == "eval"]))
    mac("ethuNunsTune", len([r for r in ROWS if r["kind"] == "summer_hist" and r["pool"] == "tune"]) + 16)
    mac("ethuHistUnsafe", f2(RU["history_unsafe_probe"])); mac("ethuHistTuneUnsafe", f2(RU["history_unsafe_tune_probe"]))
    mac("ethuKcap", RU["k_cap"]); mac("ethuCapLevel", f"{RU['k_cap'] / 41:.3f}")
    for sp in SPACES:
        for tag in ["static", "adaptive"]:
            e, t = ev(sp, tag), SMAC[sp] + TMAC[tag]
            mac(f"ethuW{t}", e["weights"]["safe"]); mac(f"ethuWu{t}", e["weights"]["unsafe"])
            mac(f"ethuCovS{t}", f"{e['test_safe']['in_safe']:.3f}"); mac(f"ethuCovU{t}", f"{e['test_unsafe']['in_unsafe']:.3f}")
            mac(f"ethuSizeS{t}", f2(e["test_unsafe"]["in_safe"])); mac(f"ethuSizeU{t}", f2(e["test_safe"]["in_unsafe"]))
            mac(f"ethuAdm{t}", f"{e['test_safe']['adm_both']:.3f}"); mac(f"ethuAdmCap{t}", f"{e['test_safe']['adm_both_cap']:.3f}")
            mac(f"ethuCapBinds{t}", f2(e["cap_binds"])); mac(f"ethuCapClaim{t}", f"{e['test_safe']['in_unsafe_cap']:.3f}")
            for rule, rm in [("safe", "S"), ("unsafe", "U"), ("both", "B"), ("cap", "C")]:
                mac(f"ethuFA{t}{rm}", f2(false_alarms(e, rule)))
                for g in ["summer", "neutral", "novel_summer", "paraphrase", "novel_winter", "disguised", "judged_unsafe", "judged_safe",
                          "test_unsafe"]:
                    mac(f"ethuEx{GMAC.get(g, {'judged_unsafe': 'JU', 'judged_safe': 'JS'}.get(g))}{t}{rm}", f2(excl(e, g, rule)))
            for g, _, gm in GROUPS:
                for c, _, _ in CELLS:
                    mac(f"ethuCell{c[2:].capitalize()}{gm}{t}", f2(e[g][c]))
        e, t = enf_cells(sp), SMAC[sp] + "E"
        mac(f"ethuCovS{t}", f"{e['test_safe']['in_safe']:.3f}"); mac(f"ethuCovU{t}", f"{e['test_unsafe']['in_unsafe']:.3f}")
        mac(f"ethuSizeS{t}", f2(e["test_unsafe"]["in_safe"])); mac(f"ethuSizeU{t}", f2(e["test_safe"]["in_unsafe"]))
        mac(f"ethuAdm{t}", f"{e['test_safe']['adm_both']:.3f}")
        for rule, rm in [("safe", "S"), ("unsafe", "U"), ("both", "B")]:
            mac(f"ethuFA{t}{rm}", f2(false_alarms(e, rule)))
            for g in ["summer", "neutral", "novel_summer", "paraphrase", "novel_winter", "disguised", "test_unsafe"]:
                mac(f"ethuEx{GMAC[g]}{t}{rm}", f2(excl(e, g, rule)))
        for g, _, gm in GROUPS:
            for c, _, _ in CELLS:
                mac(f"ethuCell{c[2:].capitalize()}{gm}{t}", f2(e[g][c]))
        tu = RU["tuning"][sp]
        for cls in ["safe", "unsafe"]:
            for cand in tu[cls]["candidates"]:
                if cand["w"] == "U":
                    mac(f"ethuTune{SMAC[sp]}{cls.capitalize()}U", f2(cand["size"]))
                if cand["w"] == tu[cls]["adaptive"]:
                    mac(f"ethuTune{SMAC[sp]}{cls.capitalize()}A", f2(cand["size"]))


def enf_cells(sp):
    """Cells of the sets used for enforcement: M = 140 anchors of each class, weights ENF_W (anchor sweep, 200 pooled splits)."""
    return RU["anchors"][f"{sp}|{ENF_W[sp]}|140"]


def extra_detection_macros():
    """Coverage ranges, the sizes of the sweep at M = 30 and 140, and how often the historical prompt of each class nearest to a request
    in the prompt embedding shows the request's scene (requests of the 16 approved scenes; all 190 prompts of each class)."""
    import e3u_regions as U
    cs = [ev(sp, t)["test_safe"]["in_safe"] for sp in SPACES for t in ["static", "adaptive"]]
    cu = [ev(sp, t)["test_unsafe"]["in_unsafe"] for sp in SPACES for t in ["static", "adaptive"]]
    mac("ethuCovMin", f"{min(cs):.3f}"); mac("ethuCovMax", f"{max(cs):.3f}")
    mac("ethuCovUMin", f"{min(cu):.3f}"); mac("ethuCovUMax", f"{max(cu):.3f}")
    sw = [RU["anchors"][k]["test_safe"]["in_safe"] for k in RU["anchors"]] + [RU["anchors"][k]["test_unsafe"]["in_unsafe"] for k in RU["anchors"]]
    mac("ethuSwCovMin", f"{min(sw):.2f}"); mac("ethuSwCovMax", f"{max(sw):.2f}")
    for sp in SPACES:
        for wn, wm in [("U", "U"), ("XN:1", "N")]:
            for M, mm in [(30, "XXX"), (70, "LXX"), (140, "CXL")]:
                e = RU["anchors"][f"{sp}|{wn}|{M}"]
                mac(f"ethuSwSizeS{SMAC[sp]}{wm}{mm}", f2(e["test_unsafe"]["in_safe"]))
                mac(f"ethuSwSizeU{SMAC[sp]}{wm}{mm}", f2(e["test_safe"]["in_unsafe"]))
                mac(f"ethuSwAdm{SMAC[sp]}{wm}{mm}", f2(e["test_safe"]["adm_both"]))
                for g in ["summer", "neutral", "novel_winter", "paraphrase"]:
                    mac(f"ethuSwEx{GMAC[g]}{SMAC[sp]}{wm}{mm}", f2(1 - e[g]["adm_both"]))
    rows, D, Fe, idx = U.load_all()
    E = U.dist_matrix("E", Fe)
    app = np.r_[idx["approved_tune"], idx["approved_eval"]]; uns = np.r_[idx["unsafe_tune"], idx["unsafe_eval"]]
    sc = np.array([r["scene"] for r in rows])
    req = np.r_[idx["summer_eval"], idx["neutral_eval"], idx["disguised_eval"], idx["paraphrase_eval"]]
    for cls, pool, m in [("safe", app, "S"), ("unsafe", uns, "U")]:
        nn = [pool[np.argmin(E[j, pool[pool != j]])] if j in pool else pool[np.argmin(E[j, pool])] for j in req]
        mac(f"ethuNNsame{m}", f2(np.mean(sc[np.array(nn)] == sc[req])))
    mac("ethuNreqNN", len(req))
    import e3_regions as R
    _, _, t_s = R.split(app, 7000, m=140, k=R.K)
    mac("ethuNappReq", int(sum(rows[j]["pool"] == "eval" for j in t_s)))
    mac("ethuCapGuar", f"{(int(np.ceil(41 * 0.9)) - RU['k_cap']) / 41:.3f}")
    for g, gm in [("summer_eval", "Sum"), ("neutral_eval", "Neu")]:
        mac(f"ethuGreen{gm}", f2(np.nanmean(POOL["green_eval"][idx[g]])))


def fig_cells():
    """For each space (column) and kind of request (bar): the fraction of requests in each cell, averaged over 200 splits. Top row: the
    protocol of Section 11 with static weights (M = 30 anchors of each class); bottom row: the sets used for enforcement (M = 140,
    static weights for the source slice, the nearest prompt of each class for the prompt and field spaces)."""
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.6), sharey=True, sharex=True)
    y = np.arange(len(GROUPS))[::-1]
    for r, (rowlab, get) in enumerate([("Static weights, $M=30$ anchors of each class (protocol of Section 11)", lambda sp: ev(sp, "static")),
                                        ("Weights of the repairs, $M=140$ anchors of each class", enf_cells)]):
        for a, sp in enumerate(SPACES):
            ax, e = axs[r, a], get(sp)
            for (g, lab, _), yy in zip(GROUPS, y):
                left = 0.0
                for c, _, col in CELLS:
                    v = e[g][c]
                    ax.barh(yy, v, left=left, color=col, height=0.72, edgecolor="white", linewidth=0.5)
                    if v >= 0.12:
                        ax.text(left + v / 2, yy, f"{v:.2f}", ha="center", va="center", fontsize=4.8,
                                color="white" if c != "c_novel" else F.INK)
                    left += v
            ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.tick_params(axis="x", labelsize=5.6)
            ax.axhline(y[1] - 0.5, color=F.INK2, lw=0.6)
            wl = "" if r == 0 else (", static" if ENF_W[sp] == "U" else ", nearest prompt")
            ax.set_title(f"({'abcdef'[3 * r + a]}) {SLAB[sp]}{wl}", fontsize=6.6, loc="left")
            if r == 1:
                ax.set_xlabel("fraction of the requests", fontsize=6.4)
        axs[r, 0].set_yticks(y); axs[r, 0].set_yticklabels([lab for _, lab, _ in GROUPS], fontsize=6.0)
        axs[r, 0].text(-0.62, 1.16, rowlab, transform=axs[r, 0].transAxes, fontsize=6.8, fontweight="bold", color=F.INK, ha="left")
    hs = [Patch(facecolor=col, edgecolor="none", label=lab) for _, lab, col in CELLS]
    fig.legend(handles=hs, loc="upper center", ncol=4, fontsize=5.9, frameon=False, bbox_to_anchor=(0.55, 1.0), handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.6, h_pad=2.2)
    F.save(fig, "fig_e3u_cells")


def fig_detection_map():
    """Unsafe requests excluded against safe unfamiliar requests excluded: an arrow from the safe set alone to the two-set rule, for
    each space and weighting."""
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.85), sharey=True)
    for a, (ax, g, title) in enumerate(zip(axs, ["summer", "neutral"], ["summer requests", "requests that do not mention the season"])):
        for sp in SPACES:
            for tag in ["static", "adaptive"]:
                e = ev(sp, tag)
                x0, y0 = false_alarms(e, "safe"), excl(e, g, "safe")
                x1, y1 = false_alarms(e, "both"), excl(e, g, "both")
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=RCOL[sp], lw=0.9,
                            shrinkA=3, shrinkB=3, mutation_scale=7, ls="-" if tag == "adaptive" else (0, (2, 1.5))), zorder=2)
                G.mark(ax, x0, y0, "PB" if sp == "PB" else ("E" if sp == "E" else "L2"), tag)
                ax.scatter([x1], [y1], marker="D", s=20 if tag == "adaptive" else 26, facecolors=RCOL[sp] if tag == "adaptive" else "white",
                           edgecolors=RCOL[sp], linewidths=1.0, zorder=5)
        ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.07); ax.grid(True)
        ax.set_xlabel("safe unfamiliar requests excluded (false alarms)", fontsize=6.6)
        ax.set_title(f"({'ab'[a]}) {title}", fontsize=7, loc="left")
    axs[0].set_ylabel("unsafe requests excluded", fontsize=6.6)
    hs = [Line2D([], [], marker="s", ls="", ms=5.5, mfc=RCOL[sp], mec=RCOL[sp], label=SLAB[sp]) for sp in SPACES]
    hs += [Line2D([], [], marker="o", ls="", ms=5.5, mfc="white", mec=F.INK2, mew=1.1, label="safe set alone"),
           Line2D([], [], marker="D", ls="", ms=4.5, mfc=F.INK2, mec=F.INK2, label="safe and unsafe sets"),
           Line2D([], [], color=F.INK2, lw=0.9, ls=(0, (2, 1.5)), label="static weights"),
           Line2D([], [], color=F.INK2, lw=0.9, label="prompt-adaptive weights")]
    fig.legend(handles=hs, loc="upper center", ncol=7, fontsize=5.7, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2,
               columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.2)
    F.save(fig, "fig_e3u_detection")


def fig_sizes():
    """Against the number M of anchors of each class (all 190 prompts of each class pooled, 200 splits, weights fixed in advance): (a)
    the fraction of unsafe test prompts in the safe set, (b) the fraction of approved test prompts in the unsafe set, (c) the
    fraction of the requests that do not mention the season that the two-set rule excludes, (d) the fraction of the safe winter
    requests of new scenes that it excludes (false alarms). Colour: the space; dashed, hollow: static weights; solid, filled:
    nearest prompt (XN:1)."""
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 2.15), sharey=True)
    Ms = RU["M_grid"]
    panels = [("test_unsafe", "in_safe", "safe set: unsafe prompts in it"), ("test_safe", "in_unsafe", "unsafe set: approved in it"),
              ("neutral", "x_both", "no season: excluded"), ("novel_winter", "x_both", "novel winter: excluded")]
    for a, (ax, (g, key, title)) in enumerate(zip(axs, panels)):
        for sp in SPACES:
            for wn, ls, filled in [("U", (0, (3, 2)), False), ("XN:1", "-", True)]:
                e = [RU["anchors"][f"{sp}|{wn}|{M}"][g] for M in Ms]
                yv = [1 - x["adm_both"] if key == "x_both" else x[key] for x in e]
                ax.plot(Ms, yv, color=RCOL[sp], ls=ls, lw=1.2, marker="o", ms=3.4, mfc=RCOL[sp] if filled else "white",
                        mec=RCOL[sp], mew=0.9, zorder=3)
        ax.set_xticks(Ms); ax.tick_params(labelsize=5.4); ax.set_xlabel("anchors per class $M$", fontsize=6.2)
        ax.set_ylim(-0.04, 1.04); ax.grid(True)
        ax.set_title(f"({'abcd'[a]}) {title}", fontsize=6.2, loc="left")
    axs[0].set_ylabel("fraction", fontsize=6.4)
    hs = [Line2D([], [], color=RCOL[sp], lw=1.2, label=SLAB[sp]) for sp in SPACES]
    hs += [Line2D([], [], color=F.INK2, lw=1.2, ls=(0, (3, 2)), marker="o", ms=3.4, mfc="white", label="static weights"),
           Line2D([], [], color=F.INK2, lw=1.2, marker="o", ms=3.4, label="nearest prompt (XN:1)")]
    fig.legend(handles=hs, loc="upper center", ncol=5, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.5)
    F.save(fig, "fig_e3u_sizes")


# ------------------------------------------------------------------ repairs
EU = json.load(open(os.path.join(C.RES, "enforce_unsafe.json"))) if os.path.exists(os.path.join(C.RES, "enforce_unsafe.json")) else None
LATU = np.load(os.path.join(C.RES, "enforce_unsafe_latents.npz")) if os.path.exists(os.path.join(C.RES, "enforce_unsafe_latents.npz")) else None
SEG = json.load(open(os.path.join(C.RES, "segment_unsafe.json"))) if os.path.exists(os.path.join(C.RES, "segment_unsafe.json")) else None
REFS = ["S>F", "S>F:local"]
NEWM = ["S2", "P2", "F1", "F2", "S>F2"]
LAB = {"S>F": "slice$\\to$field", "S>F:local": "slice$\\to$field, local", "S2": "slice, two sets", "P2": "prompt, two sets",
       "F1": "field, safe set", "F2": "field, two sets", "S>F2": "slice$\\to$field, two sets"}
CODE = {"S>F": "SF", "S>F:local": "SFl", "S2": "Stwo", "P2": "Ptwo", "F1": "Fone", "F2": "Ftwo", "S>F2": "SFtwo"}
DSP = {"S>F": "PB", "S>F:local": "PB", "S2": "PB", "S>F2": "PB", "P2": "E", "F1": "L2", "F2": "L2"}     # the space that decides
KINDS = ["summer", "neutral", "disguised", "approved"]
UNF = ["paraphrase", "novel_winter", "novel_summer"]
KM = {"summer": "Sum", "neutral": "Neu", "disguised": "Dis", "approved": "App", "paraphrase": "Par", "novel_winter": "NovW",
      "novel_summer": "NovS"}


def rows_of(m, kind):
    return [r for r in EU["rows"] if r["method"] == m and r["kind"] == kind]


def has(m, kinds):
    return any(rows_of(m, k) for k in kinds)


def unsafe(r):
    return float(np.mean(np.array(r["pwB"]) < 0.5))


def st(m, kind):
    rr = rows_of(m, kind)
    if not rr:
        return None
    alt = [r for r in rr if r["kappa"] > 0 or not r["admitted"]]
    return dict(n=len(rr), alt=np.mean([not r["admitted"] for r in rr]), unsafe=np.mean([unsafe(r) for r in rr]),
                pw=np.mean([np.mean(r["pwB"]) for r in rr]), subj=np.mean([np.mean(r["subj"]) for r in rr]),
                kappa=np.mean([r["kappa"] for r in alt]) if alt else np.nan,
                same=np.mean([r["same_scene"] for r in alt]) if alt else np.nan,
                change=np.mean([r["chi_change"] for r in rr]))


def own(kind, js=None):
    js = js if js is not None else sorted({r["j"] for r in EU["rows"] if r["kind"] == kind})
    return dict(unsafe=np.mean(POOL["pwB_eval"][js] < 0.5), subj=np.mean(POOL["subj_eval"][js]), pw=np.mean(POOL["pwB_eval"][js]))


def enforce_table():
    lines = []
    o = {k: own(k) for k in KINDS}
    subj0 = np.mean([POOL["subj_eval"][j].mean() for j in EU["requests"] if ROWS[j]["kind"] != "approved"])
    lines.append(f"no region & 0 & -- & {f2(o['summer']['unsafe'])} & 0 & -- & {f2(o['neutral']['unsafe'])} & "
                 f"{f2(o['disguised']['unsafe'])} & {subj0:.3f} & -- & 0 \\\\")
    lines.append("\\midrule")
    for block in [REFS, NEWM]:
        for m in block:
            if not has(m, ["summer"]):
                continue
            s_ = {k: st(m, k) for k in KINDS}
            cells = []
            for k in ["summer", "neutral"]:
                if s_[k] is None:
                    cells += ["--"] * 3; continue
                cells += [f2(s_[k]["alt"]), "--" if np.isnan(s_[k]["kappa"]) else f2(s_[k]["kappa"]), f2(s_[k]["unsafe"])]
            rr = [r for k in ["summer", "neutral", "disguised"] for r in rows_of(m, k)]
            alt = [r for r in rr if not r["admitted"]]
            cells += ["--" if s_["disguised"] is None else f2(s_["disguised"]["unsafe"]), f"{np.mean([np.mean(r['subj']) for r in rr]):.3f}",
                      f2(np.mean([r["same_scene"] for r in alt])) if alt else "--", "--" if s_["approved"] is None else f2(s_["approved"]["alt"])]
            lines.append(f"{LAB[m]} & " + " & ".join(cells) + " \\\\")
        if block is REFS:
            lines.append("\\midrule")
    open(os.path.join(TAB, "e3u_enforce.tex"), "w").write("\n".join(lines) + "\n")


def unfamiliar_table():
    lines = []
    for block in [REFS, NEWM]:
        for m in block:
            if not all(rows_of(m, k) for k in UNF):
                continue
            s_ = {k: st(m, k) for k in UNF}
            cells = []
            for k in ["paraphrase", "novel_winter"]:
                cells += [f2(s_[k]["alt"]), f2(s_[k]["change"]), f2(s_[k]["unsafe"])]
            k = "novel_summer"
            cells += [f2(s_[k]["alt"]), f2(s_[k]["unsafe"]), f"{s_[k]['subj']:.3f}"]
            lines.append(f"{LAB[m]} & " + " & ".join(cells) + " \\\\")
        if block is REFS:
            lines.append("\\midrule")
    open(os.path.join(TAB, "e3u_unfamiliar.tex"), "w").write("\n".join(lines) + "\n")


def enforce_macros():
    for k in KINDS + UNF:
        o = own(k)
        mac(f"ethuOwnUnsafe{KM[k]}", f2(o["unsafe"])); mac(f"ethuOwnSubj{KM[k]}", f"{o['subj']:.3f}")
    for m in REFS + NEWM:
        for k in KINDS + UNF:
            s_ = st(m, k)
            if s_ is None:
                continue
            t = CODE[m] + KM[k]
            mac(f"ethuAlt{t}", f2(s_["alt"])); mac(f"ethuAfter{t}", f2(s_["unsafe"])); mac(f"ethuSubj{t}", f"{s_['subj']:.3f}")
            mac(f"ethuKap{t}", "--" if np.isnan(s_["kappa"]) else f2(s_["kappa"])); mac(f"ethuChg{t}", f2(s_["change"]))
            mac(f"ethuSame{t}", "--" if np.isnan(s_["same"]) else f2(s_["same"]))
        rr = [r for k in ["summer", "neutral", "disguised"] for r in rows_of(m, k)]
        alt = [r for r in rr if not r["admitted"]]
        if alt:
            c = CODE[m]
            mac(f"ethuSame{c}", f2(np.mean([r["same_scene"] for r in alt])))
            mac(f"ethuSubj{c}", f"{np.mean([np.mean(r['subj']) for r in rr]):.3f}")
            mac(f"ethuKap{c}", f2(np.mean([r["kappa"] for r in alt])))
            mac(f"ethuFallback{c}", sum(r["fallback"] > 0 for r in EU["rows"] if r["method"] == m))
            mac(f"ethuNrep{c}", sum(not r["admitted"] for r in EU["rows"] if r["method"] == m))
            # where the deployed fields of the repaired summer and season-not-mentioned requests lie
            rep = [r for k in ["summer", "neutral"] for r in rows_of(m, k) if not r["admitted"]]
            if rep:
                for key, nm in [("PB|unsafe|U", "OutUS"), ("L2|unsafe|XN:1", "OutUF")]:
                    mac(f"ethuPlace{nm}{c}", f2(np.mean([r["place"][key] > 1 for r in rep])))
                for key, nm in [("PB|safe|U", "InSS"), ("L2|safe|XN:1", "InSF")]:
                    mac(f"ethuPlace{nm}{c}", f2(np.mean([r["place"][key] <= 1 for r in rep])))
                for nm, sel in [("Small", [r for r in rep if r["kappa"] < 0.5]), ("Large", [r for r in rep if r["kappa"] >= 0.5])]:
                    mac(f"ethuN{nm}{c}", len(sel))
                    mac(f"ethuRes{nm}{c}", f2(np.mean([unsafe(r) for r in sel])) if sel else "--")
    mac("ethuNunf", len(EU["unfamiliar"]) // 3)


def paired(m0, m1, kinds=("summer", "neutral")):
    """Requests repaired by both methods: (kappa0, pw0, kappa1, pw1, kind)."""
    out = []
    for k in kinds:
        r0, r1 = {r["j"]: r for r in rows_of(m0, k)}, {r["j"]: r for r in rows_of(m1, k)}
        for j in sorted(set(r0) & set(r1)):
            if r0[j]["admitted"] and r1[j]["admitted"]:
                continue
            out.append((r0[j]["kappa"], np.mean(r0[j]["pwB"]), r1[j]["kappa"], np.mean(r1[j]["pwB"]), k, j))
    return out


def depth_macros():
    for m0, m1, nm in [("S>F", "S2", "SvsStwo"), ("F1", "F2", "FonevsFtwo"), ("S>F:local", "S>F2", "SFlvsSFtwo")]:
        pp = paired(m0, m1)
        if not pp:
            continue
        k0, k1 = np.array([p[0] for p in pp]), np.array([p[2] for p in pp])
        mac(f"ethuDeeper{nm}", f2(np.mean(k1 > k0 + 1e-9))); mac(f"ethuShallower{nm}", f2(np.mean(k1 < k0 - 1e-9)))
        mac(f"ethuDkap{nm}", f2(np.mean(k1 - k0))); mac(f"ethuNpair{nm}", len(pp))


def binding_macros():
    """Which stop condition sets the depth of a repair: the conditions that fail at the largest mixing weight that the bisection found
    inadmissible (none if every tested weight was admissible). Over the repaired summer and season-not-mentioned requests."""
    names = {"PB|safe|U": "SliceSafe", "PB|unsafe|U": "SliceUnsafe", "L2|safe|XN:1": "FieldSafe", "L2|unsafe|XN:1": "FieldUnsafe",
             "E|safe|XN:1": "PromptSafe", "E|unsafe|XN:1": "PromptUnsafe", "L2|local|-": "Local"}
    for m in REFS + NEWM:
        rep = [r for k in ["summer", "neutral"] for r in rows_of(m, k) if not r["admitted"] and r["fallback"] == 0]
        if not rep:
            continue
        cnt = {}
        for r in rep:
            bad = [t for t in r["trace"] if not all((v > 1) if "unsafe" in key else (v <= 1) for key, v in t.items() if key != "kappa")]
            if not bad:
                continue
            last = max(bad, key=lambda t: t["kappa"])
            for key, v in last.items():
                if key != "kappa" and ((v <= 1) if "unsafe" in key else (v > 1)):
                    cnt[key] = cnt.get(key, 0) + 1
        for key, nm in names.items():
            if any(key in t for r in rep for t in r["trace"]):
                mac(f"ethuBind{nm}{CODE[m]}", cnt.get(key, 0))
        mac(f"ethuNbind{CODE[m]}", len(rep))
        # generalisation: repaired requests whose deployed field meets the stop conditions on the probes but whose images on the
        # evaluation noise are mostly unsafe
        mac(f"ethuGap{CODE[m]}", sum(r["meets_stop"] and unsafe(r) >= 0.5 for r in rep))


def request_macros():
    """The two pine forests (the example of Section 11) and the season-not-mentioned requests that a method leaves mostly unsafe."""
    pick = {"Sum": G.ids_of("summer", among=EU["requests"])[0], "Neu": G.ids_of("neutral", among=EU["requests"])[0]}
    for km, j in pick.items():
        for m in REFS + NEWM:
            r = rec_of(m, j)
            if r is not None:
                mac(f"ethuPine{km}Kap{CODE[m]}", f2(r["kappa"])); mac(f"ethuPine{km}Pw{CODE[m]}", f2(np.mean(r["pwB"])))
        mac(f"ethuPine{km}PwOwn", f2(np.mean(POOL["pwB_eval"][j])))
    for m in REFS + NEWM:
        bad = [r for r in rows_of(m, "neutral") if unsafe(r) >= 0.5]
        mac(f"ethuNbadNeu{CODE[m]}", len(bad))
        names = [ROWS[r["j"]]["prompt"][len("a photograph of "):] for r in sorted(bad, key=lambda r: r["j"])]
        mac(f"ethuBadNeu{CODE[m]}", "; ".join(f"``{n}''" for n in names) if names else "none")
        mac(f"ethuBadOutSF{CODE[m]}", sum(r["place"]["L2|safe|XN:1"] > 1 for r in bad))
    # the field's sets alone: do the repairs that stay unsafe violate a condition of the source slice's two sets?
    for m in ["F1", "F2"]:
        rep = [r for k in ["summer", "neutral"] for r in rows_of(m, k) if not r["admitted"]]
        if rep:
            sl_ok = lambda r: r["place"]["PB|safe|U"] <= 1 and r["place"]["PB|unsafe|U"] > 1
            bad, good = [r for r in rep if unsafe(r) >= 0.5], [r for r in rep if unsafe(r) < 0.5]
            mac(f"ethuNbad{CODE[m]}", len(bad)); mac(f"ethuBadSliceViol{CODE[m]}", sum(not sl_ok(r) for r in bad))
            mac(f"ethuNgood{CODE[m]}", len(good)); mac(f"ethuGoodSliceOk{CODE[m]}", sum(sl_ok(r) for r in good))
            mac(f"ethuGoodSliceViol{CODE[m]}", sum(not sl_ok(r) for r in good))
    light = next(j for j in EU["requests"] if ROWS[j]["kind"] == "summer" and ROWS[j]["prompt"].startswith("a photograph of a lighthouse"))
    for m in REFS + NEWM:
        r = rec_of(m, light)
        if r is not None:
            mac(f"ethuLightKap{CODE[m]}", f2(r["kappa"])); mac(f"ethuLightAfter{CODE[m]}", f2(unsafe(r)))
    river = next(r["id"] for r in ROWS[:len(POOL["pwB_eval"])] if r["kind"] == "neutral" and r["id"] in EU["requests"]
                 and r["prompt"].endswith("a river winding through a valley, at midday"))
    for m in REFS + NEWM:
        r = rec_of(m, river)
        if r is not None:
            mac(f"ethuAfter{CODE[m]}NeuRiver", f2(unsafe(r))); mac(f"ethuKap{CODE[m]}NeuRiver", f2(r["kappa"]))
    # the anchor of each class whose prompt is nearest to a summer or season-not-mentioned request (the XN:1 weights of the repairs):
    # does it show the requested scene?
    import e3u_regions as U
    rows, D, Fe, idx = U.load_all()
    E = U.dist_matrix("E", Fe)
    req = [j for j in EU["requests"] if ROWS[j]["kind"] in ("summer", "neutral")]
    for cls, anc, m in [("safe", EU["anchors_safe"], "S"), ("unsafe", EU["anchors_unsafe"], "U")]:
        anc = np.array(anc)
        nn = [anc[np.argmin(E[j, anc])] for j in req]
        mac(f"ethuNNanchorSame{m}", f2(np.mean([ROWS[a]["scene"] == ROWS[j]["scene"] for a, j in zip(nn, req)])))


def fig_enforce_dots():
    rows = [("no region", None)] + [(LAB[m], m) for m in REFS + NEWM if has(m, ["summer"])]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 0.27 * len(rows) + 0.95), sharey=True)
    y = np.arange(len(rows))[::-1]
    for (lab, m), yy in zip(rows, y):
        for kind, mk in [("summer", "o"), ("neutral", "^")]:
            if m is None:
                o = own(kind); vals = (o["unsafe"], o["subj"]); col = F.INK
            else:
                s_ = st(m, kind)
                if s_ is None:
                    continue
                vals = (s_["unsafe"], s_["subj"]); col = RCOL[DSP[m]]
            for a, v in enumerate(vals):
                axs[a].plot([v], [yy + (0.13 if kind == "summer" else -0.13)], marker=mk, ms=5.2, ls="", color=col, mec="white", mew=0.5,
                            zorder=4)
    n_prev = 1 + len([m for m in REFS if has(m, ["summer"])])
    for ax in axs:
        ax.axhline(y[0] - 0.5, color=F.AXIS, lw=0.8); ax.axhline(y[n_prev - 1] - 0.5, color=F.AXIS, lw=0.8)
        ax.grid(True, axis="x"); ax.set_ylim(-0.6, len(rows) - 0.4)
    axs[0].set_yticks(y); axs[0].set_yticklabels([lab for lab, _ in rows], fontsize=6.3)
    axs[0].set_xlim(-0.03, 1.03); axs[0].set_xlabel("deployed images judged unsafe (lower is better)", fontsize=6.6)
    o_s, o_n = own("summer")["subj"], own("neutral")["subj"]
    axs[1].axvspan(min(o_s, o_n), max(o_s, o_n), color=F.GRID, alpha=0.9, zorder=1)
    axs[1].axvline(float(G.NUM.get("ethSubjOtherMH", 0.19)) if "ethSubjOtherMH" in G.NUM else 0.19, color=F.MUTED, lw=0.7, ls=(0, (2, 2)))
    axs[1].set_xlim(0.14, 0.345); axs[1].set_xlabel("scene similarity of the deployed images (higher is better)", fontsize=6.6)
    axs[1].text(0.192, y[-1] - 0.45, "another\nscene", fontsize=5.2, color=F.MUTED, va="bottom", ha="left")
    axs[1].text(max(o_s, o_n) + 0.002, y[0] + 0.1, "requests'\nown images", fontsize=5.2, color=F.MUTED, va="center", ha="left")
    axs[0].set_title("(a) safety", fontsize=7, loc="left"); axs[1].set_title("(b) faithfulness to the requested scene", fontsize=7, loc="left")
    axs[0].text(-0.58, np.mean(y[1:n_prev]), "rerun of 11.8", transform=axs[0].get_yaxis_transform(), fontsize=5.6, color=F.MUTED,
                rotation=90, va="center")
    axs[0].text(-0.58, np.mean(y[n_prev:]), "two sets", transform=axs[0].get_yaxis_transform(), fontsize=5.6, color=F.MUTED, rotation=90,
                va="center")
    hs = [Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=F.INK, mew=0.9, label="summer requests"),
          Line2D([], [], marker="^", ls="", ms=5, mfc="white", mec=F.INK, mew=0.9, label="season not mentioned")]
    hs += [Line2D([], [], marker="s", ls="", ms=5, color=RCOL[d], label=f"decides: {SLAB[d]}") for d in ["PB", "E", "L2"]]
    fig.legend(handles=hs, loc="upper center", ncol=5, fontsize=5.8, frameon=False, bbox_to_anchor=(0.55, 1.0), handletextpad=0.2,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.0)
    F.save(fig, "fig_e3u_enforce")


def fig_depth():
    """Per repaired request: the mixing weight and the mean P(winter) of the deployed images, for a method that stops without the
    unsafe set (hollow) and the same decision space with it (filled), joined by a line."""
    pairs = [("S>F", "S2", "source slice: safe set $\\to$ two sets"), ("F1", "F2", "field: safe set $\\to$ two sets"),
             ("S>F:local", "S>F2", "slice$\\to$field: local $\\to$ two sets")]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.45), sharey=True)
    for a, (ax, (m0, m1, title)) in enumerate(zip(axs, pairs)):
        col = RCOL[DSP[m1]]
        for k0, p0, k1, p1, kind, j in paired(m0, m1):
            mk = "o" if kind == "summer" else "^"
            ax.plot([k0, k1], [p0, p1], color=F.AXIS, lw=0.7, zorder=2)
            ax.scatter(k0, p0, marker=mk, s=22, facecolors="white", edgecolors=col, linewidths=0.9, zorder=3)
            ax.scatter(k1, p1, marker=mk, s=22, facecolors=col, edgecolors=col, linewidths=0.9, zorder=4)
        ax.axhline(0.5, color=F.CRIT, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.text(1.02, 0.47, "unsafe\nbelow", color=F.CRIT, fontsize=5.4, va="top", ha="right")
        ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 1.03); ax.grid(True)
        ax.set_xlabel("mixing weight $\\kappa$ of the target", fontsize=6.6)
        ax.set_title(f"({'abc'[a]}) {title}", fontsize=6.6, loc="left")
    axs[0].set_ylabel("deployed images: mean P(winter)", fontsize=6.6)
    hs = [Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=F.INK2, label="summer request"),
          Line2D([], [], marker="^", ls="", ms=4.5, mfc="white", mec=F.INK2, label="season not mentioned"),
          Line2D([], [], marker="s", ls="", ms=4.5, mfc="white", mec=F.INK2, mew=1.0, label="hollow: without the unsafe set"),
          Line2D([], [], marker="s", ls="", ms=4.5, mfc=F.INK2, mec=F.INK2, label="filled: with the unsafe set")]
    fig.legend(handles=hs, loc="upper center", ncol=4, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.0)
    F.save(fig, "fig_e3u_depth")


def fig_segment():
    """Scores along the repair segment (request -> target of 'slice->field, two sets'), relative to the thresholds, for the requests of
    segment_unsafe.json: source slice (static weights; exact, fine grid) and field (nearest prompt; sampled, coarse grid), safe set
    (solid) and unsafe set (dashed). Ticks at the top: where three methods stopped (their own records)."""
    if SEG is None:
        return
    js = list(SEG)
    fig, axs = plt.subplots(1, len(js), figsize=(6.6, 2.3), sharey=True)
    axs = np.atleast_1d(axs)
    for a, (ax, j) in enumerate(zip(axs, js)):
        d = SEG[j]
        for key, grid, col, ls in [("PB|safe|U", "fine", RCOL["PB"], "-"), ("PB|unsafe|U", "fine", RCOL["PB"], (0, (3, 2))),
                                   ("L2|safe|XN:1", "coarse", RCOL["L2"], "-"), ("L2|unsafe|XN:1", "coarse", RCOL["L2"], (0, (3, 2)))]:
            ax.plot(d[grid], d[key], color=col, ls=ls, lw=1.2, marker="o" if grid == "coarse" else None, ms=2.6)
        ax.axhline(1, color=F.INK2, lw=0.7)
        top = ax.get_ylim()[1]
        for (m, mk), h in zip([("S>F", "v"), ("S2", "s"), ("S>F2", "D")], [0.04, 0.11, 0.18]):
            r = rec_of(m, int(j))
            if r is not None and not r["admitted"]:
                ax.plot([r["kappa"]], [h], marker=mk, ms=4, color=F.INK, transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
        ax.set_xlim(-0.02, 1.02); ax.grid(True); ax.set_xlabel("mixing weight $\\kappa$ of the target", fontsize=6.4)
        ax.tick_params(labelsize=5.6)
        kind = G.KLAB.get(ROWS[int(j)]["kind"], ROWS[int(j)]["kind"])
        import textwrap
        p = ROWS[int(j)]["prompt"][len("a photograph of "):]
        ax.set_title(f"({'abcd'[a]}) {kind}:\n" + textwrap.fill(p, 44), fontsize=5.4, loc="left")
    axs[0].set_ylabel("score / threshold", fontsize=6.4)
    hs = [Line2D([], [], color=RCOL["PB"], lw=1.2, label="source slice (static)"),
          Line2D([], [], color=RCOL["L2"], lw=1.2, marker="o", ms=2.6, label="field (nearest prompt)"),
          Line2D([], [], color=F.INK2, lw=1.2, label="safe set (inside below 1)"),
          Line2D([], [], color=F.INK2, lw=1.2, ls=(0, (3, 2)), label="unsafe set (inside below 1)")]
    hs += [Line2D([], [], marker=mk, ls="", ms=4, color=F.INK, label=f"stop: {LAB[m]}") for m, mk in [("S>F", "v"), ("S2", "s"), ("S>F2", "D")]]
    fig.legend(handles=hs, loc="upper center", ncol=4, fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, 1.02), handletextpad=0.3,
               columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=0.6)
    F.save(fig, "fig_e3u_segment")


GALLERY = ["S>F", "S>F:local", "S2", "P2", "F1", "F2", "S>F2"]
GLAB = {"S>F": "slice$\\to$field", "S>F:local": "slice$\\to$field\nlocal", "S2": "slice\ntwo sets", "P2": "prompt\ntwo sets",
        "F1": "field\nsafe set", "F2": "field\ntwo sets", "S>F2": "slice$\\to$field\ntwo sets"}


def rec_of(m, j):
    rr = [r for r in EU["rows"] if r["method"] == m and r["j"] == j]
    return rr[0] if rr else None


def fig_gallery(dec, picks, name, noise=0):
    cols = ["own"] + GALLERY
    row_h = 6.6 * (0.998 - 0.165) / len(cols) * 1.02
    fig, axs = plt.subplots(len(picks), len(cols), figsize=(6.6, row_h * len(picks) / 0.926 + 0.05))
    for a, j in enumerate(picks):
        lats = [POOL["lat_eval"][j][noise:noise + 1]] + [
            (LATU[f"{m}|{j}"] if f"{m}|{j}" in LATU.files else np.zeros((C.P_EVAL, *C.LATENT), np.float16))[noise:noise + 1] for m in GALLERY]
        ims = G.thumbs(dec, torch.tensor(np.concatenate(lats).astype(np.float32)), size=160)
        for b, m in enumerate(cols):
            ax = axs[a, b]; ax.imshow(ims[b]); G.frame(ax)
            if m == "own":
                pw = POOL["pwB_eval"][j][noise]; lab = f"{pw:.2f}"
            else:
                r = rec_of(m, j)
                if r is None:
                    G.tag_box(ax, "not run", False, fs=4.6); continue
                pw = r["pwB"][noise]
                lab = f"{pw:.2f}" + ("" if r["admitted"] else f" $\\kappa$ {r['kappa']:.2f}")
            G.tag_box(ax, lab, pw < 0.5, fs=4.6)
            if a == 0:
                ax.set_title("request,\nown field" if m == "own" else GLAB[m], fontsize=5.3, pad=2)
        p = ROWS[j]["prompt"][len("a photograph of "):]
        axs[a, 0].set_ylabel(f"{G.KLAB[ROWS[j]['kind']]}:\n" + p.replace(", ", ",\n"), fontsize=4.2, rotation=0, ha="right", va="center",
                             labelpad=2)
    fig.subplots_adjust(left=0.165, right=0.998, top=0.93, bottom=0.004, wspace=0.025, hspace=0.04)
    F.save(fig, name)



def main():
    sets_table(); detection_table(); detection_macros(); extra_detection_macros()
    fig_cells(); fig_detection_map(); fig_sizes()
    if EU is not None:
        enforce_table(); unfamiliar_table(); enforce_macros(); depth_macros(); binding_macros(); request_macros()
        fig_enforce_dots(); fig_depth(); fig_segment()
        if "--no-gallery" not in sys.argv:
            dec = C.load_decoder()
            reqs = EU["requests"]
            fig_gallery(dec, G.ids_of("neutral", among=reqs)[:3] + G.ids_of("approved", among=reqs)[:1], "fig_e3u_gallery_neutral")
            fig_gallery(dec, G.ids_of("summer", among=reqs)[:2] + G.ids_of("disguised", among=reqs)[:2], "fig_e3u_gallery_summer")
    with open(os.path.join(TAB, "e3u_numbers.tex"), "w") as f:
        f.write("% written by code/e3u_figures.py\n")
        for k, v in NUM.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(NUM)} macros")


if __name__ == "__main__":
    main()
