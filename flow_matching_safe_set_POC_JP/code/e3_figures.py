"""E3 figures, tables and numbers, from ../results/e3/ (run e3_pool.py, e3_regions.py, e3_repairs.py and e3_enforce.py first).
  fig_e3_requests     what the model generates, unconstrained, for each kind of request (probe noises, judge P(winter))
  fig_e3_detection    exclusion of each kind of request, static against adaptive weights, per region (1 - coverage dashed)
  fig_e3_per_request  how often each request is excluded over the splits, against what its own field generates
  fig_e3_gallery_*    for chosen requests, the output of each region's deployed field on one evaluation noise
  fig_e3_enforce      after enforcement: fraction of deployed images judged unsafe, and scene similarity, per region and kind
Tables: ../tables/e3_summary.tex (one row per region), e3_anchors.tex (anchor count), e3_unfamiliar.tex (safe but unfamiliar
requests), e3_numbers.tex (every number of the note).
Usage: python e3_figures.py
"""
import os, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
import e1_figures as F
import e3_common as C
import e3_regions as R

FIG, TAB = os.path.join(C.HERE, "..", "figures"), os.path.join(C.HERE, "..", "tables")
REG = json.load(open(os.path.join(C.RES, "regions.json")))
REP = json.load(open(os.path.join(C.RES, "repairs.json")))
ENF = json.load(open(os.path.join(C.RES, "enforce.json")))
LAT = np.load(os.path.join(C.RES, "enforce_latents.npz"))
POOL = np.load(os.path.join(C.RES, "pool.npz"))
ROWS = json.load(open(os.path.join(C.RES, "prompts.json")))["prompts"]
KINDS = ["approved", "neutral", "summer", "disguised"]
KLAB = {"approved": "approved (winter)", "neutral": "season not mentioned", "summer": "summer", "disguised": "summer + winter words"}
KCOL = {"approved": F.BLUE, "neutral": F.ORANGE, "summer": F.AQUA, "disguised": F.VIOLET}     # fixed per kind, never by rank
KMAC = {"approved": "App", "neutral": "Neu", "summer": "Sum", "disguised": "Dis"}
SPACES = ["E", "MH", "PB", "L2", "TR"]
SLAB = {"E": "prompt", "MH": "Mahalanobis", "PB": "source slice", "L2": "field $L^2$", "TR": "field truncated"}
SMAC = {"E": "E", "MH": "MH", "PB": "PB", "L2": "Lt", "TR": "TR"}
TMAC = {"static": "S", "adaptive": "A"}
GALLERY_REGIONS = ["E|static", "E|adaptive", "MH|adaptive", "PB|static", "PB|adaptive", "L2|static", "TR|adaptive"]
GLAB = {"E|static": "prompt\nstatic", "E|adaptive": "prompt\nadaptive", "MH|adaptive": "Mahalanobis\nadaptive",
        "PB|static": "source slice\nstatic", "PB|adaptive": "source slice\nadaptive", "L2|static": "field $L^2$\nstatic",
        "TR|adaptive": "field trunc.\nadaptive"}
# Section 11 of safe_prompt_fields.tex: one hue per region in every figure (validated all pairs: orange, violet, aqua, blue);
# the two field regions share blue and differ by marker (square: L^2, circle: truncated); static weights hollow, adaptive filled.
RCOL = {"E": F.ORANGE, "MH": F.VIOLET, "PB": F.AQUA, "L2": F.BLUE, "TR": F.BLUE}
RMARK = {"E": "o", "MH": "o", "PB": "o", "L2": "s", "TR": "o"}
KMARK = {"summer": "o", "neutral": "^", "disguised": "D", "safe_unfamiliar": "s"}     # kinds of request, by marker
SCENE_FIRST = 0                                     # "a pine forest": shown first wherever a kind has it (the user's example)
UNF = ["paraphrase", "novel_winter", "novel_summer"]           # safe but unfamiliar requests (e3_prompts.py --extend)
UMAC = {"paraphrase": "Par", "novel_winter": "NovW", "novel_summer": "NovS"}
ROMAN = {30: "XXX", 60: "LX", 100: "C", 140: "CXL"}
VMAC = {"U": "U", "N:1": "Na", "N:3": "Nc", "K:0.3": "Kc"}
NUM = {}


def mac(k, v):
    NUM[k] = v


def f2(x):
    return f"{x:.2f}"


def design_macros():
    """The constants of the design, read from the code that uses them."""
    import e3_enforce as EN
    sizes = json.load(open(os.path.join(C.RES, "prompts.json")))["sizes"]
    mac("ethGuid", f"{C.W:g}"); mac("ethJ", C.J); mac("ethShift", f"{C.SHIFT:g}"); mac("ethP", C.P_PROBE)
    mac("ethM", R.M); mac("ethK", R.K); mac("ethBeta", f"{R.BETA:g}")
    mac("ethNom", f"{int(np.ceil((R.K + 1) * (1 - R.BETA))) / (R.K + 1):.3f}")
    mac("ethNtune", R.N_TUNE); mac("ethNeval", R.N_EVAL)
    for k, key in [("ethNappTune", "approved_tune"), ("ethNappEval", "approved_eval"), ("ethNsumTune", "summer_tune"),
                   ("ethNsumEval", "summer_eval"), ("ethNneu", "neutral_eval"), ("ethNdis", "disguised_eval")]:
        mac(k, sizes[key])
    mac("ethNapp", sizes["approved_tune"] + sizes["approved_eval"])
    mac("ethNtest", sizes["approved_eval"] - R.M - R.K)
    kinds = [ROWS[j]["kind"] for j in ENF["requests"]]
    mac("ethNreqKind", kinds.count("summer")); mac("ethNreqApp", kinds.count("approved"))
    mac("ethBisField", EN.N_BIS_FIELD); mac("ethBisCheap", EN.N_BIS_CHEAP)
    mac("ethTRlevels", ", ".join(f"{r:g}" for r in REG["tr_levels"]))
    mac("ethNsplitsRep", REP["n_splits"])


def ids_of(kind, pool="eval", among=None):
    """Prompts of a kind, the pine forest first if there is one."""
    js = [r["id"] for r in ROWS if r["kind"] == kind and r["pool"] == pool and (among is None or r["id"] in among)]
    return sorted(js, key=lambda j: (ROWS[j]["scene"] != SCENE_FIRST, js.index(j)))


def own_macros():
    """What the model generates, unconstrained, on the evaluation noise (the judge's view)."""
    allv = []
    for k in KINDS:
        js = ids_of(k)
        pw = POOL["pwB_eval"][js]
        mac(f"ethOwnPw{KMAC[k]}", f2(pw.mean())); mac(f"ethOwnUnsafe{KMAC[k]}", f2((pw < 0.5).mean()))
        mac(f"ethOwnGreen{KMAC[k]}", f2(POOL["green_eval"][js].mean())); mac(f"ethOwnSnow{KMAC[k]}", f2(POOL["snow_eval"][js].mean()))
        agree = (POOL["pwL_eval"][js] < 0.5) == (pw < 0.5)
        mac(f"ethAgree{KMAC[k]}", f2(agree.mean())); allv.append(agree.ravel())
        if k == "disguised":
            mac("ethDisSafe", int((pw.mean(1) >= 0.5).sum()))
    mac("ethAgree", f2(np.concatenate(allv).mean()))


def detection_macros():
    covs = []
    for sp in SPACES:
        for tag in ["static", "adaptive"]:
            e = REG["eval"][f"{sp}|{tag}"]; b = REG["best"][f"{sp}|{tag}"]; t = SMAC[sp] + TMAC[tag]
            covs.append(e["test"])
            mac(f"ethCov{t}", f"{e['test']:.3f}")
            for g, gm in [("summer", "Sum"), ("neutral", "Neu"), ("disguised", "Dis"), ("judged_unsafe", "JU"), ("judged_safe", "JS")]:
                mac(f"ethEx{gm}{t}", f2(e[g]))
            mac(f"ethVar{t}", b["variant"]); mac(f"ethTuneJ{t}", f2(b["tune_J"]))
    mac("ethCovMin", f"{min(covs):.3f}"); mac("ethCovMax", f"{max(covs):.3f}")
    rho = REG["best"]["MH|static"]["par"]; e10 = int(round(np.log10(rho)))
    mac("ethRhoMH", f"10^{{{e10}}}" if np.isclose(rho, 10.0 ** e10) else f"{rho:g}"); mac("ethRTR", f"{REG['best']['TR|static']['par']:g}")
    labels = REG["labels"]
    mac("ethNJS", sum(1 for v in labels.values() if not v)); mac("ethNJU", sum(1 for v in labels.values() if v))
    for kind in ["neutral", "summer"]:                                # the pine forest requests: how often each region excludes them
        j = ids_of(kind, among=ENF["requests"])[0]
        for sp in SPACES:
            for tag in ["static", "adaptive"]:
                mac(f"ethPine{KMAC[kind]}{SMAC[sp]}{TMAC[tag]}", f2(REG["per_request"][f"{sp}|{tag}"][str(j)]))
        mac(f"ethPineOwnPw{KMAC[kind]}", f2(POOL["pwB_eval"][j].mean()))


def repair_macros():
    for sp in SPACES:
        g = REP["gaps"][sp]
        mac(f"ethGap{SMAC[sp]}", f2(g["ratio"])); mac(f"ethGapNeu{SMAC[sp]}", f2(g["ratio_neutral"]))
    for k, v in REP["centrality"].items():
        mac({"median": "ethCentMed", "q90": "ethCentQ", "pine_approved": "ethCentPineApp", "pine_neutral": "ethCentPineNeu",
             "pine_summer": "ethCentPineSum"}[k], f2(v))
    for M, v in REP["missing_scene"].items():
        mac(f"ethMiss{ROMAN[int(M)]}", f2(v))
    covs = [v["test"] for v in REP["anchors"].values()]
    mac("ethRepCovMin", f2(min(covs))); mac("ethRepCovMax", f2(max(covs)))
    for key, v in REP["anchors"].items():
        sp, vn, M = key.split("|")
        for g, gm in [("summer", "Sum"), ("neutral", "Neu"), ("disguised", "Dis")] + [(k, UMAC[k]) for k in UNF if k in v]:
            mac(f"ethRep{SMAC[sp]}{VMAC[vn]}{ROMAN[int(M)]}{gm}", f2(v[g]))


def unfamiliar():
    """Winter prompts with new wording or new scenes, summer prompts with new scenes: judge view and exclusion per region."""
    present = [k for k in UNF if k in REG["eval"]["E|static"]]
    if not present:
        return
    for k in present:
        js = [r["id"] for r in ROWS if r["kind"] == k]
        pw = POOL["pwB_eval"][js]
        mac(f"ethN{UMAC[k]}", len(js)); mac(f"ethOwnPw{UMAC[k]}", f2(pw.mean())); mac(f"ethOwnUnsafe{UMAC[k]}", f2((pw < 0.5).mean()))
    lines = []
    for sp in SPACES:
        for tag in ["static", "adaptive"]:
            key = f"{sp}|{tag}"; e = REG["eval"][key]; b = REG["best"][key]; t = SMAC[sp] + TMAC[tag]
            for k in present:
                mac(f"ethEx{UMAC[k]}{t}", f2(e[k]))
            wl = "static" if b["variant"] == "U" else f"adaptive, {b['variant']}"
            cells = [f"{e['test']:.3f}"] + [f2(e[k]) for k in present] + [f2(e["summer"])]
            lines.append(f"{SLAB[sp] if tag == 'static' else ''} & {wl} & " + " & ".join(cells) + " \\\\")
        if sp != SPACES[-1]:
            lines.append("\\midrule")
    open(os.path.join(TAB, "e3_unfamiliar.tex"), "w").write("\n".join(lines) + "\n")


def thumbs(dec, lat, size=192):
    return [np.asarray(Image.fromarray(a).resize((size, size), Image.LANCZOS)) for a in C.decode(dec, lat)]


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def tag_box(ax, text, unsafe, fs=5.2):
    ax.text(0.03, 0.04, text, transform=ax.transAxes, fontsize=fs, color="white",
            bbox=dict(facecolor=F.CRIT if unsafe else "black", alpha=0.6, pad=0.8, linewidth=0))


def fig_requests(dec, n_noise=3):
    """Row per kind of request, two prompts per row: what the unconstrained model generates on the first probe noises."""
    import textwrap
    fig, axs = plt.subplots(len(KINDS), 2 * n_noise, figsize=(6.6, 1.32 * len(KINDS)))
    for a, k in enumerate(KINDS):
        js = ids_of(k)
        pair = [js[0], next(j for j in js[1:] if ROWS[j]["scene"] != ROWS[js[0]]["scene"])]
        for h, j in enumerate(pair):
            ims = thumbs(dec, torch.tensor(POOL["lat_probe"][j][:n_noise].astype(np.float32)))
            for b in range(n_noise):
                ax = axs[a, h * n_noise + b]; ax.imshow(ims[b]); frame(ax)
                pw = POOL["pwB_probe"][j][b]
                tag_box(ax, f"P(winter) {pw:.2f}", pw < 0.5, fs=4.8)
            text = textwrap.fill(ROWS[j]["prompt"][len("a photograph of "):], 66)
            axs[a, h * n_noise].set_title(f"{KLAB[k]}: " + text, loc="left", fontsize=5.0, pad=2)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.955, bottom=0.004, wspace=0.03, hspace=0.36)
    F.save(fig, "fig_e3_requests")


def rec_of(g, j):
    return next(r for r in ENF["rows"] if r["region"] == g and r["j"] == j)


def fig_gallery(dec, picks, name, noise=0):
    """Rows: requests; columns: the request's own field, then the field each region deploys (same evaluation noise)."""
    cols = ["own"] + GALLERY_REGIONS
    fig, axs = plt.subplots(len(picks), len(cols), figsize=(6.6, 0.83 * len(picks) + 0.3))
    for a, j in enumerate(picks):
        lats = [POOL["lat_eval"][j][noise:noise + 1]] + [LAT[f"{g}|{j}"][noise:noise + 1] for g in GALLERY_REGIONS]
        ims = thumbs(dec, torch.tensor(np.concatenate(lats).astype(np.float32)), size=160)
        for b, g in enumerate(cols):
            ax = axs[a, b]; ax.imshow(ims[b]); frame(ax)
            if g == "own":
                pw = POOL["pwB_eval"][j][noise]; lab = f"{pw:.2f}"
            else:
                rec = rec_of(g, j); pw = rec["pwB"][noise]
                lab = f"{pw:.2f}" + ("" if rec["kappa"] == 0 else f" $\\kappa$ {rec['kappa']:.2f}")
            tag_box(ax, lab, pw < 0.5, fs=4.6)
            if a == 0:
                ax.set_title("request,\nown field" if g == "own" else GLAB[g], fontsize=5.3, pad=2)
        p = ROWS[j]["prompt"][len("a photograph of "):]
        axs[a, 0].set_ylabel(f"{KLAB[ROWS[j]['kind']]}:\n" + p.replace(", ", ",\n"), fontsize=4.2, rotation=0, ha="right", va="center",
                             labelpad=2)
    fig.subplots_adjust(left=0.165, right=0.998, top=0.93, bottom=0.004, wspace=0.025, hspace=0.04)
    F.save(fig, name)


def fig_detection():
    fig, axs = plt.subplots(1, len(SPACES), figsize=(6.6, 2.0), sharey=True)
    groups = ["summer", "neutral", "disguised"]
    for ax, sp in zip(axs, SPACES):
        for k, tag in enumerate(["static", "adaptive"]):
            e = REG["eval"][f"{sp}|{tag}"]
            x = np.arange(len(groups)) + (k - 0.5) * 0.38
            ax.bar(x, [e[g] for g in groups], width=0.36, color=[F.ORANGE, F.BLUE][k], label=tag, edgecolor="white", linewidth=0.6)
            ax.plot([-0.5, 2.5], [1 - e["test"]] * 2, color=[F.ORANGE, F.BLUE][k], lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(["summer", "no season", "disguised"], fontsize=5.6)
        ax.set_title(SLAB[sp], fontsize=6.4, loc="left"); ax.set_ylim(0, 1.04); ax.set_xlim(-0.55, 2.55); ax.grid(True, axis="y")
    axs[0].set_ylabel("requests excluded", fontsize=6.5)
    axs[0].legend(fontsize=5.6, loc="upper left")
    fig.tight_layout(w_pad=0.4)
    F.save(fig, "fig_e3_detection")


def fig_per_request(tag="static"):
    """How often each request is excluded over the evaluation splits, against what its own field generates (judge P(winter),
    evaluation noise), for the static version of every region. Colour: the region; marker: the kind of request."""
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, len(SPACES), figsize=(6.6, 2.1), sharey=True)
    for ax, sp in zip(axs, SPACES):
        ex = REG["per_request"][f"{sp}|{tag}"]
        for k in ["summer", "neutral", "disguised", "safe_unfamiliar"]:
            kinds = ["paraphrase", "novel_winter"] if k == "safe_unfamiliar" else [k]
            js = [int(j) for j in ex if ROWS[int(j)]["kind"] in kinds]
            filled = k in ("summer", "neutral")
            ax.scatter([POOL["pwB_eval"][j].mean() for j in js], [ex[str(j)] for j in js], s=13, marker=KMARK[k],
                       facecolors=RCOL[sp] if filled else "white", edgecolors=RCOL[sp], linewidths=0.7, zorder=3)
        ax.axvline(0.5, color=F.MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.05, 1.05); ax.grid(True)
        ax.set_title(SLAB[sp], fontsize=6.6, loc="left", color=F.INK)
        ax.set_xlabel("its images: P(winter)", fontsize=6)
    axs[0].set_ylabel("excluded (fraction of splits)", fontsize=6.4)
    hs = [Line2D([], [], marker=KMARK[k], ls="", ms=4.5, mfc=F.INK2 if k in ("summer", "neutral") else "white", mec=F.INK2,
                 mew=0.8, label=lab) for k, lab in [("summer", "summer"), ("neutral", "season not mentioned"),
                                                   ("disguised", "summer + winter words"),
                                                   ("safe_unfamiliar", "safe unfamiliar (paraphrase, new scene)")]]
    fig.legend(handles=hs, fontsize=5.8, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.1)
    fig.tight_layout(w_pad=0.3, rect=(0, 0, 1, 0.91))
    F.save(fig, "fig_e3_per_request")


def enforced_regions():
    done = {r["region"] for r in ENF["rows"]}
    return [f"{sp}|{t}" for sp in SPACES for t in ["static", "adaptive"] if f"{sp}|{t}" in done]


def fig_enforce():
    """Per region (x), one dot per kind of request: (a) fraction of deployed images judged unsafe, (b) scene similarity."""
    regs = enforced_regions()
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.25))
    kinds = ["neutral", "summer", "disguised", "approved"]
    for a, (metric, lab) in enumerate([("unsafe", "deployed images judged unsafe"), ("subj", "scene similarity (judge)")]):
        ax = axs[a]
        for k, kind in enumerate(kinds):
            js = [j for j in ENF["requests"] if ROWS[j]["kind"] == kind]
            own = (np.mean(POOL["pwB_eval"][js] < 0.5) if metric == "unsafe" else np.mean(POOL["subj_eval"][js]))
            vals = []
            for g in regs:
                rr = [r for r in ENF["rows"] if r["region"] == g and r["kind"] == kind]
                vals.append(np.mean([np.mean(np.array(r["pwB"]) < 0.5) if metric == "unsafe" else np.mean(r["subj"]) for r in rr]))
            x = np.arange(len(regs) + 1) + (k - 1.5) * 0.13
            ax.scatter(x, [own] + vals, s=11, color=KCOL[kind], label=KLAB[kind], zorder=3, linewidths=0.5, edgecolors="white")
        ax.axvline(0.5, color=F.MUTED, lw=0.6)
        ax.set_xticks(np.arange(len(regs) + 1))
        ax.set_xticklabels(["no region"] + [SLAB[g.split("|")[0]] + (" S" if g.endswith("static") else " A") for g in regs],
                           fontsize=5.2, rotation=55, ha="right")
        ax.set_title(f"({'ab'[a]}) {lab}", fontsize=6.4, loc="left"); ax.grid(True, axis="y")
    axs[0].set_ylim(-0.04, 1.04)
    axs[0].legend(fontsize=5.0, loc="upper right")
    fig.tight_layout(w_pad=0.8)
    F.save(fig, "fig_e3_enforce")


def enforce_stats(g, kind):
    rr = [r for r in ENF["rows"] if r["region"] == g and r["kind"] == kind]
    if not rr:
        return None
    alt = [r for r in rr if r["kappa"] > 0]
    return dict(n=len(rr), alt=np.mean([r["kappa"] > 0 for r in rr]), unsafe=np.mean([np.mean(np.array(r["pwB"]) < 0.5) for r in rr]),
                pw=np.mean([np.mean(r["pwB"]) for r in rr]), subj=np.mean([np.mean(r["subj"]) for r in rr]),
                kappa=np.mean([r["kappa"] for r in alt]) if alt else np.nan, full=np.mean([r["kappa"] >= 1 for r in rr]))


def enforce_macros():
    for kind in KINDS:
        js = [j for j in ENF["requests"] if ROWS[j]["kind"] == kind]
        mac(f"ethReqOwnUnsafe{KMAC[kind]}", f2(np.mean(POOL["pwB_eval"][js] < 0.5)))
        mac(f"ethReqOwnSubj{KMAC[kind]}", f"{np.mean(POOL['subj_eval'][js]):.3f}")
        for g in enforced_regions():
            s = enforce_stats(g, kind); sp, tag = g.split("|"); t = SMAC[sp] + TMAC[tag]
            mac(f"ethAlt{KMAC[kind]}{t}", f2(s["alt"])); mac(f"ethAfter{KMAC[kind]}{t}", f2(s["unsafe"]))
            mac(f"ethSubj{KMAC[kind]}{t}", f"{s['subj']:.3f}"); mac(f"ethFull{KMAC[kind]}{t}", f2(s["full"]))
            mac(f"ethKap{KMAC[kind]}{t}", "--" if np.isnan(s["kappa"]) else f2(s["kappa"]))
    cert = [r for r in ENF["rows"] if "cert" in r]
    for sp in ["L2", "TR"]:
        cs = [r["cert"] for r in cert if r["region"].startswith(sp + "|")]
        mac(f"ethCertMin{SMAC[sp]}", f2(min(cs))); mac(f"ethCertMed{SMAC[sp]}", f2(np.median(cs)))
    mac("ethActualMax", f2(max(r["actual_BL"] for r in cert)))
    eps = {r["region"]: r["eps"] for r in ENF["rows"]}
    mac("ethEpsTRA", f2(eps["TR|adaptive"])); mac("ethEpsLtS", f2(eps["L2|static"]))
    mac("ethEpsTRAraw", f2(eps["TR|adaptive"] * REG["best"]["TR|adaptive"]["par"]))


def mechanism_macros():
    """How uniform the tuned kernels are, whether the target keeps the request's scene, when a mixture stays unsafe, and why the
    certificates are vacuous (the anchors' outputs are closer to B than to each other)."""
    from collections import Counter
    _, D, idx = R.load()
    anc, req = np.array(ENF["anchors"]), np.array(ENF["requests"])
    for sp in ["L2", "TR"]:                                          # effective number of anchors 1 / sum w_i^2 on the requests
        b = REG["best"][f"{sp}|adaptive"]; kind, par = R.parse(b["variant"])
        if kind == "K":
            S = R.Space(sp, D, idx, b["par"]); Fx = S.feats(D)
            h = par * np.median(S.dist(Fx[anc], Fx[anc])[np.triu_indices(len(anc), 1)])
            L = -S.dist(Fx[req], Fx[anc]).astype(np.float64) ** 2 / (2 * h * h)
            w = np.exp(L - L.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            mac(f"ethNeff{SMAC[sp]}A", f"{np.median(1 / (w ** 2).sum(1)):.1f}")
    alt = [r for r in ENF["rows"] if r["kappa"] > 0 and r["kind"] != "approved"]
    for g in enforced_regions():
        rr = [r for r in alt if r["region"] == g]; sp, tag = g.split("|")
        mac(f"ethTgtSame{SMAC[sp]}{TMAC[tag]}",
            f2(np.mean([ROWS[r["target"]]["scene"] == ROWS[r["j"]]["scene"] for r in rr])) if rr else "--")
    for lab, sps in [("Field", ("L2", "TR")), ("PB", ("PB",))]:      # images still unsafe after a small or a large mixture
        rr = [r for r in alt if r["region"].split("|")[0] in sps]
        for nm, sel in [("Small", [r for r in rr if r["kappa"] < 0.5]), ("Large", [r for r in rr if r["kappa"] >= 0.5])]:
            mac(f"ethRes{nm}{lab}", f2(np.mean([np.mean(np.array(r["pwB"]) < 0.5) for r in sel])))
            mac(f"ethNRes{nm}{lab}", len(sel))
        if lab == "Field":                                           # the requests mixed least are those that score least above eps
            from scipy.stats import spearmanr
            s0 = np.array([r["s0"] for r in rr])
            mac("ethSpearField", f2(spearmanr(s0, [r["kappa"] for r in rr])[0])); mac("ethSmaxFieldPct", int(np.ceil(100 * (s0.max() - 1))))
    for kind in ["neutral", "summer"]:                               # the pine forests: mixture and unsafe images (of P) per region
        j = ids_of(kind, among=ENF["requests"])[0]
        for g in enforced_regions():
            r = rec_of(g, j); sp, tag = g.split("|"); t = SMAC[sp] + TMAC[tag]
            mac(f"ethPineKap{KMAC[kind]}{t}", f2(r["kappa"])); mac(f"ethPineAfter{KMAC[kind]}{t}", int(np.sum(np.array(r["pwB"]) < 0.5)))
    rr = [r for r in alt if r["region"] == "MH|static" and r["kind"] in ("summer", "neutral")]
    sc, n = Counter(ROWS[r["target"]]["scene"] for r in rr).most_common(1)[0]
    mac("ethHubMH", n); mac("ethHubNMH", len(rr))
    mac("ethHubSceneMH", next(ROWS[r["target"]]["subject_text"] for r in rr if ROWS[r["target"]]["scene"] == sc)[len("a photograph of "):])
    chi = C.Clip("openai/clip-vit-large-patch14")
    nL = chi.tw - chi.to; nL /= np.linalg.norm(nL)                   # B = {u : <u, nL> < 0}, where ViT-L/14 says "not winter"
    u = POOL["chi_probe"][anc].astype(np.float64)                    # (M, P, 768): the anchors' probe outputs
    s = u @ nL
    mac("ethMargMed", f"{np.median(s):.3f}"); mac("ethMargMax", f"{s.max():.2f}")
    mac("ethChiSpread", f2(np.median(R.probe_dists(u, u)[np.triu_indices(len(anc), 1)])))


def n_kind(k):
    return sum(1 for r in ROWS if r["kind"] == k)


def false_alarms(e):
    """Fraction of the safe unfamiliar requests (paraphrases and novel winter scenes, pooled) that a region excludes."""
    ks = ["paraphrase", "novel_winter"]
    return sum(n_kind(k) * e[k] for k in ks) / sum(n_kind(k) for k in ks)


def mark(ax, x, y, sp, tag):
    """A region's point: a hollow ring for static weights, a filled dot for adaptive ones (a dot in a ring when they coincide)."""
    if tag == "static":
        ax.scatter(x, y, marker=RMARK[sp], s=52, facecolors="white", edgecolors=RCOL[sp], linewidths=1.2, zorder=3)
    else:
        ax.scatter(x, y, marker=RMARK[sp], s=18, facecolors=RCOL[sp], edgecolors=RCOL[sp], linewidths=0.6, zorder=4)


def region_handles(spaces=SPACES, none=False):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=RMARK[sp], ls="", ms=5.5, mfc=RCOL[sp], mec=RCOL[sp], label=SLAB[sp]) for sp in spaces]
    hs += [Line2D([], [], marker="o", ls="", ms=6.5, mfc="white", mec=F.INK2, mew=1.2, label="static weights"),
           Line2D([], [], marker="o", ls="", ms=3.8, mfc=F.INK2, mec=F.INK2, label="adaptive weights")]
    if none:
        hs.append(Line2D([], [], marker="x", ls="", ms=5, mec=F.INK, mew=1.2, label="no region"))
    return hs


def label_points(ax, items):
    """Direct labels: (text, x, y, dx, dy, ha) with offsets in points."""
    for text, x, y, dx, dy, ha in items:
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.6, color=F.INK2, ha=ha, va="center")


def fig_detection_map():
    """Unsafe requests excluded against safe unfamiliar requests excluded, one point per region and weighting (M = 30)."""
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.85))
    fa_app = 1 - np.mean([REG["eval"][f"{sp}|{t}"]["test"] for sp in SPACES for t in ["static", "adaptive"]])
    for a, (ax, g, title) in enumerate(zip(axs, ["summer", "neutral"],
                                           ["summer requests", "requests that do not mention the season"])):
        pts = {}
        for sp in SPACES:
            for tag in ["static", "adaptive"]:
                e = REG["eval"][f"{sp}|{tag}"]; pts[sp, tag] = (false_alarms(e), e[g])
                mark(ax, *pts[sp, tag], sp, tag)
        ax.axvline(fa_app, color=F.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.text(fa_app - 0.012, 0.02, "$1-$coverage", rotation=90, fontsize=5.3, color=F.MUTED, ha="right", va="bottom")
        ax.annotate("better", xy=(0.47, 0.8), xytext=(0.66, 0.62), fontsize=6, color=F.INK2, ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=F.INK2, lw=0.8))
        label_points(ax, [("prompt", *pts["E", "static"], 7, 6, "left"), ("Mahalanobis", *pts["MH", "static"], -8, -9, "right"),
                          ("source slice", *pts["PB", "adaptive"], 7, -5, "left"),
                          ("field $L^2$", *pts["L2", "static"], -6, 7, "right"),
                          ("field truncated", *pts["TR", "adaptive"], 8, -6, "left")])
        ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.07); ax.grid(True)
        ax.set_xlabel("safe unfamiliar requests excluded (false alarms)", fontsize=6.6)
        ax.set_ylabel("unsafe requests excluded", fontsize=6.6)
        ax.set_title(f"({'ab'[a]}) {title}", fontsize=7, loc="left")
    fig.legend(handles=region_handles(), loc="upper center", ncol=7, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.1, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.6)
    F.save(fig, "fig_e3_detection_map")


def fig_anchors():
    """Exclusion against the number M of anchors, static against nearest-anchor weights (all approved prompts, 200 splits)."""
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35), sharey=True)
    Ms = REP["M_grid"]; sps = ["E", "PB", "TR"]
    fa_app = 1 - np.mean([v["test"] for v in REP["anchors"].values()])
    for a, (ax, g, title) in enumerate(zip(axs, ["summer", "neutral", "novel_winter"],
                                           ["summer requests (unsafe)", "season not mentioned (unsafe)",
                                            "winter, unseen scenes (safe)"])):
        for sp in sps:
            for vn in ["U", "N:1"]:
                y = [REP["anchors"][f"{sp}|{vn}|{M}"][g] for M in Ms]
                ax.plot(Ms, y, color=RCOL[sp], ls="-" if vn == "N:1" else (0, (3, 2)), lw=1.3, marker=RMARK[sp], ms=4,
                        mfc=RCOL[sp] if vn == "N:1" else "white", mec=RCOL[sp], mew=1.0, zorder=3)
        if g == "novel_winter":
            ax.axhline(fa_app, color=F.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.set_xticks(Ms); ax.set_xlabel("number of anchors $M$", fontsize=6.6)
        ax.set_ylim(-0.04, 1.04); ax.grid(True)
        ax.set_title(f"({'abc'[a]}) {title}", fontsize=7, loc="left")
    axs[0].set_ylabel("fraction excluded", fontsize=6.6)
    hs = [Line2D([], [], color=RCOL[sp], lw=1.3, marker=RMARK[sp], ms=4, label=SLAB[sp]) for sp in sps]
    hs += [Line2D([], [], color=F.INK2, lw=1.2, ls=(0, (3, 2)), marker="o", ms=4, mfc="white", label="static weights"),
           Line2D([], [], color=F.INK2, lw=1.2, marker="o", ms=4, label="nearest anchor (adaptive)"),
           Line2D([], [], color=F.MUTED, lw=0.7, ls=(0, (2, 2)), label="$1-$coverage")]
    fig.legend(handles=hs, loc="upper center", ncol=6, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8)
    F.save(fig, "fig_e3_anchors")


def fig_enforce_map():
    """After enforcement: fraction of deployed images judged unsafe against their scene similarity, per region and weighting."""
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.85), sharey=True)
    for a, (ax, kind, title) in enumerate(zip(axs, ["summer", "neutral"],
                                              ["summer requests", "requests that do not mention the season"])):
        js = [j for j in ENF["requests"] if ROWS[j]["kind"] == kind]
        x0, y0 = np.mean(POOL["subj_eval"][js]), np.mean(POOL["pwB_eval"][js] < 0.5)
        ax.scatter(x0, y0, marker="x", s=34, color=F.INK, linewidths=1.3, zorder=6)
        pts = {}
        for g in enforced_regions():
            st = enforce_stats(g, kind); sp, tag = g.split("|"); pts[sp, tag] = (st["subj"], st["unsafe"])
            mark(ax, *pts[sp, tag], sp, tag)
        items = [("no region", x0, y0, -8, 0, "right"), ("prompt", *pts["E", "adaptive"], -8, -4, "right"),
                 ("Mahalanobis", *pts["MH", "static"], 8, 6, "left"), ("source slice", *pts["PB", "static"], -8, 0, "right"),
                 ("field ($L^2$, truncated)", *pts["L2", "static"], -8, -3, "right")]
        label_points(ax, items)
        ax.annotate("better", xy=(0.235, 0.58), xytext=(0.2, 0.9), fontsize=6, color=F.INK2, ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=F.INK2, lw=0.8))
        ax.set_xlim(0.17, 0.345); ax.set_ylim(-0.05, 1.07); ax.grid(True)
        ax.set_xlabel("scene similarity of the deployed images", fontsize=6.6)
        ax.set_title(f"({'ab'[a]}) {title}", fontsize=7, loc="left")
    axs[0].set_ylabel("deployed images judged unsafe", fontsize=6.6)
    fig.legend(handles=region_handles(none=True), loc="upper center", ncol=8, fontsize=5.8, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.1, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.2)
    F.save(fig, "fig_e3_enforce_map")


def map_macros():
    """Numbers of the Section 11 figures: false alarms per region, the reference rates, scene similarity by target scene."""
    for sp in SPACES:
        for tag in ["static", "adaptive"]:
            mac(f"ethFA{SMAC[sp]}{TMAC[tag]}", f2(false_alarms(REG["eval"][f"{sp}|{tag}"])))
    mac("ethFAApp", f2(1 - np.mean([REG["eval"][f"{sp}|{t}"]["test"] for sp in SPACES for t in ["static", "adaptive"]])))
    mac("ethNsafeUnf", n_kind("paraphrase") + n_kind("novel_winter"))
    alt = [r for r in ENF["rows"] if r["region"] == "MH|static" and r["kappa"] > 0 and r["kind"] != "approved"]
    for nm, same in [("Same", True), ("Other", False)]:
        mac(f"ethSubj{nm}MH", f"{np.mean([np.mean(r['subj']) for r in alt if (ROWS[r['target']]['scene'] == ROWS[r['j']]['scene']) == same]):.2f}")
    d = [abs(REP["anchors"][f"L2|{vn}|{M}"][g] - REP["anchors"][f"TR|{vn}|{M}"][g])
         for vn in ["U", "N:1"] for M in REP["M_grid"] for g in ["summer", "neutral", "novel_winter"]]
    mac("ethAncLtTRdiff", f2(max(d)))
    d = [abs(REG["eval"][f"L2|{t}"][g] - REG["eval"][f"TR|{t}"][g]) for t in ["static", "adaptive"]
         for g in ["test", "summer", "neutral", "disguised", "novel_summer", "paraphrase", "novel_winter"]]
    mac("ethDetLtTRdiff", f2(max(d)))
    d = [abs(enforce_stats("L2|static", k)["unsafe"] - enforce_stats("TR|static", k)["unsafe"]) for k in KINDS]
    mac("ethEnfLtTRdiff", f2(max(d)))
    n_alt = lambda g, k: sum(r["kappa"] > 0 for r in ENF["rows"] if r["region"] == g and r["kind"] == k)
    mac("ethEnfLtTRaltdiff", max(abs(n_alt("L2|static", k) - n_alt("TR|static", k)) for k in KINDS))


def tables_manuscript():
    """Table bodies of Section 11 of safe_prompt_fields.tex: detection (tab:e3_detection) and enforcement (tab:e3_enforce)."""
    wlab = lambda b: "static" if b["variant"] == "U" else f"adaptive ({b['variant']})"
    lines = []
    for sp in SPACES:
        for tag in ["static", "adaptive"]:
            key = f"{sp}|{tag}"; e = REG["eval"][key]
            cells = [f"{e['test']:.3f}"] + [f2(e[g]) for g in ["summer", "neutral", "novel_summer", "paraphrase", "novel_winter",
                                                                "disguised"]]
            lines.append(f"{SLAB[sp] if tag == 'static' else ''} & {wlab(REG['best'][key])} & " + " & ".join(cells) + " \\\\")
        if sp != SPACES[-1]:
            lines.append("\\midrule")
    open(os.path.join(TAB, "e3_detection.tex"), "w").write("\n".join(lines) + "\n")
    req = {k: [j for j in ENF["requests"] if ROWS[j]["kind"] == k] for k in KINDS}
    own = lambda k: f2(np.mean(POOL["pwB_eval"][req[k]] < 0.5))
    subj0 = np.mean([POOL["subj_eval"][j].mean() for k in ["summer", "neutral", "disguised"] for j in req[k]])
    lines = [f"no region & & 0 & -- & {own('summer')} & 0 & -- & {own('neutral')} & {own('disguised')} & {subj0:.3f} & -- & 0 \\\\",
             "\\midrule"]
    for sp in SPACES:
        for tag in ["static", "adaptive"]:
            g = f"{sp}|{tag}"
            if g in enforced_regions():
                st = {k: enforce_stats(g, k) for k in KINDS}
                rr = [r for r in ENF["rows"] if r["region"] == g and r["kind"] != "approved"]
                alt = [r for r in rr if r["kappa"] > 0]
                cells = []
                for k in ["summer", "neutral"]:
                    cells += [f2(st[k]["alt"]), "--" if np.isnan(st[k]["kappa"]) else f2(st[k]["kappa"]), f2(st[k]["unsafe"])]
                same = np.mean([ROWS[r["target"]]["scene"] == ROWS[r["j"]]["scene"] for r in alt]) if alt else np.nan
                cells += [f2(st["disguised"]["unsafe"]), f"{np.mean([np.mean(r['subj']) for r in rr]):.3f}",
                          "--" if np.isnan(same) else f2(same), f2(st["approved"]["alt"])]
            else:
                cells = ["\\multicolumn{10}{c}{not enforced: its weights are nearly uniform}"]
            lines.append(f"{SLAB[sp] if tag == 'static' else ''} & {wlab(REG['best'][g])} & " + " & ".join(cells) + " \\\\")
        if sp != SPACES[-1]:
            lines.append("\\midrule")
    open(os.path.join(TAB, "e3_enforce.tex"), "w").write("\n".join(lines) + "\n")


def tables():
    req = {k: [j for j in ENF["requests"] if ROWS[j]["kind"] == k] for k in KINDS}
    own_unsafe = {k: np.mean(POOL["pwB_eval"][req[k]] < 0.5) for k in KINDS}
    own_subj = np.mean([POOL["subj_eval"][j].mean() for k in ["summer", "neutral", "disguised"] for j in req[k]])
    lines = ["no region & -- & 1 & 0 & 0 & 0 & " + " & ".join(f2(own_unsafe[k]) for k in ["summer", "neutral", "disguised"])
             + f" & 0 & {own_subj:.3f} \\\\", "\\midrule"]
    for sp in SPACES:
        for tag in ["static", "adaptive"]:
            key = f"{sp}|{tag}"; e = REG["eval"][key]; b = REG["best"][key]
            wl = "static" if b["variant"] == "U" else f"adaptive, {b['variant']}"
            cells = [f"{e['test']:.3f}", f2(e["summer"]), f2(e["neutral"]), f2(e["disguised"])]
            st = {k: enforce_stats(key, k) for k in KINDS}
            if st["summer"] is None:
                cells += ["--"] * 5
            else:
                subj = np.mean([np.mean(r["subj"]) for r in ENF["rows"] if r["region"] == key and r["kind"] != "approved"])
                cells += [f2(st[k]["unsafe"]) for k in ["summer", "neutral", "disguised"]] + [f2(st["approved"]["alt"]), f"{subj:.3f}"]
            lines.append(f"{SLAB[sp] if tag == 'static' else ''} & {wl} & " + " & ".join(cells) + " \\\\")
        if sp != SPACES[-1]:
            lines.append("\\midrule")
    open(os.path.join(TAB, "e3_summary.tex"), "w").write("\n".join(lines) + "\n")
    # anchor count: exclusion for M in the grid, uniform against nearest-anchor weights; the last block is the false alarms on
    # winter scenes that no approved prompt shows (the disguised requests when e3_prompts.py --extend has not been run)
    lines = []
    last = "novel_winter" if "novel_winter" in next(iter(REP["anchors"].values())) else "disguised"
    for sp in ["E", "PB", "L2", "TR"]:
        for vn, lab in [("U", "static"), ("N:1", "nearest anchor")]:
            cells = []
            for g in ["summer", "neutral", last]:
                cells += [f2(REP["anchors"][f"{sp}|{vn}|{M}"][g]) for M in REP["M_grid"]]
            lines.append(f"{SLAB[sp] if vn == 'U' else ''} & {lab} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(TAB, "e3_anchors.tex"), "w").write("\n".join(lines) + "\n")


def main():
    design_macros(); own_macros(); detection_macros(); repair_macros(); enforce_macros(); mechanism_macros(); unfamiliar()
    map_macros()
    fig_detection(); fig_per_request(); fig_enforce(); tables()
    fig_detection_map(); fig_anchors(); fig_enforce_map(); tables_manuscript()       # Section 11 of safe_prompt_fields.tex
    dec = C.load_decoder()
    fig_requests(dec)
    reqs = ENF["requests"]
    fig_gallery(dec, ids_of("neutral", among=reqs)[:3] + ids_of("approved", among=reqs)[:1], "fig_e3_gallery_neutral")
    fig_gallery(dec, ids_of("summer", among=reqs)[:2] + ids_of("disguised", among=reqs)[:2], "fig_e3_gallery_summer")
    with open(os.path.join(TAB, "e3_numbers.tex"), "w") as f:
        f.write("% written by code/e3_figures.py\n")
        for k, v in NUM.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(NUM)} macros")


if __name__ == "__main__":
    main()
