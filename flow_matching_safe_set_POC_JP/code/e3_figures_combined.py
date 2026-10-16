"""Figures, tables and numbers of Section 11.8 of safe_prompt_fields.tex (methods that decide with one distance and repair with
another), from ../results/e3/combined.json (e3_combined.py) and enforce_combined.json (e3_enforce_combined.py), with the regions of
e3_regions.py and e3_enforce.py as references.
  fig_e3c_enforce       per method: deployed images judged unsafe, and their scene similarity, previous and new methods
  fig_e3c_depth         per excluded request: mixing weight and P(winter) of the deployed images, threshold against deep back-off
  fig_e3c_gallery_*     what the new methods deploy for the requests of fig_e3_gallery_neutral and fig_e3_gallery_summer
Tables: ../tables/e3c_detection.tex, e3c_enforce.tex, e3c_unfamiliar.tex (row bodies), e3c_numbers.tex (macros).
Usage: python e3_figures_combined.py [--no-gallery]
"""
import os, sys, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import e1_figures as F
import e3_common as C
import e3_figures as G

TAB = G.TAB
COMB = json.load(open(os.path.join(C.RES, "combined.json")))
ENC = json.load(open(os.path.join(C.RES, "enforce_combined.json")))
LATC = np.load(os.path.join(C.RES, "enforce_combined_latents.npz"))
ENF, POOL, ROWS, REG = G.ENF, G.POOL, G.ROWS, G.REG
OLD = ["PB", "L2", "MH"]                                                  # previous regions (static weights) shown with the new ones
NEW = ["PB>F", "PB:deep", "PB>F:deep", "PB>F:local", "L2:deep", "J>F:deep", "MH>F", "PB:proj"]
LAB = {"PB": "source slice", "L2": "field $L^2$", "MH": "Mahalanobis", "E": "prompt",
       "PB>F": "slice$\\to$field", "PB:deep": "slice, deep", "PB>F:deep": "slice$\\to$field, deep", "L2:deep": "field, deep",
       "J>F:deep": "joint$\\to$field, deep", "MH>F": "Mahalanobis$\\to$field", "PB:proj": "slice, projection",
       "PB>F:local": "slice$\\to$field, local"}
CODE = {"PB": "S", "L2": "F", "MH": "M", "E": "P", "PB>F": "SF", "PB:deep": "Sd", "PB>F:deep": "SFd", "L2:deep": "Fd",
        "J>F:deep": "JFd", "MH>F": "MF", "PB:proj": "Sp", "PB>F:local": "SFl"}                   # macro codes (letters only)
DCOL = {"PB": F.AQUA, "L2": F.BLUE, "MH": F.VIOLET, "E": F.ORANGE, "J": F.INK2}  # colour: the distance that decides
KINDS = ["summer", "neutral", "disguised", "approved"]
UNF = ["paraphrase", "novel_winter", "novel_summer"]
NUM = {}


def mac(k, v):
    NUM[k] = v


f2 = G.f2


def decision(m):
    return ENC["methods"][m][0] if m in ENC["methods"] else m


def target_rule(m):
    return ENC["methods"][m][2] if m in ENC["methods"] else "own"


def stop(m):
    return ENC["methods"][m][1] if m in ENC["methods"] else "thr"


def rows_of(m, kind):
    """Records of method m for one kind of request: the previous regions come from e3_enforce.py on the original requests."""
    if m in OLD + ["E"] and kind in KINDS:
        return [dict(r, method=m) for r in ENF["rows"] if r["region"] == f"{m}|static" and r["kind"] == kind]
    return [r for r in ENC["rows"] if r["method"] == m and r["kind"] == kind]


def st(m, kind):
    rr = rows_of(m, kind)
    if not rr:
        return None
    alt = [r for r in rr if r["kappa"] > 0]
    unsafe = lambda r: np.mean(np.array(r["pwB"]) < 0.5)
    out = dict(n=len(rr), alt=np.mean([r["kappa"] > 0 for r in rr]), unsafe=np.mean([unsafe(r) for r in rr]),
               pw=np.mean([np.mean(r["pwB"]) for r in rr]), subj=np.mean([np.mean(r["subj"]) for r in rr]),
               kappa=np.mean([r["kappa"] for r in alt]) if alt else np.nan,
               same=np.mean([ROWS[r["target"]]["scene"] == ROWS[r["j"]]["scene"] for r in alt]) if alt else np.nan,
               unsafe_alt=np.mean([unsafe(r) for r in alt]) if alt else np.nan)
    if "chi_change" in rr[0]:
        out["change"] = np.mean([r["chi_change"] for r in rr])
        out["inL2"] = np.mean([r["sL2_dep"] <= 1 for r in alt]) if alt else np.nan
        out["inPB"] = np.mean([r["sPB_dep"] <= 1 + 1e-9 for r in alt]) if alt else np.nan
    return out


def own(kind, js=None):
    js = js if js is not None else sorted({r["j"] for r in ENC["rows"] if r["kind"] == kind})
    return dict(unsafe=np.mean(POOL["pwB_eval"][js] < 0.5), subj=np.mean(POOL["subj_eval"][js]), pw=np.mean(POOL["pwB_eval"][js]))


# ------------------------------------------------------------------ detection
def detection():
    e = COMB["eval"]
    rows = [("source slice", "PB"), ("field $L^2$", "L2"), ("joint (slice and field)", "joint"), ("Bonferroni (slice and field)", "bonferroni")]
    lines = []
    for lab, m in rows:
        cells = [f"{e[m]['test']:.3f}"] + [f2(e[m][g]) for g in ["summer", "neutral", "novel_summer", "paraphrase", "novel_winter", "disguised"]]
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(TAB, "e3c_detection.tex"), "w").write("\n".join(lines) + "\n")
    for m, c in [("PB", "S"), ("L2", "F"), ("MH", "M"), ("joint", "J"), ("bonferroni", "B")]:
        for g, gm in [("test", "Cov"), ("summer", "Sum"), ("neutral", "Neu"), ("disguised", "Dis"), ("paraphrase", "Par"),
                      ("novel_winter", "NovW"), ("novel_summer", "NovS"), ("judged_unsafe", "JU"), ("judged_safe", "JS")]:
            mac(f"ethcEx{gm}{c}", f"{e[m][g]:.3f}" if g == "test" else f2(e[m][g]))
        if m != "bonferroni":
            mac(f"ethcDeepCov{c}", f"{e[m]['deep_test']:.2f}")
        fa = (16 * e[m]["paraphrase"] + 12 * e[m]["novel_winter"]) / 28
        mac(f"ethcFA{c}", f2(fa))
    mac("ethcNomDeep", f"{COMB['nominal_deep']:.3f}"); mac("ethcBetaDeep", f"{COMB['beta_deep']:g}")
    mac("ethcAdmNeuS", f2(1 - e["PB"]["neutral"]))                      # requests without season that the source slice admits
    mac("ethcKdeep", int(round(COMB["nominal_deep"] * 41)))
    for g, gm in [("summer", "Sum"), ("neutral", "Neu"), ("novel_winter", "NovW"), ("paraphrase", "Par"), ("novel_summer", "NovS")]:
        v = COMB["trigger_PB"][g]
        mac(f"ethcTrig{gm}", "--" if v is None else f2(v))
    for n, c in [("PB", "S"), ("L2", "F")]:
        mac(f"ethcJointRatio{c}", f"{COMB['joint_threshold_ratio'][n]:.3f}")
    sl = sorted(COMB["scene_loss"].items(), key=lambda kv: kv[1]["centrality"])  # per scene, by centrality in the field distance
    least, most = sl[-4:], sl[:5]
    for n, c in [("PB", "S"), ("L2", "F"), ("joint", "J")]:
        mac(f"ethcLossSpear{c}", f2(COMB["scene_loss_spearman"][n]))
        mac(f"ethcLossLeastMin{c}", f2(min(v[n] for _, v in least))); mac(f"ethcLossLeastMax{c}", f2(max(v[n] for _, v in least)))
        mac(f"ethcLossMostMax{c}", f2(max(v[n] for _, v in most)))
        mac(f"ethcLossMax{c}", f2(max(v[n] for _, v in sl)))
    mac("ethcLossMaxSscene", max(sl, key=lambda kv: kv[1]["PB"])[0])
    mac("ethcLeastScenes", ", ".join(k for k, _ in least[::-1]))
    for n, c in [("PB", "S"), ("L2", "F"), ("MH", "M"), ("joint", "J")]:          # scenes without an admissible target
        for lv, lm in [("thr", "Thr"), ("deep", "Deep")]:
            mac(f"ethcNoTgt{c}{lm}", f2(COMB["no_target"][f"{n}|{lv}"]))


def tune_macros():
    """Prompt-kernel weights X:c (weights from the prompt distance, scores in another) on the tuning pool, against the weights chosen."""
    t = REG["tune"]
    mac("ethcTuneXF", f2(max(t[f"L2|None|X:{c}"]["J"] for c in [0.1, 0.3]))); mac("ethcTuneUF", f2(t["L2|None|U"]["J"]))
    mac("ethcTuneXS", f2(max(t[f"PB|None|X:{c}"]["J"] for c in [0.1, 0.3]))); mac("ethcTuneNS", f2(t["PB|None|N:1"]["J"]))


def has(m, kinds):
    return any(rows_of(m, k) for k in kinds)


# ------------------------------------------------------------------ enforcement
def enforce_table():
    """Rows: the previous regions, then the new methods; columns as in tab:e3_enforce."""
    lines = []
    o = {k: own(k, [j for j in ENC["requests"] if ROWS[j]["kind"] == k]) for k in KINDS}
    subj0 = np.mean([POOL["subj_eval"][j].mean() for j in ENC["requests"] if ROWS[j]["kind"] != "approved"])
    lines.append(f"no region & 0 & -- & {f2(o['summer']['unsafe'])} & 0 & -- & {f2(o['neutral']['unsafe'])} & "
                 f"{f2(o['disguised']['unsafe'])} & {subj0:.3f} & -- & 0 \\\\")
    lines.append("\\midrule")
    for block in [OLD, NEW]:
        for m in block:
            if not has(m, ["summer", "neutral"]):
                continue
            s = {k: st(m, k) for k in KINDS}
            cells = []
            for k in ["summer", "neutral"]:
                if s[k] is None:
                    cells += ["--"] * 3
                    continue
                cells += [f2(s[k]["alt"]), "--" if np.isnan(s[k]["kappa"]) else f2(s[k]["kappa"]), f2(s[k]["unsafe"])]
            rr = [r for k in ["summer", "neutral", "disguised"] for r in rows_of(m, k)]
            alt = [r for r in rr if r["kappa"] > 0]
            same = np.mean([ROWS[r["target"]]["scene"] == ROWS[r["j"]]["scene"] for r in alt]) \
                if alt and target_rule(m) != "proj" else np.nan                  # a projection has no single target
            cells += ["--" if s["disguised"] is None else f2(s["disguised"]["unsafe"]),
                      f"{np.mean([np.mean(r['subj']) for r in rr]):.3f}" + ("$^*$" if s["disguised"] is None else ""),
                      "--" if np.isnan(same) else f2(same), "--" if s["approved"] is None else f2(s["approved"]["alt"])]
            lines.append(f"{LAB[m]} & " + " & ".join(cells) + " \\\\")
        if block is OLD:
            lines.append("\\midrule")
    open(os.path.join(TAB, "e3c_enforce.tex"), "w").write("\n".join(lines) + "\n")


def unfamiliar_table():
    """Safe unfamiliar requests (paraphrases, novel winter scenes) and novel summer requests: how often each method alters them,
    how much it changes their images, and, for novel summer requests, how many deployed images remain unsafe."""
    lines = []
    for block in [["E"] + OLD, NEW[:-1]]:
        for m in block:
            if not all(rows_of(m, k) for k in UNF):
                continue
            s = {k: st(m, k) for k in UNF}
            cells = []
            for k in ["paraphrase", "novel_winter"]:
                cells += [f2(s[k]["alt"]), f2(s[k]["change"]), f2(s[k]["unsafe"])]
            k = "novel_summer"
            cells += [f2(s[k]["alt"]), f2(s[k]["unsafe"]), f"{s[k]['subj']:.3f}"]
            lines.append(f"{LAB[m]} & " + " & ".join(cells) + " \\\\")
        if block[0] == "E":
            lines.append("\\midrule")
    open(os.path.join(TAB, "e3c_unfamiliar.tex"), "w").write("\n".join(lines) + "\n")


def change_scale():
    """Reference values for the image change (CLIP ViT-L/14 distance between two images on the same evaluation noise): two
    approved prompts of the same scene, two of different scenes, and an approved prompt against the summer version of its scene."""
    X = POOL["chi_eval"].astype(np.float64)
    app = [r["id"] for r in ROWS if r["kind"] == "approved" and r["pool"] == "eval"]
    summ = [r["id"] for r in ROWS if r["kind"] == "summer" and r["pool"] == "eval"]
    sc = {j: ROWS[j]["scene"] for j in app + summ}
    d = lambda a, b: np.mean(np.linalg.norm(X[a] - X[b], axis=1))
    same = [d(a, b) for i, a in enumerate(app) for b in app[i + 1:] if sc[a] == sc[b]]
    other = [d(a, b) for i, a in enumerate(app) for b in app[i + 1:] if sc[a] != sc[b]]
    season = [d(a, b) for a in summ for b in app if sc[a] == sc[b]]
    mac("ethcChgSame", f2(np.median(same))); mac("ethcChgOther", f2(np.median(other))); mac("ethcChgSeason", f2(np.median(season)))
    return np.median(same), np.median(other), np.median(season)


def enforce_macros():
    for m in OLD + ["E"] + NEW:
        for k, km in [("summer", "Sum"), ("neutral", "Neu"), ("disguised", "Dis"), ("approved", "App"), ("paraphrase", "Par"),
                      ("novel_winter", "NovW"), ("novel_summer", "NovS")]:
            s = st(m, k)
            if s is None:
                continue
            t = CODE[m] + km
            mac(f"ethcAlt{t}", f2(s["alt"])); mac(f"ethcAfter{t}", f2(s["unsafe"])); mac(f"ethcSubj{t}", f"{s['subj']:.3f}")
            mac(f"ethcKap{t}", "--" if np.isnan(s["kappa"]) else f2(s["kappa"]))
            mac(f"ethcSame{t}", "--" if np.isnan(s["same"]) else f2(s["same"]))
            if "change" in s:
                mac(f"ethcChg{t}", f2(s["change"]))
        rr = [r for k in ["summer", "neutral", "disguised"] for r in rows_of(m, k)]
        alt = [r for r in rr if r["kappa"] > 0]
        if alt:
            mac(f"ethcSame{CODE[m]}", f2(np.mean([ROWS[r["target"]]["scene"] == ROWS[r["j"]]["scene"] for r in alt])))
            mac(f"ethcSubj{CODE[m]}", f"{np.mean([np.mean(r['subj']) for r in rr]):.3f}")
            mac(f"ethcKap{CODE[m]}", f2(np.mean([r["kappa"] for r in alt])))
            if "sL2_dep" in alt[0]:
                mac(f"ethcInF{CODE[m]}", f2(np.mean([r["sL2_dep"] <= 1 for r in alt])))
                mac(f"ethcInS{CODE[m]}", f2(np.mean([r["sPB_dep"] <= 1 + 1e-9 for r in alt])))
                mac(f"ethcCertMin{CODE[m]}", f2(min(r["cert"] for r in alt)))
                mac(f"ethcCertMed{CODE[m]}", f2(np.median([r["cert"] for r in alt])))
            # unsafe images of the excluded requests by mixing weight, as in Section 11.6
            for nm, sel in [("Small", [r for r in alt if r["kappa"] < 0.5]), ("Large", [r for r in alt if r["kappa"] >= 0.5])]:
                mac(f"ethcN{nm}{CODE[m]}", len(sel))
                mac(f"ethcRes{nm}{CODE[m]}", f2(np.mean([np.mean(np.array(r["pwB"]) < 0.5) for r in sel])) if sel else "--")
    for k, km in [("summer", "Sum"), ("neutral", "Neu"), ("disguised", "Dis"), ("approved", "App")] + \
                 [(k, {"paraphrase": "Par", "novel_winter": "NovW", "novel_summer": "NovS"}[k]) for k in UNF]:
        o = own(k)
        mac(f"ethcOwnUnsafe{km}", f2(o["unsafe"])); mac(f"ethcOwnSubj{km}", f"{o['subj']:.3f}")
    cert = [r["cert"] for r in ENC["rows"] if r["method"] in NEW and r["kappa"] > 0 and "cert" in r]
    if cert:
        mac("ethcCertMinAll", f2(min(cert))); mac("ethcCertMedAll", f2(np.median(cert)))
    proj = [r for r in ENC["rows"] if r["method"] == "PB:proj" and "gamma" in r]
    if proj:
        g0 = [r["gamma"][0] for r in proj]; nf = [r["n_fields"] for r in proj]
        top = [np.max(r["gamma"][1:]) for r in proj]
        mac("ethcProjGzero", f2(np.median(g0))); mac("ethcProjNfields", int(np.median(nf))); mac("ethcProjTop", f2(np.median(top)))
        mac("ethcNproj", len(proj))
    mac("ethcNunf", len(ENC["unfamiliar"]) // 3)


def depth_macros():
    """The median level relative to the threshold, per deciding score (split 0), and how far above the threshold the requests are."""
    for m in ["PB>F:deep", "L2:deep", "J>F:deep"]:
        rr = [r for r in ENC["rows"] if r["method"] == m]
        if rr:
            mac(f"ethcDeepRatio{CODE[m][:-1]}", f2(rr[0]["level"] / rr[0]["eps"]))
    for k, km in [("summer", "Sum"), ("neutral", "Neu")]:
        for m in ["PB>F", "L2:deep", "MH>F"]:
            rr = [r for r in rows_of(m, k) if r["kappa"] > 0]
            if rr:
                mac(f"ethcSzero{km}{CODE[m][0]}", f2(np.median([r["s0"] for r in rr])))
                mac(f"ethcSzeroMin{km}{CODE[m][0]}", f2(min(r["s0"] for r in rr)))
                mac(f"ethcSzeroMax{km}{CODE[m][0]}", f2(max(r["s0"] for r in rr)))


def slice_macros():
    """Section 11.9. (a) How much of the estimated source-slice distance depends on the probe noise: with D_l the difference of two
    prompts' slices on probe l and Dbar its mean over the probes, d0^2 = |Dbar|^2 + (1/P) sum_l |D_l - Dbar|^2, and the second term
    is the only part that the exact-field formula d0 = omega |m(c) - m(c')| does not predict. (b) Seen from the anchors, the angle
    between the request and its target in the probe embedding y(c): mean over the anchors of cos(y(c) - y(c_i), y(c') - y(c_i)), for
    the source slice's own targets and for targets chosen in the field distance (altered summer and season-not-mentioned requests)."""
    import e3_regions as R
    _, D, idx = R.load()
    anc = np.array(ENC["anchors"])
    S = D["slice"].astype(np.float64); P = S.shape[1]

    def leak(a, b):
        d = S[a] - S[b]; dbar = d.mean(0)
        return ((d - dbar) ** 2).sum(1).mean() / (d ** 2).sum(1).mean()
    app = [j for j in idx["approved_eval"] if j not in set(anc)]
    for kind, km, js in [("approved", "App", app), ("summer", "Sum", idx["summer_eval"]), ("neutral", "Neu", idx["neutral_eval"])]:
        sh = [leak(j, i) for j in js for i in anc if j != i]
        mac(f"ethcLeak{km}", f2(np.median(sh))); mac(f"ethcLeakQ{km}", f2(np.quantile(sh, 0.9)))
    y = S.reshape(len(S), -1) / np.sqrt(P)

    def cosmean(j, t):
        keep = anc[anc != t]
        a, b = y[j] - y[keep], y[t] - y[keep]
        return np.mean((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1))
    for m in ["PB", "PB>F"]:
        rr = [r for k in ["summer", "neutral"] for r in rows_of(m, k) if r["kappa"] > 0]
        mac(f"ethcCos{CODE[m]}", f2(np.median([cosmean(r["j"], r["target"]) for r in rr])))
    mac("ethcPn", P * S.shape[2]); mac("ethcLatentDim", S.shape[2])


def city_macros():
    """The city park request without season: replaced by another scene at the median level of the source slice, kept with local depth."""
    js = [j for j in ENC["requests"] if ROWS[j]["kind"] == "neutral" and "city park" in ROWS[j]["subject_text"]]
    if not js:
        return
    for m in ["PB>F:deep", "PB>F:local"]:
        r = [r for r in rows_of(m, "neutral") if r["j"] == js[0]]
        if r:
            mac(f"ethcCityKap{CODE[m]}", f2(r[0]["kappa"])); mac(f"ethcCityPw{CODE[m]}", f2(np.mean(r[0]["pwB"])))


def target_macros():
    """Where the targets lie in the source-slice distance: their own score and their distance to the request, relative to the
    threshold, for the source slice's own targets and for targets chosen in the field distance (summer and season-not-mentioned
    requests that the method alters)."""
    import e3_regions as R
    _, D, idx = R.load()
    anc = np.array(ENC["anchors"]); _, cal, _ = R.split(idx["approved_eval"], 1000)
    sp = R.Space("PB", D, idx); Fx = sp.feats(D)
    score = lambda js: sp.dist(Fx[js], Fx[anc]).mean(1)
    eps = C.kth_threshold(score(cal), R.BETA)
    for m in ["PB", "PB>F", "PB>F:deep", "PB>F:local"]:
        rr = [r for k in ["summer", "neutral"] for r in rows_of(m, k) if r["kappa"] > 0]
        if not rr:
            continue
        t = np.array([r["target"] for r in rr]); j = np.array([r["j"] for r in rr])
        mac(f"ethcTgtScore{CODE[m]}", f2(np.median(score(t) / eps)))
        mac(f"ethcTgtDist{CODE[m]}", f2(np.median([sp.dist(Fx[[a]], Fx[[b]])[0, 0] for a, b in zip(j, t)]) / eps))
        mac(f"ethcKapSN{CODE[m]}", f2(np.mean([r["kappa"] for r in rr])))
        # the static score is convex along the segment: kappa against its linear prediction (s0 - eps) / (s0 - s_target)
        s0, stg = score(j) / eps, score(t) / eps
        klin = (s0 - 1) / (s0 - stg)
        mac(f"ethcKlin{CODE[m]}", f2(np.mean(klin)))
        mac(f"ethcKratio{CODE[m]}", f2(np.median(np.array([r["kappa"] for r in rr]) / klin)))


# ------------------------------------------------------------------ figures
ROW_ORDER = [("previous", ["PB", "L2", "MH"]),
             ("new", ["PB>F", "PB:deep", "PB>F:deep", "PB>F:local", "L2:deep", "J>F:deep", "MH>F", "PB:proj"])]


def fig_enforce_dots():
    """One row per method: (a) fraction of deployed images judged unsafe, (b) their scene similarity; filled circle: summer
    requests, filled triangle: requests without season; colour: the distance that decides. First row: no region."""
    rows = [("no region", None)] + [(LAB[m], m) for _, ms in ROW_ORDER for m in ms if has(m, ["summer"])]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 0.26 * len(rows) + 0.95), sharey=True)
    y = np.arange(len(rows))[::-1]
    for (lab, m), yy in zip(rows, y):
        for kind, mk in [("summer", "o"), ("neutral", "^")]:
            if m is None:
                o = own(kind, [j for j in ENC["requests"] if ROWS[j]["kind"] == kind]); vals = (o["unsafe"], o["subj"]); col = F.INK
            else:
                st_ = st(m, kind)
                if st_ is None:
                    continue
                vals = (st_["unsafe"], st_["subj"]); col = DCOL[decision(m)]
            for a, v in enumerate(vals):
                axs[a].plot([v], [yy + (0.13 if kind == "summer" else -0.13)], marker=mk, ms=5.2, ls="", color=col,
                            mec="white", mew=0.5, zorder=4)
    n_prev = 1 + len([m for m in ROW_ORDER[0][1] if has(m, ["summer"])])
    for ax in axs:
        ax.axhline(y[n_prev - 1] - 0.5, color=F.AXIS, lw=0.8)
        ax.axhline(y[0] - 0.5, color=F.AXIS, lw=0.8)
        ax.grid(True, axis="x"); ax.set_ylim(-0.6, len(rows) - 0.4)
    axs[0].set_yticks(y); axs[0].set_yticklabels([lab for lab, _ in rows], fontsize=6.3)
    axs[0].set_xlim(-0.03, 1.03); axs[0].set_xlabel("deployed images judged unsafe (lower is better)", fontsize=6.6)
    o_s = own("summer", [j for j in ENC["requests"] if ROWS[j]["kind"] == "summer"])["subj"]
    o_n = own("neutral", [j for j in ENC["requests"] if ROWS[j]["kind"] == "neutral"])["subj"]
    axs[1].axvspan(min(o_s, o_n), max(o_s, o_n), color=F.GRID, alpha=0.9, zorder=1)
    axs[1].axvline(G.NUM.get("ethSubjOtherMH") and float(G.NUM["ethSubjOtherMH"]) or 0.19, color=F.MUTED, lw=0.7, ls=(0, (2, 2)))
    axs[1].set_xlim(0.14, 0.345); axs[1].set_xlabel("scene similarity of the deployed images (higher is better)", fontsize=6.6)
    axs[1].text(0.192, y[-1] - 0.45, "another\nscene", fontsize=5.2, color=F.MUTED, va="bottom", ha="left")
    axs[1].text(max(o_s, o_n) + 0.002, y[n_prev] + 0.5, "requests'\nown images", fontsize=5.2,
                color=F.MUTED, va="center", ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    axs[0].set_title("(a) safety", fontsize=7, loc="left"); axs[1].set_title("(b) faithfulness to the requested scene", fontsize=7, loc="left")
    axs[0].text(-0.62, y[n_prev - 1] + 0.2, "previous", transform=axs[0].get_yaxis_transform(), fontsize=5.6, color=F.MUTED, rotation=90, va="center")
    axs[0].text(-0.62, np.mean(y[n_prev:]), "new", transform=axs[0].get_yaxis_transform(), fontsize=5.6, color=F.MUTED, rotation=90, va="center")
    hs = [Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=F.INK, mew=0.9, label="summer requests"),
          Line2D([], [], marker="^", ls="", ms=5, mfc="white", mec=F.INK, mew=0.9, label="season not mentioned")]
    hs += [Line2D([], [], marker="s", ls="", ms=5, color=DCOL[d], label=f"decides: {lab}")
           for d, lab in [("PB", "source slice"), ("L2", "field"), ("MH", "Mahalanobis"), ("J", "joint")]]
    fig.legend(handles=hs, loc="upper center", ncol=6, fontsize=5.8, frameon=False, bbox_to_anchor=(0.55, 1.0), handletextpad=0.2,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.0)
    F.save(fig, "fig_e3c_enforce")


def fig_depth():
    """Each request excluded by the deciding region (summer or season not mentioned): its mixing weight kappa and the mean
    P(winter) of its deployed images, with the back-off stopped at the threshold (hollow) and at the median level (filled), joined
    by a line. (a) source slice decides, target in the field distance; (b) field region. The decision is the same in both cases."""
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.45), sharey=True)
    for a, (ax, (m0, m1), title) in enumerate(zip(axs, [("PB>F", "PB>F:deep"), ("PB>F", "PB>F:local"), ("L2", "L2:deep")],
                                                    ["slice$\\to$field: median level", "slice$\\to$field: local depth",
                                                     "field region: median level"])):
        col = DCOL[decision(m1)]
        for kind, mk in [("summer", "o"), ("neutral", "^")]:
            r0 = {r["j"]: r for r in rows_of(m0, kind)}
            r1 = {r["j"]: r for r in rows_of(m1, kind)}
            for j in sorted(set(r0) & set(r1)):
                if r0[j]["kappa"] == 0 and r1[j]["kappa"] == 0:
                    continue
                xs = [r0[j]["kappa"], r1[j]["kappa"]]
                ys = [np.mean(r0[j]["pwB"]), np.mean(r1[j]["pwB"])]
                ax.plot(xs, ys, color=F.AXIS, lw=0.7, zorder=2)
                ax.scatter(xs[0], ys[0], marker=mk, s=22, facecolors="white", edgecolors=col, linewidths=0.9, zorder=3)
                ax.scatter(xs[1], ys[1], marker=mk, s=22, facecolors=col, edgecolors=col, linewidths=0.9, zorder=4)
        ax.axhline(0.5, color=F.CRIT, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.text(1.02, 0.47, "unsafe\nbelow", color=F.CRIT, fontsize=5.4, va="top", ha="right")
        ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 1.03); ax.grid(True)
        ax.set_xlabel("mixing weight $\\kappa$ of the target", fontsize=6.6)
        ax.set_title(f"({'abc'[a]}) {title}", fontsize=7, loc="left")
    axs[0].set_ylabel("deployed images: mean P(winter)", fontsize=6.6)
    hs = [Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=F.INK2, label="summer request"),
          Line2D([], [], marker="^", ls="", ms=4.5, mfc="white", mec=F.INK2, label="season not mentioned"),
          Line2D([], [], marker="s", ls="", ms=4.5, mfc="white", mec=F.INK2, mew=1.0, label="hollow: stops at the threshold"),
          Line2D([], [], marker="s", ls="", ms=4.5, mfc=F.INK2, mec=F.INK2, label="filled: deeper stop (median or local)")]
    fig.legend(handles=hs, loc="upper center", ncol=4, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.0)
    F.save(fig, "fig_e3c_depth")


GALLERY = ["PB", "L2", "PB>F", "PB>F:deep", "PB>F:local", "L2:deep", "J>F:deep", "MH>F"]
GLAB = {"PB": "source slice\n(previous)", "L2": "field $L^2$\n(previous)", "PB>F": "slice$\\to$field", "PB>F:deep": "slice$\\to$field\ndeep",
        "L2:deep": "field\ndeep", "J>F:deep": "joint$\\to$field\ndeep", "MH>F": "Mahalanobis\n$\\to$field",
        "PB>F:local": "slice$\\to$field\nlocal"}


def lat_of(m, j):
    if m in OLD and f"{m}|static|{j}" in G.LAT.files:
        return G.LAT[f"{m}|static|{j}"]
    return LATC[f"{m}|{j}"] if f"{m}|{j}" in LATC.files else np.zeros((C.P_EVAL, *C.LATENT), np.float16)


def rec_of(m, j):
    rr = [r for r in (ENF["rows"] if m in OLD else ENC["rows"])
          if (r.get("region") == f"{m}|static" if m in OLD else r["method"] == m) and r["j"] == j]
    return rr[0] if rr else None


def fig_gallery(dec, picks, name, noise=0):
    cols = ["own"] + GALLERY
    row_h = 6.6 * (0.998 - 0.165) / len(cols) * 1.02                      # square thumbnails: the row height follows the width
    fig, axs = plt.subplots(len(picks), len(cols), figsize=(6.6, row_h * len(picks) / 0.926 + 0.05))
    for a, j in enumerate(picks):
        lats = [POOL["lat_eval"][j][noise:noise + 1]] + [lat_of(m, j)[noise:noise + 1] for m in GALLERY]
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
                lab = f"{pw:.2f}" + ("" if r["kappa"] == 0 else f" $\\kappa$ {r['kappa']:.2f}")
            G.tag_box(ax, lab, pw < 0.5, fs=4.6)
            if a == 0:
                ax.set_title("request,\nown field" if m == "own" else GLAB[m], fontsize=5.3, pad=2)
        p = ROWS[j]["prompt"][len("a photograph of "):]
        axs[a, 0].set_ylabel(f"{G.KLAB[ROWS[j]['kind']]}:\n" + p.replace(", ", ",\n"), fontsize=4.2, rotation=0, ha="right",
                             va="center", labelpad=2)
    fig.subplots_adjust(left=0.165, right=0.998, top=0.93, bottom=0.004, wspace=0.025, hspace=0.04)
    F.save(fig, name)


def main():
    detection(); tune_macros(); enforce_table(); unfamiliar_table(); change_scale(); enforce_macros(); target_macros(); depth_macros(); city_macros(); slice_macros()
    fig_enforce_dots(); fig_depth()
    if "--no-gallery" not in sys.argv:
        dec = C.load_decoder()
        reqs = ENF["requests"]
        fig_gallery(dec, G.ids_of("neutral", among=reqs)[:3] + G.ids_of("approved", among=reqs)[:1], "fig_e3c_gallery_neutral")
        fig_gallery(dec, G.ids_of("summer", among=reqs)[:2] + G.ids_of("disguised", among=reqs)[:2], "fig_e3c_gallery_summer")
    with open(os.path.join(TAB, "e3c_numbers.tex"), "w") as f:
        f.write("% written by code/e3_figures_combined.py\n")
        for k, v in NUM.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(NUM)} macros")


if __name__ == "__main__":
    main()
