"""Table bodies and number macros of the E1 section of safe_prompt_fields.tex, written from ../results/e1/ only.
Writes ../tables/e1_detection.tex, ../tables/e1_certificates.tex and ../tables/e1_numbers.tex.
Usage: python e1_tables.py"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES, TAB = os.path.join(HERE, "..", "results", "e1"), os.path.join(HERE, "..", "tables")
D = np.load(os.path.join(RES, "fit.npz"))
P = json.load(open(os.path.join(RES, "phase2.json")))
SCR = json.load(open(os.path.join(RES, "improve_screen.json")))
PH1 = json.load(open(os.path.join(RES, "eval_splits.json")))
ENF1 = json.load(open(os.path.join(RES, "eval_enforce.json")))
NS = int(D["n_safe"]); Z = D["Z"]
LABEL = {"U-L2": "uniform, $L^2$ map metric (as written)", "kNN1-L1": "nearest safe field, $L^1$",
         "kNN1-tr1": "nearest safe field, truncated $L^1$", "kNN1-tl.5": "nearest safe field, tail score",
         "E-U": "uniform, prompt embeddings", "E-kNN1": "nearest safe field, prompt embeddings"}
ORDER = list(LABEL)
CFGS = [("rect1", "1-rectified model, 50 Euler steps"), ("rect2", "2-rectified model, 50 Euler steps")]
macros = {}


def mac(name, val):
    macros[name] = val


def blocks(cfg, me):
    return [b for b in P["enforce"] if b["cfg"] == cfg and b["method"] == me]


def rows_of(cfg, me):
    return [r for b in blocks(cfg, me) for r in b["rows"]]


def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def pct(x, d=1): return f"{100 * x:.{d}f}\\%"


# ------------------------------------------------------------------ table: detection and enforcement
lines = []
for cfg, title in CFGS:
    lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{title}}}}}\\\\")
    base = rows_of(cfg, "U-L2")
    pa0 = {m: np.mean([r["pA0"] for r in base if r["m"] == m]) for m in [0.1, 0.3]}
    for me in ORDER:
        B, R = blocks(cfg, me), rows_of(cfg, me)
        saf = [r for r in R if r["safe"]]
        cells = [f3(np.mean([b["coverage"] for b in B])), f"{np.median([b['eps'] for b in B]):.3f}"]
        cells += [f2(np.mean([b[f"excl_{m}"] for b in B])) for m in [0.03, 0.1]]
        cells += [f"{np.mean([r['pA'] for r in R if r['m'] == m]):.3f}" for m in [0.1, 0.3]]
        cells += [f"{np.mean([r['mmd'] for r in R if r['m'] == 0.3]):.0f}", f2(np.mean([r["kappa"] > 0 for r in saf]))]
        lines.append(f"{LABEL[me]} & " + " & ".join(cells) + " \\\\")
    e1 = [b for b in ENF1 if b["cfg"] == cfg][0]["rows"]                 # the unconstrained field does not depend on the split
    mmd_none = np.mean([r["none.mmd"] for r in e1 if r["m"] == 0.3])
    lines.append(f"no enforcement & -- & -- & -- & -- & {pa0[0.1]:.3f} & {pa0[0.3]:.3f} & {mmd_none:.0f} & -- \\\\")
    if cfg == "rect1":
        lines.append("\\midrule")
open(os.path.join(TAB, "e1_detection.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ table: certificates and attack
R_A, RG = 0.6, np.linspace(0.05, 2.0, 40)
import sys
sys.path.insert(0, HERE)


def region_bound(cfg, me, eps, anc):
    Te = D[f"{cfg}_Te"][anc].astype(float)
    q = np.array([[np.mean(np.linalg.norm(Te[i], axis=1) < R_A + r) for r in RG] for i in range(len(anc))])
    if me == "U-L2":
        return float(np.min(q.mean(0) + eps / RG))
    per = {"kNN1-L1": q + eps / RG, "kNN1-tr1": q + eps * np.maximum(1, 1.0 / RG),
           "kNN1-tl.5": np.where(RG >= 0.5, q + eps, np.inf)}[me]
    return float(per.min(1).max())


perm = np.random.default_rng(1000).permutation(NS); anc0 = perm[:30]     # split 0 of e1_eval.split
ROB = json.load(open(os.path.join(RES, "robustness.json")))              # e1_robustness.py: probe sensitivity, directed search
VER = json.load(open(os.path.join(RES, "verify.json")))                  # e1_verify.py: verified deployment


def directed(cfg, me):
    v = [r["pA_fresh"] for r in ROB["attack2"] if r["cfg"] == cfg and r["method"] == me]
    return max(v) if v else None


lines = []
for cfg, title in CFGS:
    lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{title}}}}}\\\\")
    for me in ORDER:
        R = rows_of(cfg, me)
        A = [a for a in P["attack"] if a["cfg"] == cfg and a["method"] == me][0]
        worst = max(r["pA_fresh"] for r in A["runs"])
        if me.startswith("E-"):
            lines.append(f"{LABEL[me]} & -- & -- & -- & -- & {worst:.3f} & -- & -- \\\\"); continue
        bd = np.array([r["bound"] for r in R]); pa = np.array([r["pA"] for r in R])
        fr = np.array([r["fresh"] / r["budget"] for r in R]); te = np.array([r["tail_emp"] for r in R])
        lines.append(f"{LABEL[me]} & {np.median(bd):.3f} & {np.median(pa):.3f} & {pct(np.mean(fr > 1))} & "
                     f"{np.median(te):.3f} & {worst:.3f} & {directed(cfg, me):.3f} & {region_bound(cfg, me, A['eps'], anc0):.3f} \\\\")
    if cfg == "rect1":
        lines.append("\\midrule")
open(os.path.join(TAB, "e1_certificates.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ table: probe sensitivity (e1_robustness.py)
PR = ROB["probe"]; PROBE = PR["flagged"]["rect1"][0]
assert PR["flagged"]["rect1"] == PR["flagged"]["rect2"] == [PROBE]
PNAME = {"L2|U": "uniform, $L^2$ (as written)", "L4|U": "uniform, $L^4$", "L8|U": "uniform, $L^8$", "L2|kNN1": "nearest, $L^2$",
         "L4|kNN1": "nearest, $L^4$", "L1|kNN1": "nearest, $L^1$", "trunc1|kNN1": "nearest, truncated $L^1$",
         "tail0.5|kNN1": "nearest, tail score"}
ARW = lambda a, b, f="{:.2f}": f"{f.format(a)}$\\to${f.format(b)}"
lines = []
for cfg, title, keys in [("rect1", "1-rectified model, 50 Euler steps", list(PNAME)),
                         ("rect2", "2-rectified model, 50 Euler steps", ["L4|U", "L8|U", "L2|kNN1", "trunc1|kNN1"])]:
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}}\\\\")
    for key in keys:
        a, d = PR["screen"][cfg]["all"][key], PR["screen"][cfg]["drop"][key]
        lines.append(f"{PNAME[key]} & " + " & ".join(ARW(a[f'excl_{m}'], d[f'excl_{m}']) for m in [0.01, 0.03, 0.1, 0.3]) +
                     f" & {ARW(a['eps_median'], d['eps_median'], '{:.3f}')} \\\\")
    if cfg == "rect1":
        lines.append("\\midrule")
open(os.path.join(TAB, "e1_probe.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ table: verified deployment (e1_verify.py)
VLAB = {"kNN1-tr1": "nearest, truncated $L^1$", "kNN1-tl.5": "nearest, tail score"}
lines = []
for cfg, title in CFGS:
    lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{title}}}}}\\\\")
    for me in ["kNN1-tr1", "kNN1-tl.5"]:
        B = [b for b in VER["runs"] if b["cfg"] == cfg and b["method"] == me]; R = [r for b in B for r in b["rows"]]
        saf, u3 = [r for r in R if r["safe"]], [r for r in R if r["m"] == 0.3]
        over = np.mean([r["d_big"] > b["b"] for b in B for r in b["rows"]])
        lines.append(f"{VLAB[me]} & {np.median([b['b'] for b in B]):.3f} & {np.median([b['bprime'] for b in B]):.3f} & "
                     f"{np.mean([r['stage'] != 'as is' for r in saf]):.2f} & {np.mean([r['stage'] == 'fallback' for r in u3]):.2f} & "
                     f"{np.mean([r['pA'] for r in u3]):.3f} & {np.mean([r['mmd'] for r in u3]):.0f} & {pct(over)} & "
                     f"{np.median([r['bound'] for r in R]):.3f} \\\\")
    if cfg == "rect1":
        lines.append("\\midrule")
open(os.path.join(TAB, "e1_verify.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ numbers quoted in the prose
for cfg, tag in [("rect1", "A"), ("rect2", "B")]:
    mac(f"eoneSafePA{tag}", f"{D[f'{cfg}_pA'][:NS].mean():.4f}")
    mac(f"eoneZeta{tag}", f"{np.median(D[f'{cfg}_zeta'][:NS]):.2f}")
    for m, mt in [(0.01, "a"), (0.03, "b"), (0.1, "c"), (0.3, "d")]:
        mac(f"eonePA{tag}{mt}", f"{D[f'{cfg}_pA'][Z[:, 2] == m].mean():.3f}")
mac("eoneTruePAd", f"{D['true_pA'][Z[:, 2] == 0.3].mean():.3f}")
mac("eoneOneStepPA", f"{D['rect2_1step_pA'][:NS].mean():.4f}")
# phase 1, uniform weights, over 100 splits
for cfg, tag in [("rect1", "A"), ("rect2", "B")]:
    for met, mt in [("T", "T"), ("0", "Z"), ("rho", "R"), ("E", "E")]:
        R = [r for r in PH1 if r["cfg"] == cfg and r["metric"] == met]
        mac(f"eoneCov{tag}{mt}", f3(np.mean([r["coverage"] for r in R])))
        for m, mt2 in [(0.03, "b"), (0.1, "c"), (0.3, "d")]:
            mac(f"eoneEx{tag}{mt}{mt2}", f2(np.mean([r[f"excl_{m}"] for r in R])))
        mac(f"eoneBudget{tag}{mt}", f"{np.median([31 * r['eps'] for r in R]):.2f}")
# screen: L^p and tail/trunc variants with kNN1 and uniform weights (rect1)
def scr(cfg, p, kind, par, key):
    R = [r for r in SCR if r["cfg"] == cfg and r["p"] == p and r["kind"] == kind and r["par"] == par]
    return np.mean([r[key] for r in R]) if key != "eps" else np.median([r["eps"] for r in R])
mac("eoneLfourUd", f2(scr("rect1", 4, "U", None, "excl_0.3")))
mac("eoneLeightUd", f2(scr("rect1", 8, "U", None, "excl_0.3")))
mac("eoneKnnLtwoc", f2(scr("rect1", 2, "kNN", 1, "excl_0.1")))
mac("eoneKnnLonec", f2(scr("rect1", 1, "kNN", 1, "excl_0.1")))
mac("eoneKnnLfourc", f2(scr("rect1", 4, "kNN", 1, "excl_0.1")))
mac("eoneKnnTruncc", f2(scr("rect1", "trunc1.0", "kNN", 1, "excl_0.1")))
mac("eoneKnnTailb", f2(scr("rect1", "tail0.5", "kNN", 1, "excl_0.03")))
mac("eoneKnnTailCov", f3(scr("rect1", "tail0.5", "kNN", 1, "coverage")))
# phase-1 certificate on fresh noise (rect1): fraction of region fields whose fresh coupling cost exceeds the budget
ins = [(r, me) for r in ENF1[0]["rows"] for me in ["none", "backoff_T", "proj_0", "proj_rho", "proj_E"] if r[me + ".sT"] <= 1 + 1e-9]
ratio = np.array([r[me + ".coupling"] / r[me + ".budgetT"] for r, me in ins])
mac("eonePlugFail", pct(np.mean(ratio > 1)))
mac("eonePlugMax", f"{ratio.max():.1f}")
mac("eoneMarkovMed", f"{np.median([r[me + '.bound_markov_T'] for r, me in ins]):.2f}")
mac("eoneRegionWorst", f"{max(r[me + '.pA'] for r, me in ins):.3f}")
# one-step map of the 1-rectified model: fraction of probe outputs inside A (safe prompts)
mac("eoneOneStepRectOne", f"{np.mean(np.linalg.norm(D['rect1_T1p'][:NS].astype(float), axis=-1) <= 0.6):.2f}")
# fidelity of the three model/sampler configurations (safe prompts, every 10th)
from e1_common import mmd2
for cfg, tag in [("rect1", "A"), ("rect2", "B"), ("rect2_1step", "C")]:
    Te = D[f"{cfg}_Te"]; ts = D["true_safe_samples"]
    mac(f"eoneMmd{tag}", f"{np.mean([mmd2(Te[i, :1000].astype(float), ts[i].astype(float)) for i in range(0, NS, 10)]) * 1e3:.1f}")
# phase 2 (means over the enforcement splits)
def pa_after(cfg, me, m): return np.mean([r["pA"] for r in rows_of(cfg, me) if r["m"] == m])
def mmd_after(cfg, me, m): return np.mean([r["mmd"] for r in rows_of(cfg, me) if r["m"] == m])
def attack_max(cfg, me): return max(r["pA_fresh"] for r in [a for a in P["attack"] if a["cfg"] == cfg and a["method"] == me][0]["runs"])
def fresh_ratio(cfg, me): return np.array([r["fresh"] / r["budget"] for r in rows_of(cfg, me)])
mac("eoneTwoPAasWritten", f"{pa_after('rect1', 'U-L2', 0.3):.3f}")
mac("eoneTwoPAtrunc", f"{pa_after('rect1', 'kNN1-tr1', 0.3):.3f}")
mac("eoneTwoMmdAsWritten", f"{mmd_after('rect1', 'U-L2', 0.3):.0f}")
mac("eoneTwoMmdTrunc", f"{mmd_after('rect1', 'kNN1-tr1', 0.3):.0f}")
mac("eoneTwoMmdNone", f"{np.mean([r['none.mmd'] for r in [b for b in ENF1 if b['cfg'] == 'rect1'][0]['rows'] if r['m'] == 0.3]):.0f}")
mac("eoneTwoBudgetUL", f"{np.median([b['eps'] for b in blocks('rect1', 'U-L2')]):.2f}")
mac("eoneTwoBudgetTrunc", f"{np.median([b['eps'] for b in blocks('rect1', 'kNN1-tr1')]):.2f}")
for me, tag in [("U-L2", "UL"), ("kNN1-L1", "Lone"), ("kNN1-tr1", "Trunc"), ("kNN1-tl.5", "Tail"), ("E-U", "EU"), ("E-kNN1", "Eknn")]:
    mac(f"eoneTwoAttack{tag}", f"{max(attack_max('rect1', me), attack_max('rect2', me)):.3f}")
fr_all = {me: np.r_[fresh_ratio("rect1", me), fresh_ratio("rect2", me)] for me in ["kNN1-L1", "kNN1-tr1", "kNN1-tl.5"]}
mac("eoneTwoPlugMin", pct(min(np.mean(fresh_ratio(c, me) > 1) for c in ["rect1", "rect2"] for me in fr_all), 0))
mac("eoneTwoPlugMax", pct(max(np.mean(fresh_ratio(c, me) > 1) for c in ["rect1", "rect2"] for me in fr_all), 0))
mac("eoneTwoTruncMaxRatio", f"{fr_all['kNN1-tr1'].max():.2f}")
mac("eoneTwoLoneMaxRatio", f"{fr_all['kNN1-L1'].max():.1f}")
mac("eoneTwoCertTail", f"{np.median([r['bound'] for r in rows_of('rect1', 'kNN1-tl.5')]):.3f}")
mac("eoneTwoCertTrunc", f"{np.median([r['bound'] for r in rows_of('rect1', 'kNN1-tr1')]):.2f}")
# probe sensitivity and the sampler's divergence (e1_robustness.py)
SC = PR["screen"]
mac("eoneProbe", str(PROBE)); mac("eoneProbeNorm", f"{PR['x0_norm']['rect1'][0]:.2f}")
mac("eoneProbeMaxT", f"{PR['maxnorm_probe']['rect1']:.1f}")
ev = PR["maxnorm_eval"]["rect1"]; e10 = int(np.floor(np.log10(ev)))
mac("eoneEvalMaxT", f"{ev / 10 ** e10:.1f}\\times10^{{{e10}}}")
blow = np.linalg.norm(D["rect1_Tp"][:, PROBE].astype(float), axis=-1) > 4.0
mac("eoneBlowSafe", f"{100 * blow[:NS].mean():.0f}"); mac("eoneBlowUns", f"{100 * blow[Z[:, 2] == 0.3].mean():.0f}")
mac("eoneShareMed", f"{100 * PR['share_L2']['rect1']['median']:.0f}"); mac("eoneShareMean", f"{100 * PR['share_L2']['rect1']['mean']:.0f}")
mac("eoneShareNinety", f"{100 * PR['share_L2']['rect1']['q90']:.0f}")
for key, kt in [("L4|U", "LfourU"), ("L8|U", "LeightU"), ("L2|U", "LtwoU"), ("L2|kNN1", "LtwoN")]:
    for tag in ["all", "drop"]:
        for m, mt in [(0.01, "a"), (0.03, "b"), (0.1, "c"), (0.3, "d")]:
            mac(f"eonePr{kt}{tag.capitalize()}{mt}", f2(SC["rect1"][tag][key][f"excl_{m}"]))
        mac(f"eonePr{kt}{tag.capitalize()}Eps", f"{SC['rect1'][tag][key]['eps_median']:.3f}")
mac("eonePrBudRatioDrop", f"{SC['rect1']['drop']['L2|U']['eps_median'] / SC['rect1']['drop']['L2|kNN1']['eps_median']:.1f}")
mac("eonePrTwoLeightUAlld", f2(SC["rect2"]["all"]["L8|U"]["excl_0.3"]))
# coverage of every score of Figure 8, and the ties of the tail score (improve_screen.json)
covs = [scr(c, 2, "U", None, "coverage") for c in ["rect1", "rect2"]] + \
       [scr(c, p, "kNN", 1, "coverage") for c in ["rect1", "rect2"] for p in [2, "trunc1.0", "tail0.5"]] + \
       [np.mean([r["coverage"] for r in PH1 if r["cfg"] == c and r["metric"] == "E"]) for c in ["rect1", "rect2"]]
mac("eoneFigCovMin", f3(min(covs))); mac("eoneFigCovMax", f3(max(covs)))
mac("eoneKnnTailCovTwo", f3(scr("rect2", "tail0.5", "kNN", 1, "coverage")))
mac("eoneTailProbesOne", f"{256 * scr('rect1', 'tail0.5', 'kNN', 1, 'eps'):.0f}")
mac("eoneTailProbesTwo", f"{256 * scr('rect2', 'tail0.5', 'kNN', 1, 'eps'):.0f}")
mac("eoneNonTailCovMax", f3(max(scr(c, p, "kNN", 1, "coverage") for c in ["rect1", "rect2"] for p in [2, "trunc1.0"])))
# what binds the uniform region (Step 5): budget (M+1) eps, floor of Proposition 5.6(iii), typical safe score (100 splits, d_T)
T1 = D["rect1_Tp"].astype(float); bud_, flo_, typ_ = [], [], []
for s_ in range(100):
    prm = np.random.default_rng(1000 + s_).permutation(NS); a_, c_, t_ = prm[:30], prm[30:90], prm[90:]
    dT = lambda X: np.sqrt(((X[:, None] - T1[a_][None]) ** 2).sum(-1).mean(-1))
    flo_.append(dT(T1[a_]).sum() / (2 * 30 * 30)); typ_.append(np.median(dT(T1[t_]).mean(1)))
    sc_ = np.sort(dT(T1[c_]).mean(1)); bud_.append(sc_[int(np.ceil(61 * 0.9)) - 1])
mac("eoneFloorAT", f"{np.median(flo_):.2f}"); mac("eoneTypicalAT", f"{np.median(typ_):.2f}")
assert abs(np.median(bud_) - float(macros["eoneBudgetAT"])) < 0.006
# realised unsafe mass of the 1-rectified model (excess P(A) over approved prompts)
for m, mt in [(0.01, "a"), (0.03, "b")]:
    exc = D["rect1_pA"][Z[:, 2] == m].mean() - D["rect1_pA"][:NS].mean(); tgt = D["true_pA"][Z[:, 2] == m].mean()
    mac(f"eoneExcess{mt}", f"{exc:.4f}"); mac(f"eoneTarget{mt}", f"{tgt:.4f}"); mac(f"eoneRealised{mt}", f"{100 * exc / tgt:.0f}")
# prompt mixing (Corollary 6.3): largest deviation between the field of the mixed prompt and the mixture of fields (phase 1)
mx_ = max(r[f"mixcheck_{k}"] for b in ENF1 for r in b["rows"] for k in ["rho", "0", "E"]); ex_ = int(np.floor(np.log10(mx_)))
mac("eoneMixcheck", f"{mx_ / 10 ** ex_:.0f}\\times10^{{{ex_}}}")
# directed search (e1_robustness.py)
mac("eoneDirUL", f"{max(directed(c, 'U-L2') for c in ['rect1', 'rect2']):.3f}")
mac("eoneDirNear", f"{max(directed(c, me) for c in ['rect1', 'rect2'] for me in ['kNN1-L1', 'kNN1-tr1', 'kNN1-tl.5']):.3f}")
# slack of the plug-in certificate of the improved method (enforced fields, 1-rectified model)
Rk = [r for r in rows_of("rect1", "kNN1-tr1") if r["kappa"] > 0]
mac("eoneSlackBound", f"{np.median([r['bound'] for r in Rk]):.2f}"); mac("eoneSlackPA", f"{np.median([r['pA'] for r in Rk]):.3f}")
# verified deployment (e1_verify.py)
def vrows(cfg, me):
    return [r for b in VER["runs"] if b["cfg"] == cfg and b["method"] == me for r in b["rows"]]
mac("eoneVerDelta", f"{VER['delta_d'] + VER['delta_q']:.2f}"); mac("eoneVerL", f"{VER['L_fresh']}"); mac("eoneVerLbig", f"{VER['L_big']:,}".replace(",", "{,}"))
mac("eoneVerHoeff", f"{np.sqrt(np.log(2 / VER['delta_d']) / (2 * VER['L_fresh'])):.3f}")
allv = [(b, r) for b in VER["runs"] for r in b["rows"]]
mac("eoneVerViol", f"{sum(r['pA'] > r['bound'] for b, r in allv)}"); mac("eoneVerN", f"{len(allv)}")
mac("eoneVerOverMax", pct(max(np.mean([r["d_big"] > b["b"] for r in b["rows"]]) for b in VER["runs"])))
for cfg, t in [("rect1", "One"), ("rect2", "Two")]:
    for me, mt in [("kNN1-tr1", "Tr"), ("kNN1-tl.5", "Tail")]:
        R = vrows(cfg, me); saf = [r for r in R if r["safe"]]; u3 = [r for r in R if r["m"] == 0.3]
        mac(f"eoneVer{t}{mt}PA", f"{np.mean([r['pA'] for r in u3]):.3f}")
        mac(f"eoneVer{t}{mt}Mmd", f"{np.mean([r['mmd'] for r in u3]):.0f}")
        mac(f"eoneVer{t}{mt}Alt", f2(np.mean([r['stage'] != 'as is' for r in saf])))
        mac(f"eoneVer{t}{mt}Fall", f2(np.mean([r['stage'] == 'fallback' for r in u3])))
        mac(f"eoneVer{t}{mt}Bound", f"{np.median([r['bound'] for r in R]):.3f}")
with open(os.path.join(TAB, "e1_numbers.tex"), "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote 2 table bodies and {len(macros)} macros")
for k, v in macros.items():
    print(f"  {k} = {v}")
