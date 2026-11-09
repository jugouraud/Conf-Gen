"""Every number quoted in safe_prompt_fields_audit.tex, recomputed from ../results/ (run e1_robustness.py, e2_robustness.py and
e1_verify.py first). Writes ../tables/audit_numbers.tex (macros \\au...), ../tables/audit_probe.tex and ../tables/audit_attack.tex
(table bodies).     Usage: python audit_checks.py
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import re, json
import numpy as np
from scipy.stats import beta as betad

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES1, RES2, TAB = (os.path.join(ROOT, "results", "e1"), os.path.join(ROOT, "results", "e2"), os.path.join(ROOT, "tables"))
D = np.load(os.path.join(RES1, "fit.npz"))
SCR = json.load(open(os.path.join(RES1, "improve_screen.json")))
PH1 = json.load(open(os.path.join(RES1, "eval_splits.json")))
PH2 = json.load(open(os.path.join(RES1, "phase2.json")))
ENF1 = json.load(open(os.path.join(RES1, "eval_enforce.json")))
ROB1 = json.load(open(os.path.join(RES1, "robustness.json")))
ROB2 = json.load(open(os.path.join(RES2, "robustness.json")))
EA = json.load(open(os.path.join(RES2, "enforce_attack.json")))
REG = json.load(open(os.path.join(RES2, "regions.json")))
NS = int(D["n_safe"]); Z = D["Z"]; MASSES = [0.01, 0.03, 0.1, 0.3]
NUM = {}


def mac(k, v):
    NUM[k] = v


def sci(x, d=1):
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    return f"{m:.{d}f}\\times10^{{{e}}}"


def arrow(a, b, fmt="{:.2f}"):
    return f"{fmt.format(a)}$\\to${fmt.format(b)}"


# ---------------------------------------------------------------- F1: sampler blow-ups and the extreme probe
P1 = ROB1["probe"]
mac("auMaxProbeOne", f"{P1['maxnorm_probe']['rect1']:.1f}"); mac("auMaxProbeTwo", f"{P1['maxnorm_probe']['rect2']:.1f}")
mac("auMaxEvalOne", sci(P1["maxnorm_eval"]["rect1"])); mac("auMaxEvalTwo", f"{P1['maxnorm_eval']['rect2']:.1f}")
mac("auMaxGridOne", sci(ROB2["probe"]["maxnorm_grid"]))
log = open(os.path.join(ROOT, "results", "e1_fit.log")).read()
mac("auMaxReflow", sci(float(re.search(r"max \|x1\| = ([0-9.e+]+)\)", log).group(1))))
assert P1["flagged"]["rect1"] == P1["flagged"]["rect2"] == ROB2["probe"]["flagged"], "flagged probes differ between pools/models"
mac("auProbe", str(P1["flagged"]["rect1"][0])); mac("auProbeNorm", f"{P1['x0_norm']['rect1'][0]:.2f}")
assert int(np.argmax(np.linalg.norm(D["x_probe"], axis=1))) == P1["flagged"]["rect1"][0], "flagged probe is not the largest"
blow = np.linalg.norm(D["rect1_Tp"][:, P1["flagged"]["rect1"][0]].astype(float), axis=-1) > 4.0      # probe beyond radius 4
mac("auBlowSafe", f"{100 * blow[:NS].mean():.0f}")
for m, mt in zip(MASSES, "abcd"):
    mac(f"auBlow{mt}", f"{100 * blow[Z[:, 2] == m].mean():.0f}")
for k, key in [("Two", "share_L2"), ("Four", "share_L4")]:
    s = P1[key]["rect1"]
    mac(f"auShare{k}Med", f"{100 * s['median']:.0f}"); mac(f"auShare{k}Mean", f"{100 * s['mean']:.0f}")
    mac(f"auShare{k}Ninety", f"{100 * s['q90']:.0f}")
S1 = P1["screen"]
for cfg, t in [("rect1", "One"), ("rect2", "Two")]:
    for key, kt in [("L4|U", "LfourU"), ("L8|U", "LeightU"), ("L2|kNN1", "LtwoN"), ("L2|U", "LtwoU"), ("L4|kNN1", "LfourN")]:
        for tag in ["all", "drop"]:
            r = S1[cfg][tag][key]
            for m, mt in zip(MASSES, "abcd"):
                mac(f"au{t}{kt}{tag.capitalize()}{mt}", f"{r[f'excl_{m}']:.2f}")
            mac(f"au{t}{kt}{tag.capitalize()}Eps", f"{r['eps_median']:.3f}")
P2r = ROB2["probe"]["regions"]
for key, kt in [("L2|all|N:1", "LtwoNAll"), ("L2|drop|N:1", "LtwoNDrop"), ("TR|all|N:1", "TrNAll"), ("L2|all|U", "LtwoUAll"),
                ("L2|drop|U", "LtwoUDrop"), ("TR|all|U", "TrUAll"), ("TR|drop|U", "TrUDrop"), ("TR|drop|N:1", "TrNDrop")]:
    for m, mt in zip(MASSES, "abcd"):
        mac(f"auE{kt}{mt}", f"{P2r[key][f'excl_{m}']:.2f}")

lines = []
names = {"L2|U": "uniform, $L^2$", "L4|U": "uniform, $L^4$", "L8|U": "uniform, $L^8$", "trunc1|U": "uniform, truncated $L^1$",
         "L1|kNN1": "nearest, $L^1$", "L2|kNN1": "nearest, $L^2$", "L4|kNN1": "nearest, $L^4$", "trunc1|kNN1": "nearest, truncated $L^1$",
         "tail0.5|kNN1": "nearest, tail score"}
for cfg, title, keys in [("rect1", "E1, 1-rectified model (100 splits)", list(names)),
                         ("rect2", "E1, 2-rectified model (100 splits)", ["L4|U", "L8|U", "L2|kNN1", "trunc1|kNN1"])]:
    lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}}\\\\")
    for key in keys:
        a, d = S1[cfg]["all"][key], S1[cfg]["drop"][key]
        lines.append(f"{names[key]} & " + " & ".join(arrow(a[f'excl_{m}'], d[f'excl_{m}']) for m in MASSES) +
                     f" & {arrow(a['eps_median'], d['eps_median'], '{:.3f}')} \\\\")
lines.append("\\midrule")
lines.append("\\multicolumn{6}{l}{\\emph{E2, field regions of the 1-rectified model (evaluation pool, 200 splits)}}\\\\")
for key, lab in [("L2|{}|U", "uniform, $L^2$"), ("TR|{}|U", "uniform, truncated $L^1$"), ("L2|{}|N:1", "nearest, $L^2$"),
                 ("TR|{}|N:1", "nearest, truncated $L^1$")]:
    a, d = P2r[key.format("all")], P2r[key.format("drop")]
    lines.append(f"{lab} & " + " & ".join(arrow(a[f'excl_{m}'], d[f'excl_{m}']) for m in MASSES) +
                 f" & {arrow(a['eps_median'], d['eps_median'], '{:.3f}')} \\\\")
open(os.path.join(TAB, "audit_probe.tex"), "w").write("\n".join(lines) + "\n")
pc = [S1[c][t][k]["coverage"] for c in ["rect1", "rect2"] for t in ["all", "drop"] for k in S1[c][t]] + \
     [P2r[k]["coverage"] for k in P2r]
mac("auProbeCovMin", f"{min(pc):.3f}"); mac("auProbeCovMax", f"{max(pc):.3f}")

# ---------------------------------------------------------------- F2, F3: attacks and the Mahalanobis prompt baseline
for cfg in ["rect1", "rect2"]:
    q = ROB2[f"Q_{cfg}"]
    t = "One" if cfg == "rect1" else "Two"
    mac(f"auQmin{t}", sci(q["min"])); mac(f"auQmax{t}", sci(q["max"])); mac(f"auQcond{t}", sci(q["cond"]))


def a2(cfg, space, tag):
    v = [r["pA_fresh"] for r in ROB2["attack2"] if (r["cfg"], r["space"], r["weights"]) == (cfg, space, tag)]
    return max(v), float(np.median(v))


def a1(cfg, space, tag):
    if space == "MH":
        b = [x for x in ROB2["mh_attack"] if (x["cfg"], x["weights"]) == (cfg, tag)][0]
    else:
        b = [x for x in EA["attack"] if (x["cfg"], x["space"], x["weights"]) == (cfg, space, tag)][0]
    return max(r["pA_fresh"] for r in b["runs"])


def a2e1(cfg, name):
    v = [r["pA_fresh"] for r in ROB1["attack2"] if (r["cfg"], r["method"]) == (cfg, name)]
    return max(v), float(np.median(v))


def a1e1(cfg, name):
    return max(r["pA_fresh"] for r in [a for a in PH2["attack"] if a["cfg"] == cfg and a["method"] == name][0]["runs"])


lines = []
lab1 = {"U-L2": "field, uniform $L^2$ (as written)", "kNN1-L1": "field, nearest $L^1$", "kNN1-tr1": "field, nearest truncated $L^1$",
        "kNN1-tl.5": "field, nearest tail score"}
for cfg, title in [("rect1", "E1, 1-rectified model, split 0"), ("rect2", "E1, 2-rectified model, split 0")]:
    lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}}\\\\")
    for name in ["U-L2", "kNN1-L1", "kNN1-tr1", "kNN1-tl.5"]:
        w, md = a2e1(cfg, name)
        lines.append(f"{lab1[name]} & {a1e1(cfg, name):.3f} & {w:.3f} & {md:.3f} \\\\")
lines.append("\\midrule")
lab2 = {"E": "prompt, Euclidean", "MH": "prompt, Mahalanobis", "PB:rect1": "pull-back", "PB:rect2": "pull-back",
        "TR:rect1": "field, truncated $L^1$", "TR:rect2_1step": "field, truncated $L^1$"}
for cfg, title, spaces in [("rect1", "E2, 1-rectified model (50 steps), evaluation split 0", ["E", "MH", "PB:rect1", "TR:rect1"]),
                           ("rect2_1step", "E2, reflowed model (one step), evaluation split 0", ["E", "MH", "PB:rect2", "TR:rect2_1step"])]:
    lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}}\\\\")
    for sp in spaces:
        for tag in ["static", "adaptive"]:
            p1 = a1(cfg, sp, tag)
            cells = f"{p1:.3f} & -- & --" if sp == "E" else "{:.3f} & {:.3f} & {:.3f}".format(p1, *a2(cfg, sp, tag))
            lines.append(f"{lab2[sp]}, {tag} & {cells} \\\\")
open(os.path.join(TAB, "audit_attack.tex"), "w").write("\n".join(lines) + "\n")
for tag, t in [("static", "S"), ("adaptive", "A")]:
    w, md = a2("rect1", "PB:rect1", tag)
    mac(f"auPBoneAtkOne{t}", f"{a1('rect1', 'PB:rect1', tag):.3f}"); mac(f"auPBoneAtkTwo{t}", f"{w:.2f}")
    mac(f"auTRoneAtkTwo{t}", f"{a2('rect1', 'TR:rect1', tag)[0]:.3f}")
    mac(f"auPBtwoAtkTwo{t}", f"{a2('rect2_1step', 'PB:rect2', tag)[0]:.3f}")
    mac(f"auTRtwoAtkTwo{t}", f"{a2('rect2_1step', 'TR:rect2_1step', tag)[0]:.3f}")
    mac(f"auMHAtkOne{t}", f"{a1('rect1', 'MH', tag):.3f}"); mac(f"auMHAtkTwo{t}", f"{a2('rect1', 'MH', tag)[0]:.3f}")
    r = ROB2["mh"][tag]
    mac(f"auMHCov{t}", f"{r['coverage']:.3f}"); mac(f"auMHRho{t}", sci(r["rho"], 0) if r["rho"] < 1 else "1")
    for m, mt in zip(MASSES, "abcd"):
        mac(f"auMHEx{t}{mt}", f"{r[f'excl_{m}']:.2f}")
mac("auMHcAd", ROB2["mh"]["adaptive"]["variant"].split(":")[1])
for tag, t in [("static", "S"), ("adaptive", "A")]:                   # back-off of the Mahalanobis region, 1-rectified model, m = 0.3
    rows = [r for b in ROB2["mh_enforce"] if (b["cfg"], b["weights"]) == ("rect1", tag) for r in b["rows"] if r["m"] == 0.3]
    mac(f"auMHEnf{t}", f"{np.mean([r['pA'] for r in rows]):.3f}"); mac(f"auMHMmd{t}", f"{np.mean([r['mmd'] for r in rows]):.0f}")
fa = [a1(c, sp, tg) for c, sps in [("rect1", ["TR:rect1"]), ("rect2_1step", ["TR:rect2_1step"])] for sp in sps for tg in ["static", "adaptive"]]
mac("auFieldAtkMin", f"{min(fa):.3f}"); mac("auFieldAtkMax", f"{max(fa):.3f}")
mac("auEoneAtkTwoMax", f"{max(a2e1(c, n)[0] for c in ['rect1', 'rect2'] for n in ['U-L2', 'kNN1-L1', 'kNN1-tr1', 'kNN1-tl.5']):.3f}")
mac("auEoneAtkTwoNearMax", f"{max(a2e1(c, n)[0] for c in ['rect1', 'rect2'] for n in ['kNN1-L1', 'kNN1-tr1', 'kNN1-tl.5']):.3f}")
# the m coordinate of the approved embeddings is identically 0
P2pool = np.load(os.path.join(RES2, "pool.npz"))
mac("auMvar", f"{np.var(P2pool['E'][:int(P2pool['n_tune']), 2]):.0f}")
nu = ROB2["mh_nuisance"]
mac("auMHNuMin", f"{min(nu[l][t]['excl_0.1'] for l in nu for t in ['static', 'adaptive']):.2f}")

# ---------------------------------------------------------------- M4: the floor is not what binds
def kth(s, beta=0.1):
    k = int(np.ceil((len(s) + 1) * (1 - beta))); return np.sort(s)[k - 1]


for cfg, t in [("rect1", "One"), ("rect2", "Two")]:
    T = D[f"{cfg}_Tp"].astype(float); bud, flo, typ = [], [], []
    for s in range(100):
        perm = np.random.default_rng(1000 + s).permutation(NS); anc, cal, te = perm[:30], perm[30:90], perm[90:]
        dist = lambda X: np.sqrt(((X[:, None] - T[anc][None]) ** 2).sum(-1).mean(-1))
        Daa = dist(T[anc]); flo.append(Daa.sum() / (2 * 30 * 30))
        bud.append(kth(dist(T[cal]).mean(1))); typ.append(np.median(dist(T[te]).mean(1)))
    mac(f"auBudget{t}", f"{np.median(bud):.2f}"); mac(f"auFloor{t}", f"{np.median(flo):.2f}"); mac(f"auTypical{t}", f"{np.median(typ):.2f}")

# ---------------------------------------------------------------- R1: coverage of the scores of Figure 8, ties of the tail score
def scr(cfg, p, kind, par, key):
    R = [r for r in SCR if r["cfg"] == cfg and r["p"] == p and r["kind"] == kind and r["par"] == par]
    return np.mean([r[key] for r in R]) if key != "eps" else np.median([r["eps"] for r in R])


covs = [scr(c, 2, "U", None, "coverage") for c in ["rect1", "rect2"]] + \
       [scr(c, p, "kNN", 1, "coverage") for c in ["rect1", "rect2"] for p in [2, "trunc1.0", "tail0.5"]] + \
       [np.mean([r["coverage"] for r in PH1 if r["cfg"] == c and r["metric"] == "E"]) for c in ["rect1", "rect2"]]
mac("auFigCovMin", f"{min(covs):.3f}"); mac("auFigCovMax", f"{max(covs):.3f}")
mac("auTailCovOne", f"{scr('rect1', 'tail0.5', 'kNN', 1, 'coverage'):.3f}"); mac("auTailCovTwo", f"{scr('rect2', 'tail0.5', 'kNN', 1, 'coverage'):.3f}")
mac("auTailProbesOne", f"{256 * scr('rect1', 'tail0.5', 'kNN', 1, 'eps'):.0f}"); mac("auTailProbesTwo", f"{256 * scr('rect2', 'tail0.5', 'kNN', 1, 'eps'):.0f}")
mac("auTailOneCovTwo", f"{scr('rect2', 'tail1.0', 'kNN', 1, 'coverage'):.3f}"); mac("auTailOneEpsTwo", f"{scr('rect2', 'tail1.0', 'kNN', 1, 'eps'):.3f}")
mac("auTruncCovOne", f"{scr('rect1', 'trunc1.0', 'kNN', 1, 'coverage'):.3f}")

# ---------------------------------------------------------------- R2-R3: certificates, margins
def rows2(cfg, me):
    return [r for b in PH2["enforce"] if b["cfg"] == cfg and b["method"] == me for r in b["rows"]]


for me, t in [("kNN1-tr1", "Tr"), ("kNN1-tl.5", "Tail")]:
    R = [r for r in rows2("rect1", me) if r["kappa"] > 0]
    mac(f"auSlack{t}Bound", f"{np.median([r['bound'] for r in R]):.3f}"); mac(f"auSlack{t}PA", f"{np.median([r['pA'] for r in R]):.3f}")
    mac(f"auBudget{t}", f"{np.median([b['eps'] for b in PH2['enforce'] if b['cfg'] == 'rect1' and b['method'] == me]):.3f}")
hoeff = np.sqrt(np.log(1 / 0.05) / (2 * 2000))
mac("auHoeff", f"{hoeff:.3f}")
mac("auHoeffTailPct", f"{100 * hoeff / float(NUM['auBudgetTail']):.0f}"); mac("auHoeffTrPct", f"{100 * hoeff / float(NUM['auBudgetTr']):.0f}")
mac("auCPzero", f"{betad.ppf(0.95, 1, 2000):.4f}")

RG = np.linspace(0.05, 2.0, 40)                                   # e1_tables.region_bound, split 0, 1-rectified model
anc0 = np.random.default_rng(1000).permutation(NS)[:30]
Te0 = D["rect1_Te"][anc0].astype(float)
q0 = np.array([[np.mean(np.linalg.norm(Te0[i], axis=1) < 0.6 + r) for r in RG] for i in range(30)])
for me, t in [("kNN1-tr1", "Tr"), ("kNN1-tl.5", "Tail")]:
    eps0 = [a for a in PH2["attack"] if a["cfg"] == "rect1" and a["method"] == me][0]["eps"]
    per = q0 + eps0 * np.maximum(1, 1.0 / RG) if me == "kNN1-tr1" else np.where(RG >= 0.5, q0 + eps0, np.inf)
    mac(f"auRegionBound{t}", f"{per.min(1).max():.2f}")

# ---------------------------------------------------------------- R5-R6: one-step safety cost, realised unsafe mass
pa1, pa2 = D["rect1_pA"][:NS].mean(), D["rect2_1step_pA"][:NS].mean()
mac("auOneStepPA", f"{pa2:.4f}"); mac("auRectOnePA", f"{pa1:.4f}"); mac("auOneStepRatio", f"{pa2 / pa1:.0f}")
for m, mt in [(0.01, "a"), (0.03, "b")]:
    ex = D["rect1_pA"][Z[:, 2] == m].mean() - pa1; tr = D["true_pA"][Z[:, 2] == m].mean()
    mac(f"auExcess{mt}", f"{ex:.4f}"); mac(f"auTarget{mt}", f"{tr:.4f}"); mac(f"auRealised{mt}", f"{100 * ex / tr:.0f}")

# ---------------------------------------------------------------- M3: the Gronwall factor
nums = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(os.path.join(TAB, "numbers.tex")).read()))
mac("auLipMin", nums["lipmin"]); mac("auGronLog", nums["gronwalllog"])
mac("auExpLLog", f"{float(nums['lipmin']) * np.log10(np.e):.1f}")

# ---------------------------------------------------------------- housekeeping
C = np.load(os.path.join(ROOT, "results", "part_c_state.npz")); Dd = json.load(open(os.path.join(ROOT, "results", "part_d_splits.json")))
mac("auPocOnOne", f"{C['excl_on'][0]:.3f}"); mac("auPocOnThree", f"{C['excl_on'][1]:.3f}")
mac("auPocRevOne", f"{np.mean(Dd['rev']['excl']['0.01']):.3f}"); mac("auPocRevThree", f"{np.mean(Dd['rev']['excl']['0.03']):.3f}")
mac("auMixcheck", sci(max(r[f"mixcheck_{m}"] for b in ENF1 for r in b["rows"] for m in ["rho", "0", "E"])))
DIST = np.load(os.path.join(RES2, "dist.npz")); P2 = json.load(open(os.path.join(RES2, "pool_config.json")))
NS2, NT2 = int(P2pool["n_safe"]), int(P2pool["n_tune"]); Z2 = P2pool["Z"]; base2 = P2pool["base_idx"]
Mx = DIST["TR:rect1"]
mac("auSpreadOld", f"{Mx[300:NS2, :NS2].mean():.3f}")
ev = np.arange(NT2, NS2); iu = np.triu_indices(len(ev), 1)
mac("auSpreadNew", f"{Mx[np.ix_(ev, ev)][iu].mean():.3f}")
rows_old = NS2 + np.where(Z2[NS2:, 2] == 0.1)[0]
rows_new = NS2 + np.where((Z2[NS2:, 2] == 0.1) & (base2 >= P2["n_bases_tune"]))[0]
mac("auDispOld", f"{np.median(Mx[rows_old, :NS2].min(1)):.3f}"); mac("auDispNew", f"{np.median(Mx[np.ix_(rows_new, ev)].min(1)):.3f}")
# verified deployment (e1_verify.py), for the resolution section
VER = json.load(open(os.path.join(RES1, "verify.json")))
B = [b for b in VER["runs"] if b["cfg"] == "rect1" and b["method"] == "kNN1-tr1"]
mac("auVerOverOneTr", f"{100 * np.mean([r['d_big'] > b['b'] for b in B for r in b['rows']]):.1f}\\%")
allv = [r for b in VER["runs"] for r in b["rows"]]
mac("auVerN", f"{len(allv)}"); mac("auVerViol", f"{sum(r['pA'] > r['bound'] for r in allv)}")
with open(os.path.join(TAB, "audit_numbers.tex"), "w") as f:
    f.write("% written by code/audit_checks.py\n")
    for k, v in NUM.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(NUM)} macros, tables/audit_probe.tex and tables/audit_attack.tex")
for k, v in NUM.items():
    print(f"  {k} = {v}")
