"""Every number quoted in the note, written as LaTeX macros / table bodies into ../tables/ from the result files."""
import os, json, glob, numpy as np
from lfm_common import Features, tgrid

HERE = os.path.dirname(os.path.abspath(__file__))
RES, TAB = os.path.join(HERE, "..", "results"), os.path.join(HERE, "..", "tables")
os.makedirs(TAB, exist_ok=True)
A = json.load(open(os.path.join(RES, "part_a_splits.json")))
Ac = np.load(os.path.join(RES, "part_a_cache.npz"))
C = np.load(os.path.join(RES, "part_c_state.npz"))
Dc = np.load(os.path.join(RES, "part_d_cache.npz"))
Dd = json.load(open(os.path.join(RES, "part_d_splits.json")))
E = np.load(os.path.join(RES, "example_fits.npz"))
rows_ft = [r for f in sorted(glob.glob(os.path.join(RES, "part_c_ft_*.json"))) for r in json.load(open(f))]
macros = {}
def mac(name, val): macros[name] = val

def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def pct(x, d=1): return f"{100 * x:.{d}f}\\%"

# ---------------- pilot A: pairwise KL of the safe family, P(A) of the safe fields
P = Ac["G_LL"].shape[0]; off = ~np.eye(P, dtype=bool)
klA = Ac["G_LL"][off] / 2
mac("pairKLAmed", f3(np.median(klA))); mac("pairKLAninety", f3(np.quantile(klA, .9)))
mac("pALmean", f"{Ac['pA_L'].mean():.1e}".replace("e-0", "\\times 10^{-") + "}")
klD = Dc["REV_LL"][~np.eye(Dc["REV_LL"].shape[0], dtype=bool)]
mac("pairKLDmed", f3(np.median(klD))); mac("pairKLDninety", f3(np.quantile(klD, .9)))
mac("pALDmean", f"{Dc['pA_L'].mean():.4f}"); mac("pALCmean", f"{C['pA_all'].mean():.4f}")

# ---------------- Table: KL of a contaminated field to its own clean field (pilot A exact, pilot D linear FM SDE)
lines = []
B_mass, B_base = Ac["B_mass"], Ac["B_base"]
for m in np.unique(B_mass):
    s = B_mass == m; base = B_base[s]
    rev = np.median(Ac["G_BL"][s][np.arange(s.sum()), base] / 2)
    fwd = np.median(Ac["G_LB"][base, np.where(s)[0]] / 2)
    pAb = Ac["pA_B"][s].mean()
    dpart = "& -- & -- & --"
    if m in list(Dc["masses"]):
        k = list(Dc["masses"]).index(m); Pd = Dc["REV_BL"].shape[1]
        drev = np.median(Dc["REV_BL"][k][np.arange(Pd), np.arange(Pd)]); dfwd = np.median(Dc["FWD_BL"][k][np.arange(Pd), np.arange(Pd)])
        dpart = f"& {Dc['pA_B'][k].mean():.4f} & {dfwd:.3f} & {drev:.3f}"
    lines.append(f"{m:g} & {-np.log(1 - m):.4f} & {pAb:.4f} & {fwd:.4f} & {rev:.3f} {dpart} \\\\")
open(os.path.join(TAB, "tab_kl_mass.tex"), "w").write("\n".join(lines) + "\n")

# ---------------- Table: coverage and exclusion
def row(name, cov, excl_by_m, masses_all):
    cells = [f"{np.mean(cov):.3f}"] + [f"{excl_by_m[m]:.2f}" if m in excl_by_m else "--" for m in masses_all]
    return f"{name} & " + " & ".join(cells) + " \\\\"
masses_all = [0.002, 0.01, 0.03, 0.1, 0.3]
mA = A["masses"]
exA = {k: {m: np.mean(A["exclusion"][k][str(m)]) for m in mA} for k in ["F", "R", "K"]}
mC = [float(x) for x in C["masses"]]
exC = {"ref": dict(zip(mC, C["excl_ref"])), "on": dict(zip(mC, C["excl_on"]))}
exD = {k: {m: np.mean(Dd[k]["excl"][str(m)]) for m in mC} for k in ["fwd", "rev"]}
lines = [
    "\\multicolumn{7}{l}{\\emph{(a) diffusion, exact scores (200 splits)}}\\\\",
    row("forward, $c=\\sqrt{2\\mathrm{KL}(P^{(i)}\\|P^{v})}$", A["coverage"]["F"], exA["F"], masses_all),
    row("reverse, $c=\\sqrt{2\\mathrm{KL}(P^{v}\\|P^{(i)})}$", A["coverage"]["R"], exA["R"], masses_all),
    row("reverse, $c=\\mathrm{KL}(P^{v}\\|P^{(i)})$", A["coverage"]["K"], exA["K"], masses_all),
    "\\midrule \\multicolumn{7}{l}{\\emph{(b) linear flow matching, ODE sampler (100 splits)}}\\\\",
    row("fixed Gram, $c=\\|\\theta-\\theta_i\\|_{G}$", C["cov_ref"], exC["ref"], masses_all),
    row("on-policy Gram, $c=\\|\\theta-\\theta_i\\|_{G_{\\mathrm{on}}(\\theta)}$", C["cov_on"], exC["on"], masses_all),
    "\\midrule \\multicolumn{7}{l}{\\emph{(c) linear flow matching, SDE sampler (200 splits)}}\\\\",
    row("forward, $c=\\mathrm{KL}(P^{(i)}\\|P^{v})$", Dd["fwd"]["cov"], exD["fwd"], masses_all),
    row("reverse, $c=\\mathrm{KL}(P^{v}\\|P^{(i)})$", Dd["rev"]["cov"], exD["rev"], masses_all),
]
open(os.path.join(TAB, "tab_coverage.tex"), "w").write("\n".join(lines) + "\n")
mac("covAF", f3(np.mean(A["coverage"]["F"]))); mac("covAK", f3(np.mean(A["coverage"]["K"])))
mac("exclAFthree", f2(exA["F"][0.03])); mac("exclAKthree", f2(exA["K"][0.03])); mac("exclAKone", f2(exA["K"][0.01]))
mac("exclAFone", f2(exA["F"][0.01]))
mac("exclCrefthree", f2(exC["ref"][0.03])); mac("exclConthree", f2(exC["on"][0.03]))
mac("exclDfwdthree", f2(exD["fwd"][0.03])); mac("exclDrevthree", f2(exD["rev"][0.03]))
mac("exclDfwdten", f2(exD["fwd"][0.1]))
mac("exclCononeP", f3(exC["on"][0.01])); mac("exclConthreeP", f3(exC["on"][0.03]))      # three decimals: the on-policy rows of
mac("exclDrevoneP", f3(exD["rev"][0.01])); mac("exclDrevthreeP", f3(exD["rev"][0.03]))  # pilots C and D agree to two only

# ---------------- Table: certificates
def cert(name, bounds, inside):
    b, x = np.array(bounds), np.array(inside)
    return f"{name} & {np.median(b):.3f} & {np.median(x):.4f} & {x.max():.4f} & {int(np.sum(x > b))}/{len(b)} \\\\"
lines = [
    "\\multicolumn{5}{l}{\\emph{(a) diffusion, exact scores}}\\\\",
    cert("forward region, Theorem~\\ref{th:tv}", A["bounds"]["F_tv"], A["inside_max"]["F_tv"]),
    cert("reverse $\\sqrt{2\\mathrm{KL}}$ region, Theorem~\\ref{th:tv}", A["bounds"]["R_tv"], A["inside_max"]["R_tv"]),
    cert("reverse-KL region, Theorem~\\ref{th:rare}", A["bounds"]["K_kl"], A["inside_max"]["K_kl"]),
    "\\midrule \\multicolumn{5}{l}{\\emph{(b) linear flow matching, SDE sampler}}\\\\",
    cert("forward region, Theorem~\\ref{th:tv}", Dd["fwd"]["bound"], Dd["fwd"]["inside"]),
    cert("reverse-KL region, Theorem~\\ref{th:rare}", Dd["rev"]["bound"], Dd["rev"]["inside"]),
]
open(os.path.join(TAB, "tab_certificates.tex"), "w").write("\n".join(lines) + "\n")
mac("certAKmed", f3(np.median(A["bounds"]["K_kl"]))); mac("certAFmed", f3(np.median(A["bounds"]["F_tv"])))
mac("certDrevmed", f3(np.median(Dd["rev"]["bound"]))); mac("certDfwdmed", f3(np.median(Dd["fwd"]["bound"])))
mac("insideAKmed", f"{np.median(A['inside_max']['K_kl']):.4f}"); mac("insideAFmed", f"{np.median(A['inside_max']['F_tv']):.4f}")
mac("insideAFmax", f"{np.max(A['inside_max']['F_tv']):.4f}")
mac("insideDrevmed", f"{np.median(Dd['rev']['inside']):.4f}"); mac("insideDfwdmed", f"{np.median(Dd['fwd']['inside']):.4f}")

# ---------------- Table: few-shot fine-tuning
names = [("U_lam1e-05", "least squares, $\\lambda=10^{-5}$"), ("U_lam0.001", "least squares, $\\lambda=10^{-3}$"),
         ("U_lam0.1", "least squares, $\\lambda=10^{-1}$"), ("S_gam0.1", "shrinkage, $\\gamma=0.1$"),
         ("S_gam1", "shrinkage, $\\gamma=1$"), ("S_gam10", "shrinkage, $\\gamma=10$"),
         ("C_ref", "conformal, fixed $G$ (SOCP)"), ("C_on", "conformal, on-policy (Alg.~\\ref{alg:scp})"),
         ("centre", "centre $\\theta_c$ (no data)"), ("oracle", "ample-data fit $f(\\tau)$")]
lines = []
for k, lab in names:
    cells = []
    for nb in [0, 3]:
        R = [r["res"][k] for r in rows_ft if r["n_bad"] == nb]
        mm = np.array([x["mmd"] for x in R]) * 1e3; pA = np.array([x["pA"] for x in R])
        cells += [f"{mm.mean():.1f}", f"{pA.mean():.4f}"]
    R3 = [r["res"][k] for r in rows_ft if r["n_bad"] == 3]
    cells += [f"{np.mean([x['s_ref'] <= 1 + 1e-6 for x in R3]):.2f}", f"{np.mean([x['s_on'] <= 1 + 1e-6 for x in R3]):.2f}"]
    lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    if k in ("U_lam0.1", "S_gam10", "C_on"):
        lines.append("\\midrule")
open(os.path.join(TAB, "tab_fewshot.tex"), "w").write("\n".join(lines) + "\n")
mm = lambda k, nb: np.array([r["res"][k]["mmd"] for r in rows_ft if r["n_bad"] == nb]) * 1e3
pa = lambda k, nb: np.array([r["res"][k]["pA"] for r in rows_ft if r["n_bad"] == nb])
for nb, tag in [(0, "clean"), (3, "cont")]:
    mac(f"geoCref{tag}", f2(np.exp(np.mean(np.log(mm("C_ref", nb) / mm("U_lam0.001", nb))))))
    mac(f"geoCon{tag}", f2(np.exp(np.mean(np.log(mm("C_on", nb) / mm("U_lam0.001", nb))))))
    mac(f"winCref{tag}", f"{int(np.sum(mm('C_ref', nb) < mm('U_lam0.001', nb)))}")
    mac(f"winCon{tag}", f"{int(np.sum(mm('C_on', nb) < mm('U_lam0.001', nb)))}")
mac("ntasks", f"{len([r for r in rows_ft if r['n_bad'] == 0])}")
mac("pAUcont", f"{pa('U_lam0.001', 3).mean():.3f}"); mac("pACrefcont", f"{pa('C_ref', 3).mean():.3f}")
mac("pAConcont", f"{pa('C_on', 3).mean():.3f}"); mac("pASten", f"{pa('S_gam10', 3).mean():.4f}")
bo = np.array([r["res"]["C_on"]["backoff"] for r in rows_ft])
mac("backoffmax", f"{bo.max():.3f}"); mac("backofffrac", f"{np.mean(bo > 0):.2f}")
mac("socptime", f"{np.mean([r['res']['C_ref']['time'] for r in rows_ft]):.2f}")
mac("epsref", f"{float(C['eps_ref']):.4f}"); mac("epson", f"{float(C['eps_on']):.4f}")
mac("exampletask", f"{int(E['j'])}")
for k, tag in [("U_lam0.001", "U"), ("C_ref", "Cref"), ("C_on", "Con"), ("S_gam10", "S"), ("centre", "centre"), ("oracle", "oracle")]:
    mac(f"ex{tag}pA", f"{float(E[f'pA_{k}_3']):.3f}")
mac("excentresref", f2(float(E["sref_centre_3"]))); mac("excentreson", f2(float(E["son_centre_3"])))

# ---------------- Gronwall factor of Proposition 3 for the centre field (a-priori Lipschitz bound)
feat = Features(256, seed=0)
W = E["W_centre_3"]; Fn = feat.F
h = np.diff(tgrid)
Lk = []
for t in tgrid[:-1]:
    rff = np.sqrt(2.0 / Fn) * np.sum(np.linalg.norm(W[:, :Fn], axis=0) * np.linalg.norm(feat.Wx, axis=1))
    Jaff = W[:, Fn + 2:Fn + 4] + t * W[:, Fn + 4:Fn + 6]          # columns x1, x2, t*x1, t*x2
    Lk.append(rff + np.linalg.norm(Jaff, 2))
Lk = np.array(Lk); log10Gamma = np.sum(np.log10(1 + h[1:] * Lk[1:]))   # Gamma_0 = prod_{l=1}^{J-1}(1 + h_l L_l)
mac("lipmin", f"{Lk.min():.0f}"); mac("lipmax", f"{Lk.max():.0f}"); mac("gronwalllog", f"{log10Gamma:.0f}")

with open(os.path.join(TAB, "numbers.tex"), "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros and 4 tables to {TAB}")
for k, v in macros.items(): print(f"  {k} = {v}")
