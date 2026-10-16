"""E1, step 2: the method of safe_prompt_fields.tex on the prompt pool of e1_fit.py.

Phase 1 (the method as written): uniform weights; distances d_T (map metric, Def. 3.1(ii)), d_0 (source
slice), d_rho (pooled interpolation path of the safe anchors) and, as a baseline, d_E (Euclidean distance
between prompt embeddings); enforcement by projection = prompt mixing (Prop. 5.2(iii), Cor. 5.3) for the
Mahalanobis distances and by back-off (Prop. 5.4) for d_T; certificates of Theorem 6.1 / Corollary 6.2.
Usage: python e1_eval.py [n_splits]      Writes ../results/e1/eval_splits.json and eval_enforce.json.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time
import numpy as np, cvxpy as cp
from joblib import Parallel, delayed
from e1_common import *

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "e1")
CFG = json.load(open(os.path.join(RES, "fit_config.json")))
D = {k: v for k, v in np.load(os.path.join(RES, "fit.npz")).items()}     # in memory: npz access decompresses each time
D["Gp_safe"] = D["Gp_safe"].astype(float); D["true_safe_samples"] = D["true_safe_samples"].astype(float)
M, K, BETA = 30, 60, 0.1
CONFIGS = ["rect1", "rect2", "rect2_1step"]
JOF = {"rect1": CFG["J"], "rect2": CFG["J"], "rect2_1step": 1}
WST = {"rect1": D["W1"], "rect2": D["W2"], "rect2_1step": D["W2"]}
NS = int(D["n_safe"]); Z, E = D["Z"], D["E"]; NP = len(Z)
MASS = np.r_[np.zeros(NS), Z[NS:, 2]]
feat = Features(CFG["F"], seed=CFG["feat_seed"]); enc = Encoder(CFG["Dr"], seed=CFG["enc_seed"])
x_probe, x_eval = D["x_probe"], D["x_eval"]
R_GRID = np.linspace(0.05, 2.0, 40)


# ------------------------------------------------------------------ distances
def d_rms(Ta, Tb):
    """Map metric on the probe measure: RMS over probes of ||Ta - Tb||, (na, L, 2) x (nb, L, 2) -> (na, nb)."""
    out = np.empty((len(Ta), len(Tb)))
    for i0 in range(0, len(Ta), 64):
        out[i0:i0 + 64] = np.sqrt(((Ta[i0:i0 + 64, None].astype(float) - Tb[None]) ** 2).sum(-1).mean(-1))
    return out


def whitener(G, delta_rel=1e-6):
    """C with CC^T = G + delta I (delta relative to the mean eigenvalue): ||theta - theta'||_H = ||(W - W') C||_F."""
    return np.linalg.cholesky(G + delta_rel * np.trace(G) / len(G) * np.eye(len(G)))


def d_eucl(Xa, Xb):
    Xa, Xb = Xa.reshape(len(Xa), -1), Xb.reshape(len(Xb), -1)
    sq = (Xa ** 2).sum(1)[:, None] + (Xb ** 2).sum(1)[None] - 2 * Xa @ Xb.T
    return np.sqrt(np.maximum(sq, 0.0))


PSI0 = feat(x_probe, 0.0)
C0 = whitener(PSI0.T @ PSI0 / len(x_probe))           # source slice on the probe measure (d_0 = map metric of T^1)


def setup(cfg, anc):
    """Per configuration and split: field coordinates, whitened coordinates and the anchor distance matrices."""
    Ws = fields(WST[cfg], E)
    Crho = whitener(D["Gp_safe"][anc].mean(0))
    Zr, Z0 = Ws @ Crho, Ws @ C0
    T = D[f"{cfg}_Tp"].astype(float)
    dist = {"T": d_rms(T, T[anc]), "0": d_eucl(Z0, Z0[anc]), "rho": d_eucl(Zr, Zr[anc]), "E": d_eucl(E, E[anc])}
    return Ws, {"rho": Zr, "0": Z0, "E": E}, {"rho": Crho, "0": C0}, dist


def split(seed):
    perm = np.random.default_rng(seed).permutation(NS)
    return perm[:M], perm[M:M + K], perm[M + K:]


# ------------------------------------------------------------------ coverage and detection over splits
def run_splits(n_splits):
    rows = []
    for s in range(n_splits):
        anc, cal, te = split(1000 + s)
        for cfg in CONFIGS:
            _, _, _, dist = setup(cfg, anc)
            for met, Dm in dist.items():
                sc = Dm.sum(1) / (M * (M + 1)); eps = kth_threshold(sc[cal], BETA)
                r = dict(split=s, cfg=cfg, metric=met, eps=float(eps), coverage=float(np.mean(sc[te] <= eps)))
                for m in CFG["masses"]:
                    r[f"excl_{m}"] = float(np.mean(sc[NS:][Z[NS:, 2] == m] > eps))
                rows.append(r)
    return rows


# ------------------------------------------------------------------ enforcement
def mix_project(Zc, Zanc, w, eps):
    """Prompt mixing (Prop. 5.2(iii)): gamma in the simplex of R^{M+1} minimising ||theta - theta_c||_H over the region,
    from the Gram matrix of the whitened fields only.  Returns gamma (index 0 = candidate)."""
    Mloc = len(Zanc); Zall = np.vstack([Zc.ravel()[None], Zanc.reshape(Mloc, -1)])
    if w @ np.linalg.norm(Zall[1:] - Zall[0], axis=1) / (Mloc + 1) <= eps:
        g = np.zeros(Mloc + 1); g[0] = 1.0; return g
    Gam = Zall @ Zall.T; lam, V = np.linalg.eigh(Gam); keep = lam > 1e-12 * lam.max()
    L = V[:, keep] * np.sqrt(lam[keep]); sc = np.sqrt(np.trace(Gam) / len(Gam)); L = L / sc
    g = cp.Variable(Mloc + 1); I = np.eye(Mloc + 1)
    cons = [g >= 0, cp.sum(g) == 1, sum(w[i] * cp.norm(L.T @ (g - I[i + 1])) for i in range(Mloc)) <= (Mloc + 1) * eps / sc]
    cp.Problem(cp.Minimize(cp.sum_squares(L.T @ (g - I[0]))), cons).solve(solver=cp.CLARABEL)
    g = np.maximum(g.value, 0); return g / g.sum()


def score_T(Wf, cfg, Tanc):
    T = (simulate(feat, Wf[None], x_probe, JOF[cfg]) if JOF[cfg] > 1 else one_step(feat, Wf[None], x_probe))
    return d_rms(T, Tanc)[0].mean() / (M + 1)


def backoff(Wt, Wc, cfg, Tanc, eps, steps=18):
    """Prop. 5.4(i): bisection on the segment from the proposal to a feasible safe field."""
    if score_T(Wt, cfg, Tanc) <= eps:
        return Wt, 0.0
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        if score_T((1 - mid) * Wt + mid * Wc, cfg, Tanc) <= eps: hi = mid
        else: lo = mid
    return (1 - hi) * Wt + hi * Wc, hi


def evaluate(Wf, cfg, ctx, j):
    """Outputs of a final field on fresh noise; P(A), fidelity, scores, certificate quantities."""
    Te = simulate(feat, Wf[None], x_eval, JOF[cfg])[0] if JOF[cfg] > 1 else one_step(feat, Wf[None], x_eval)[0]
    Tp = simulate(feat, Wf[None], x_probe, JOF[cfg])[0] if JOF[cfg] > 1 else one_step(feat, Wf[None], x_probe)[0]
    T1p = one_step(feat, Wf[None], x_probe)[0]
    res = dict(pA=float(np.mean(np.linalg.norm(Te, axis=1) <= r_A)),
               mmd=float(mmd2(Te[:1000], D["true_safe_samples"][j]) * 1e3),
               zeta=float(np.sqrt(((Tp - T1p) ** 2).sum(-1).mean())))
    # scores relative to each threshold
    res["sT"] = float(d_rms(Tp[None], ctx["Tanc"])[0].mean() / (M + 1) / ctx["eps"]["T"])
    for met in ["0", "rho"]:
        z = (Wf @ ctx["C"][met]).ravel()
        res[f"s{met}"] = float(np.linalg.norm(ctx["Zanc"][met] - z, axis=1).mean() / (M + 1) / ctx["eps"][met])
    # Theorem 6.1 on fresh noise: coupling cost E||Y - Y'|| (shared noise, I uniform) against the budget (M+1) eps_T
    dist_n = np.linalg.norm(Te[None] - ctx["Te_anc"], axis=-1)            # (M, N)
    res["coupling"] = float(dist_n.mean())
    res["budgetT"] = float((M + 1) * ctx["eps"]["T"])
    res["budget0"] = float((M + 1) * ctx["eps"]["0"] + res["zeta"] + ctx["zeta_anc"].mean())   # Cor. 6.2(ii)
    q = ctx["q_anc"]                                                      # q(r) = mean_i P(||Y_i|| < r_A + r)
    res["bound_markov_T"] = float(np.min(q + res["budgetT"] / R_GRID))
    res["bound_markov_0"] = float(np.min(q + res["budget0"] / R_GRID))
    res["bound_tail"] = float(np.min(q + np.array([np.mean(dist_n >= r) for r in R_GRID])))
    return res


def enforce_one(cfg, j, ctx, Wsj):
    """All phase-1 enforcement methods for pool prompt j."""
    out = {}
    Wt = Wsj
    finals = {"none": Wt}
    for met in ["rho", "0", "E"]:
        zc = (Wt @ ctx["C"][met]) if met != "E" else E[j]
        g = mix_project(zc, ctx["Zanc"][met], np.full(M, 1.0 / M), ctx["eps"][met])
        Wmix = g[0] * Wt + np.einsum("i,icf->cf", g[1:], ctx["Wanc"])
        e_sharp = g[0] * E[j] + g[1:] @ E[ctx["anc"]]
        out[f"mixcheck_{met}"] = float(np.abs(fields(WST[cfg], e_sharp[None])[0] - Wmix).max())   # Cor. 5.3
        out[f"gamma0_{met}"] = float(g[0])
        finals[f"proj_{met}"] = Wmix
    # back-off in the map metric towards the nearest feasible safe field (anchors and calibration fields)
    dT = d_rms(ctx["Tp_all"][[j]], ctx["Tp_feas"])[0]
    Wb, kap = backoff(Wt, ctx["W_feas"][np.argmin(dT)], cfg, ctx["Tanc"], ctx["eps"]["T"])
    finals["backoff_T"] = Wb; out["kappa_T"] = float(kap)
    for name, Wf in finals.items():
        for k, v in evaluate(Wf, cfg, ctx, j).items():
            out[f"{name}.{k}"] = v
    return out


def run_enforcement(cfg, split_seed, cand):
    anc, cal, te = split(split_seed)
    Ws, Zw, Cw, dist = setup(cfg, anc)
    eps = {met: kth_threshold(dist[met][cal].sum(1) / (M * (M + 1)), BETA) for met in dist}
    Tp_all = D[f"{cfg}_Tp"].astype(float); Te_all = D[f"{cfg}_Te"].astype(float)
    feas = np.r_[anc, cal]
    sT = dist["T"][feas].sum(1) / (M * (M + 1)); feas = feas[sT <= eps["T"]]
    ctx = dict(anc=anc, eps=eps, C=Cw, Zanc={k: v[anc].reshape(M, -1) for k, v in Zw.items()}, Wanc=Ws[anc],
               Tanc=Tp_all[anc], Te_anc=Te_all[anc], Tp_all=Tp_all, Tp_feas=Tp_all[feas], W_feas=Ws[feas],
               zeta_anc=D[f"{cfg}_zeta"][anc],
               q_anc=np.array([np.mean(np.linalg.norm(Te_all[anc], axis=-1) < r_A + r) for r in R_GRID]))
    rows = Parallel(n_jobs=12)(delayed(enforce_one)(cfg, int(j), ctx, Ws[j]) for j in cand)
    for j, r in zip(cand, rows):
        r.update(cfg=cfg, j=int(j), m=float(MASS[j]), safe=bool(j < NS), in_test=bool(j in set(te)))
    return dict(cfg=cfg, eps={k: float(v) for k, v in eps.items()}, n_feasible=int(len(feas)), rows=rows)


def main():
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    t0 = time.time()
    rows = run_splits(n_splits)
    json.dump(rows, open(os.path.join(RES, "eval_splits.json"), "w"))
    print(f"[{time.time()-t0:.0f}s] {n_splits} splits done", flush=True)
    anc, cal, te = split(1000)
    cand = np.r_[te[:80], np.arange(NS, NP)]
    res = [run_enforcement(cfg, 1000, cand) for cfg in CONFIGS]
    json.dump(res, open(os.path.join(RES, "eval_enforce.json"), "w"))
    print(f"[{time.time()-t0:.0f}s] enforcement done", flush=True)


if __name__ == "__main__":
    main()
