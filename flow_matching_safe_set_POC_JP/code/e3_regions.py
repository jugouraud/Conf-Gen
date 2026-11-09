"""E3, step 2: conformal regions of approved (winter) prompts on the Sana pool, in every space of safe_prompt_fields.tex, with static
and adaptive weights; tuning on the tuning pool, coverage and exclusion on the evaluation pool.

Spaces (distance between a request and an anchor):
  E    prompt, Euclidean distance between pooled Gemma embeddings;
  MH   prompt, Mahalanobis distance, covariance S of the approved embeddings of the TUNING pool regularised by rho tr(S) / D;
  PB   source slice (the pull-back distance of Proposition 4.2): RMS over the probes of the distance between guided velocities at
       the noise end, || v(x, s=1; c) - v(x, s=1; c_i) ||, in latent space;
  L2   field, map metric in chi-space (Definition 4.1(ii)): sqrt(mean_l || chi(T_c(x_l)) - chi(T_{c_i}(x_l)) ||^2), the method as written;
  TR   field, truncated map distance (Proposition 7.5): mean_l min(|| chi(T_c(x_l)) - chi(T_{c_i}(x_l)) ||, r) / r.
Weights: U (static, uniform); adaptive: K:c Gaussian kernel in the region's distance with bandwidth c x median anchor-anchor
distance, N:k equal weights on the k nearest anchors, X:c kernel in the Euclidean prompt distance (non-prompt spaces).
Protocol (as in e2_regions.py): M = 30 anchors and K = 40 calibration prompts, beta = 0.1 (nominal coverage 37/41 = 0.902). For each
space, the adaptive variant (and rho for MH, r for TR, also for the static region) maximises the exclusion of the 16 summer prompts
of the tuning pool over 50 tuning splits (70 approved prompts: anchors + calibration). Evaluation: 200 splits of the 120 approved
prompts of the evaluation pool (test = the other 50); exclusion of the summer, neutral and disguised requests, of the requests
whose own outputs the judge finds unsafe (P(winter) < 0.5 on the evaluation noise) or safe, and of the safe but unfamiliar
requests (paraphrased winter prompts, winter and summer prompts of new scenes) when e3_prompts.py --extend has added them.
Writes ../results/e3/regions.json.     Usage: python e3_regions.py
"""
import os, json
import numpy as np
from e3_common import RES, kth_threshold

M, K, BETA = 30, 40, 0.1
N_TUNE, N_EVAL = 50, 200
RHOS = [1e-4, 1e-2, 1e-1, 1.0]
CS = [0.03, 0.1, 0.3, 1.0]
KS = [1, 3]


def load():
    rows = json.load(open(os.path.join(RES, "prompts.json")))["prompts"]
    D = {k: v for k, v in np.load(os.path.join(RES, "pool.npz")).items()}
    kind = np.array([r["kind"] for r in rows]); pool = np.array([r["pool"] for r in rows])
    idx = {f"{k}_{p}": np.where((kind == k) & (pool == p))[0] for k in np.unique(kind) for p in ["tune", "eval"]}
    return rows, D, idx


def eucl(A, B):
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.sqrt(np.maximum(sq, 0.0))


def probe_dists(F, G):
    """Per-probe Euclidean distances between (n, P, d) and (m, P, d) features: (n, m, P)."""
    return np.stack([eucl(F[:, l], G[:, l]) for l in range(F.shape[1])], -1)


def whitener(X, rho):
    S = np.cov(X.T); lam, V = np.linalg.eigh(S)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(lam, 0) + rho * np.trace(S) / len(S))) @ V.T


class Space:
    """A distance between objects described by features: pooled embeddings (E, MH), slices (PB) or chi features (L2, TR)."""
    def __init__(self, name, D, idx, par=None):
        self.name, self.par = name, par
        if name == "MH":
            self.Wm = whitener(D["pooled"][idx["approved_tune"]], par)

    def feats(self, D):
        return {"E": D["pooled"], "MH": D["pooled"], "PB": D["slice"].astype(np.float32), "L2": D["chi_probe"],
                "TR": D["chi_probe"]}[self.name]

    def dist(self, F, G):
        if self.name == "E":
            return eucl(F, G)
        if self.name == "MH":
            return eucl(F @ self.Wm, G @ self.Wm)
        pd = probe_dists(F, G)
        if self.name == "PB":
            return np.sqrt((pd ** 2).mean(-1))
        if self.name == "L2":
            return np.sqrt((pd ** 2).mean(-1))
        return np.minimum(pd, self.par).mean(-1) / self.par


def score(Dc, variant, Daa, Ec=None, Eaa=None):
    """Order-1 score of candidates with distances Dc (n, M) to the anchors (Definition 5.2 without its factor 1/(M+1))."""
    kind, par = variant
    if kind == "U":
        return Dc.mean(1)
    if kind == "N":
        return np.sort(Dc, 1)[:, :par].mean(1)
    iu = np.triu_indices(len(Daa), 1)
    h = par * np.median((Daa if kind == "K" else Eaa)[iu])
    L = -((Dc if kind == "K" else Ec).astype(np.float64) ** 2) / (2 * h * h)
    L -= L.max(1, keepdims=True); w = np.exp(L); w /= w.sum(1, keepdims=True)
    return (w * Dc).sum(1)


def variants(space):
    v = [("U", None)] + [("K", c) for c in CS] + [("N", k) for k in KS]
    return v + ([("X", c) for c in [0.1, 0.3]] if space not in ("E", "MH") else [])


def vname(v):
    return v[0] if v[1] is None else f"{v[0]}:{v[1]:g}"


def parse(vn):
    if vn == "U":
        return ("U", None)
    k, p = vn.split(":")
    return (k, int(p) if k == "N" else float(p))


def split(pool, seed, m=M, k=K):
    perm = np.random.default_rng(seed).permutation(pool)
    return perm[:m], perm[m:m + k], perm[m + k:]


def evaluate(Dfull, Efull, v, anc, cal, groups):
    Daa, Eaa = Dfull[np.ix_(anc, anc)], Efull[np.ix_(anc, anc)]
    sc = lambda rows: score(Dfull[np.ix_(rows, anc)], v, Daa, Efull[np.ix_(rows, anc)], Eaa)
    eps = kth_threshold(sc(cal), BETA)
    out = {"eps": float(eps)}
    for g, rows in groups.items():
        if len(rows):
            s = sc(rows)
            out[g] = float(np.mean(s > eps)) if g != "test" else float(np.mean(s <= eps))
            out[f"{g}_margin"] = float(np.median(s / eps))
    return out


def space_grid(name):
    return {"MH": [("MH", r) for r in RHOS], "TR": [("TR", r) for r in TR_RS]}.get(name, [(name, None)])


def main():
    global TR_RS
    rows, D, idx = load()
    chi = D["chi_probe"]
    pd_app = probe_dists(chi[idx["approved_tune"]], chi[idx["approved_tune"]])
    q = np.quantile(pd_app[pd_app > 0], [0.25, 0.5, 0.75])
    TR_RS = [float(np.round(x, 2)) for x in q]                                # truncation levels: quartiles of approved distances
    E_full = eucl(D["pooled"], D["pooled"])
    label_unsafe = np.nanmean(D["pwB_eval"], 1) < 0.5 if "pwB_eval" in D else None
    req = np.r_[idx["summer_eval"], idx["neutral_eval"], idx["disguised_eval"]]
    unfamiliar = [k for k in UNFAMILIAR if len(idx.get(f"{k}_eval", []))]       # appended by e3_prompts.py --extend
    req_all = np.r_[req, *[idx[f"{k}_eval"] for k in unfamiliar]]
    res = {"tr_levels": TR_RS, "tune": {}, "eval": {}, "best": {}, "per_request": {}}
    for name in ["E", "MH", "PB", "L2", "TR"]:
        cands = []
        for sname, par in space_grid(name):
            sp = Space(sname, D, idx, par); F = sp.feats(D); Dfull = sp.dist(F, F)
            for v in variants(name):
                T = [evaluate(Dfull, E_full, v, *split(idx["approved_tune"], 5000 + s)[:2], {"summer": idx["summer_tune"]})
                     for s in range(N_TUNE)]
                J, mg = np.mean([t["summer"] for t in T]), np.median([t["summer_margin"] for t in T])
                res["tune"][f"{name}|{par}|{vname(v)}"] = dict(J=float(J), margin=float(mg))
                cands.append((J, mg, par, v))
        # the variant with the largest exclusion of the tuning summer prompts; ties broken by the median score/threshold ratio
        for tag, pool in [("static", [c for c in cands if c[3][0] == "U"]), ("adaptive", [c for c in cands if c[3][0] != "U"])]:
            J, mg, par, v = max(pool, key=lambda c: (round(c[0], 6), c[1]))
            sp = Space(name, D, idx, par); F = sp.feats(D); Dfull = sp.dist(F, F)
            R = []
            for s in range(N_EVAL):
                anc, cal, te = split(idx["approved_eval"], 1000 + s)
                groups = {"test": te, "summer": idx["summer_eval"], "neutral": idx["neutral_eval"], "disguised": idx["disguised_eval"]}
                groups.update({k: idx[f"{k}_eval"] for k in unfamiliar})
                if label_unsafe is not None:
                    groups["judged_unsafe"] = req[label_unsafe[req]]; groups["judged_safe"] = req[~label_unsafe[req]]
                R.append(evaluate(Dfull, E_full, v, anc, cal, groups))
            key = f"{name}|{tag}"
            res["best"][key] = dict(par=par, variant=vname(v), tune_J=float(J), tune_margin=float(mg))
            res["eval"][key] = {k: float(np.mean([r[k] for r in R])) for k in R[0] if not k.endswith("_margin")}
            res["eval"][key].update({k: float(np.median([r[k] for r in R])) for k in R[0] if k.endswith("_margin")})
            res["eval"][key]["eps_median"] = float(np.median([r["eps"] for r in R]))
            # how often each request is excluded over the evaluation splits
            excl = np.zeros(len(req_all))
            for s in range(N_EVAL):
                anc, cal, te = split(idx["approved_eval"], 1000 + s)
                Daa, Eaa = Dfull[np.ix_(anc, anc)], E_full[np.ix_(anc, anc)]
                eps = kth_threshold(score(Dfull[np.ix_(cal, anc)], v, Daa, E_full[np.ix_(cal, anc)], Eaa), BETA)
                excl += score(Dfull[np.ix_(req_all, anc)], v, Daa, E_full[np.ix_(req_all, anc)], Eaa) > eps
            res["per_request"][key] = dict(zip([int(j) for j in req_all], (excl / N_EVAL).tolist()))
            print(f"{key:12s} {vname(v):7s} par={par}: tune J {J:.2f} | eval cov {res['eval'][key]['test']:.3f} excl summer "
                  f"{res['eval'][key]['summer']:.2f} neutral {res['eval'][key]['neutral']:.2f} disguised {res['eval'][key]['disguised']:.2f}"
                  + (f" | judged unsafe {res['eval'][key].get('judged_unsafe', float('nan')):.2f} judged safe {res['eval'][key].get('judged_safe', float('nan')):.2f}"
                     if label_unsafe is not None else ""), flush=True)
    if label_unsafe is not None:
        res["labels"] = {int(j): bool(label_unsafe[j]) for j in req}
    json.dump(res, open(os.path.join(RES, "regions.json"), "w"), indent=0)


TR_RS = []
UNFAMILIAR = ["paraphrase", "novel_winter", "novel_summer"]
if __name__ == "__main__":
    main()
