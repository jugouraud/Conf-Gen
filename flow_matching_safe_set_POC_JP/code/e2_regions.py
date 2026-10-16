"""E2, step 2: static and adaptive order-1 regions on the prompt embeddings and on the fields, tuned and evaluated on
disjoint prompts (pool of e2_pool.py).

Spaces, i.e. the distance d between a candidate and the safe anchors:
  E          Euclidean distance between prompt embeddings: the region on the (encoded) prompt;
  PB:<cfg>   the one-step map metric of model <cfg> pulled back to the embeddings, |e - e'|_Q with Q = U^T G_0 U
             (Proposition 4.2(ii)-(iii)); for the one-step reflowed model it is exactly the field metric of that model;
  L2:<cfg>   map metric of model/sampler <cfg> (root mean square over the probes; the metric of Theorem 7.1);
  TR:<cfg>   truncated L^1 map distance, r = 1 (Proposition 7.5).
Weights of the order-1 score sum_i w_i d(candidate, anchor_i) (Definition 5.2). By Lemma 5.3 the score is the order-1
Wasserstein distance between the w-weighted law of the anchors and the same law with the candidate added, and the
optimal plan moves the mass w_i / (M + 1) from anchor i to the candidate; adaptive weights move it from close anchors:
  U          uniform: the static region;
  K:c        Gaussian kernel in the same distance, w_i ~ exp(-d_i^2 / (2 h^2)), h = c x median anchor-anchor distance
             (c -> 0: nearest anchor; c -> infinity: uniform);
  N:k        uniform on the k nearest anchors in the same distance;
  X:c        (field spaces only) prompt-similarity weights w_i(e) ~ exp(-|e - e_i|^2 / (2 h^2)), h = c x median anchor-anchor
             embedding distance: the task-adaptive score of the manuscript, with field distances as transport costs.
The weights may depend on the candidate; the score is still a fixed function of the candidate given the anchors, so
split-conformal coverage holds for every reweighting (Proposition 5.4).
Protocol: tuning splits draw anchors and calibration prompts from the tuning pool (300 safe prompts) and score the unsafe
prompts built on bases 0-99; the reweighting of each space and anchor count M is the one that maximises the mean
exclusion over the four masses. Evaluation splits use the evaluation pool (700 safe prompts) and bases 100-199.
Writes ../results/e2/regions.json and ../results/e2/maps.npz (scores of the admission grid on evaluation split 0).
Usage: python e2_regions.py [n_tune] [n_eval]
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys, json, time
import numpy as np
from e1_common import Features, kth_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "e2")
RES1 = os.path.join(HERE, "..", "results", "e1")
CFG1 = json.load(open(os.path.join(RES1, "fit_config.json")))
D = {k: v for k, v in np.load(os.path.join(RES, "pool.npz")).items()}
P2 = json.load(open(os.path.join(RES, "pool_config.json")))
NS, NT = int(D["n_safe"]), int(D["n_tune"])
Z, E = D["Z"], D["E"]
MASSES = P2["masses"]
K_CAL, BETA = 60, 0.1
M_LIST = [15, 30, 60, 120, 240]
CS = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
KS = [1, 2, 3, 5, 10]
SPACES = ["E", "PB:rect1", "PB:rect2", "L2:rect1", "TR:rect1", "L2:rect2", "TR:rect2", "TR:rect2_1step"]
FIELD = lambda sp: not (sp == "E" or sp.startswith("PB"))
SAFE_TUNE, SAFE_EVAL = np.arange(0, NT), np.arange(NT, NS)
UNSAFE = np.arange(NS, len(Z))
BASE = D["base_idx"]; NB_T = P2["n_bases_tune"]
UNS_TUNE, UNS_EVAL = UNSAFE[BASE < NB_T], UNSAFE[BASE >= NB_T]


# ------------------------------------------------------------------ distances to all safe prompts
def pullback_whitener(cfg):
    """C with |C^T (e - e')| = one-step map metric of model cfg on the probes (Proposition 4.2(ii)-(iii))."""
    W = np.load(os.path.join(RES1, "fit.npz"))["W1" if cfg == "rect1" else "W2"]     # (D_e + 1, 2, F)
    feat = Features(CFG1["F"], seed=CFG1["feat_seed"])
    Psi0 = feat(np.load(os.path.join(RES1, "fit.npz"))["x_probe"], 0.0)                 # (L, F)
    V = np.einsum("jcf,lf->jlc", W[1:], Psi0)                                         # (D_e, L, 2): W_j psi(x_l, 0)
    Q = np.einsum("jlc,klc->jk", V, V) / Psi0.shape[0]
    return np.linalg.cholesky(Q + 1e-12 * np.trace(Q) * np.eye(len(Q)))


def eucl(A, B):
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.sqrt(np.maximum(sq, 0.0))


def map_dist(Ta, Tb, kind):
    out = np.empty((len(Ta), len(Tb)), dtype=np.float32)
    Tb = Tb.astype(np.float64)
    for i0 in range(0, len(Ta), 32):
        nrm = np.sqrt(((Ta[i0:i0 + 32, None].astype(np.float64) - Tb[None]) ** 2).sum(-1))
        out[i0:i0 + 32] = np.sqrt((nrm ** 2).mean(-1)) if kind == "L2" else np.minimum(nrm, 1.0).mean(-1)
    return out


def distances():
    """Distance matrices (pool + grid prompts) x (safe prompts), one per space; cached in dist.npz."""
    path = os.path.join(RES, "dist.npz")
    if os.path.exists(path):
        return {k: v for k, v in np.load(path).items()}
    Eall = np.r_[E, D["Eg"]]
    out = {"E": eucl(Eall, E[:NS]).astype(np.float32)}
    for cfg in ["rect1", "rect2"]:
        C = pullback_whitener(cfg); Zw = Eall @ C
        out[f"PB:{cfg}"] = eucl(Zw, Zw[:NS]).astype(np.float32)
    for cfg in ["rect1", "rect2", "rect2_1step"]:
        T = np.r_[D[f"{cfg}_Tp"], D[f"{cfg}_Tpg"]]
        for kind in ["L2", "TR"]:
            out[f"{kind}:{cfg}"] = map_dist(T, T[:NS], kind)
    # Proposition 4.2(ii): the pull-back of the reflowed model's one-step metric is its one-step map metric
    gap = np.abs(out["PB:rect2"] - out["L2:rect2_1step"]).max() / out["L2:rect2_1step"].max()
    print(f"pull-back check: max relative gap between PB:rect2 and L2:rect2_1step = {gap:.2e}", flush=True)
    np.savez_compressed(path, **out)
    return out


# ------------------------------------------------------------------ scores
def softmax_rows(L):
    L = L - L.max(1, keepdims=True); W = np.exp(L)
    return W / W.sum(1, keepdims=True)


def score(Dc, variant, Daa, Ec=None, Eaa=None):
    """Order-1 score of candidates with distances Dc (n, M) to the anchors; Daa: anchor-anchor distances (for bandwidths);
    Ec, Eaa: embedding distances (candidate-anchor, anchor-anchor) for prompt-similarity weights."""
    kind, par = variant
    if kind == "U":
        return Dc.mean(1)
    if kind == "N":
        return np.sort(Dc, axis=1)[:, :par].mean(1)
    iu = np.triu_indices(len(Daa), 1)
    if kind == "K":
        h = par * np.median(Daa[iu])
        return (softmax_rows(-Dc.astype(np.float64) ** 2 / (2 * h * h)) * Dc).sum(1)
    if kind == "X":
        h = par * np.median(Eaa[iu])
        return (softmax_rows(-Ec.astype(np.float64) ** 2 / (2 * h * h)) * Dc).sum(1)
    raise ValueError(kind)


def variants(space):
    v = [("U", None)] + [("K", c) for c in CS] + [("N", k) for k in KS]
    return v + ([("X", c) for c in CS] if FIELD(space) else [])


def vname(v):
    return v[0] if v[1] is None else f"{v[0]}:{v[1]:g}"


def split(pool, M, seed):
    perm = np.random.default_rng(seed).permutation(pool)
    return perm[:M], perm[M:M + K_CAL], perm[M + K_CAL:]


def evaluate(DS, space, v, anc, cal, te, uns):
    Dm = DS[space]
    Daa = Dm[np.ix_(anc, anc)]
    rows = np.r_[cal, te, uns]
    Ec = DS["E"][np.ix_(rows, anc)] if v[0] == "X" else None
    Eaa = DS["E"][np.ix_(anc, anc)] if v[0] == "X" else None
    sc = score(Dm[np.ix_(rows, anc)], v, Daa, Ec, Eaa)
    sc_cal, sc_te, sc_un = sc[:len(cal)], sc[len(cal):len(cal) + len(te)], sc[len(cal) + len(te):]
    eps = kth_threshold(sc_cal, BETA)
    r = dict(eps=float(eps), coverage=float(np.mean(sc_te <= eps)) if len(te) else float("nan"))
    for m in MASSES:
        r[f"excl_{m}"] = float(np.mean(sc_un[Z[uns, 2] == m] > eps))
    r["J"] = float(np.mean([r[f"excl_{m}"] for m in MASSES]))
    return r


def run(DS, n_tune, n_eval):
    res = {"tune": {}, "eval": {}, "best": {}}
    for M in M_LIST:
        for space in SPACES:
            for v in variants(space):
                key = f"{space}|{M}|{vname(v)}"
                R = [evaluate(DS, space, v, *split(SAFE_TUNE, M, 5000 + s), UNS_TUNE) for s in range(n_tune)]
                res["tune"][key] = {k: float(np.nanmean([r[k] for r in R])) for k in R[0]}
                n = n_eval if M == 30 else max(n_eval // 2, 1)
                R = [evaluate(DS, space, v, *split(SAFE_EVAL, M, 1000 + s), UNS_EVAL) for s in range(n)]
                res["eval"][key] = {k: float(np.nanmean([r[k] for r in R])) for k in R[0]}
                res["eval"][key]["eps_median"] = float(np.median([r["eps"] for r in R]))
            ad = [v for v in variants(space) if v[0] != "U"]
            best = max(ad, key=lambda v: res["tune"][f"{space}|{M}|{vname(v)}"]["J"])
            res["best"][f"{space}|{M}"] = vname(best)
        print(f"M = {M}: done", flush=True)
    return res


def maps(DS, res):
    """Scores of the admission grid on evaluation split 0 (M = 30), static and tuned adaptive, for every space."""
    anc, cal, te = split(SAFE_EVAL, 30, 1000)
    ng = len(D["Eg"]); grid_rows = np.arange(len(Z), len(Z) + ng)
    out = {"anc": anc, "cal": cal}
    for space in SPACES:
        for tag, vn in [("static", "U"), ("adaptive", res["best"][f"{space}|30"])]:
            v = (vn, None) if vn == "U" else (vn.split(":")[0], float(vn.split(":")[1]) if vn[0] != "N" else int(vn.split(":")[1]))
            Dm = DS[space]; Daa = Dm[np.ix_(anc, anc)]
            rows = np.r_[cal, grid_rows]
            Ec = DS["E"][np.ix_(rows, anc)] if v[0] == "X" else None
            Eaa = DS["E"][np.ix_(anc, anc)] if v[0] == "X" else None
            sc = score(Dm[np.ix_(rows, anc)], v, Daa, Ec, Eaa)
            eps = kth_threshold(sc[:len(cal)], BETA)
            out[f"{space}|{tag}"] = (sc[len(cal):] / eps).reshape(tuple(D["grid_shape"]))
    np.savez_compressed(os.path.join(RES, "maps.npz"), **out)


def main():
    n_tune = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    t0 = time.time()
    DS = distances()
    print(f"[{time.time()-t0:.0f}s] distances", flush=True)
    res = run(DS, n_tune, n_eval)
    json.dump(res, open(os.path.join(RES, "regions.json"), "w"), indent=0)
    maps(DS, res)
    print(f"[{time.time()-t0:.0f}s] saved regions.json and maps.npz")
    for space in SPACES:
        b = res["best"][f"{space}|30"]
        u, a = res["eval"][f"{space}|30|U"], res["eval"][f"{space}|30|{b}"]
        fmt = lambda r: f"cov {r['coverage']:.3f} excl " + " ".join(f"{r[f'excl_{m}']:.2f}" for m in MASSES) + f" J {r['J']:.3f}"
        print(f"{space:16s} static: {fmt(u)} | adaptive {b:7s}: {fmt(a)}")


if __name__ == "__main__":
    main()
