"""E1, phase 2: full evaluation of the improved scores against the phase-1 method, paired on the same splits.

Methods (all calibrated by split conformal, beta = 0.1, M = 30 safe anchors, K = 60 calibration prompts):
  U-L2      phase 1: uniform order-1 score, map metric d_T (L^2 over the probes).
  kNN1-L1   nearest safe field, L^1 map metric.
  kNN1-tr1  nearest safe field, truncated L^1 map metric E[min(|dT|, 1)] (a bounded pseudometric).
  kNN1-tl.5 nearest safe field, tail score P(|dT| >= 0.5) (not a metric).
  E-U, E-kNN1  prompt-space baselines: the same scores on the Euclidean distance between embeddings.
Enforcement: back-off (Prop. 5.4) towards a feasible safe field: the nearest anchor for kNN1 scores (score 0), the
nearest feasible anchor or calibration field otherwise; for this prompt-affine model it is two-prompt mixing.
Certificate for a final field theta with nearest anchor i (Theorem 6.1, pointwise in the weights), for B = A:
  L^p, p >= 1:   P(Y in A) <= q_i(A^{+r}) + b / r            (Markov),         b = budget of the region
  truncated r0:  P(Y in A) <= q_i(A^{+r}) + b max(1, r0 / r)  (no Markov for r >= r0)
  tail r0:       P(Y in A) <= q_i(A^{+r}) + b, r >= r0
Also a white-box attack: random search over embeddings for the largest P(A) among fields the region admits.
Usage: python e1_phase2.py [n_splits]     -> ../results/e1/phase2.json
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time
import numpy as np
from joblib import Parallel, delayed
from e1_common import Features, fields, simulate, one_step, mmd2, kth_threshold, r_A
from e1_eval import D, CFG, Z, E, NS, NP, M, K, BETA, split, d_eucl, RES, feat, x_probe, x_eval, WST, JOF, MASS

R_GRID = np.linspace(0.05, 2.0, 40)
METHODS = {  # name: (space, distance kind, parameter, aggregation)
    "U-L2": ("T", "lp", 2, "U"), "kNN1-L1": ("T", "lp", 1, "kNN1"), "kNN1-tr1": ("T", "trunc", 1.0, "kNN1"),
    "kNN1-tl.5": ("T", "tail", 0.5, "kNN1"), "E-U": ("E", None, None, "U"), "E-kNN1": ("E", None, None, "kNN1"),
}


def pdist(Ta, Tb, kind, par):
    """(na, L, 2) x (nb, L, 2) -> (na, nb) for the map distances; a row block at a time."""
    out = np.empty((len(Ta), len(Tb)))
    for i0 in range(0, len(Ta), 64):
        nrm = np.sqrt(((Ta[i0:i0 + 64, None] - Tb[None]) ** 2).sum(-1))
        out[i0:i0 + 64] = ((nrm ** par).mean(-1) ** (1 / par) if kind == "lp" else
                           np.minimum(nrm, par).mean(-1) / par if kind == "trunc" else (nrm >= par).mean(-1))
    return out


def agg(Dm, how):
    return Dm.mean(1) if how == "U" else Dm.min(1)


def outputs(Wf, cfg, x0):
    return simulate(feat, Wf[None], x0, JOF[cfg])[0] if JOF[cfg] > 1 else one_step(feat, Wf[None], x0)[0]


class Region:
    """Score, threshold and nearest anchor of one method on one split."""
    def __init__(self, name, cfg, anc, cal):
        self.name, self.cfg, self.anc = name, cfg, anc
        self.space, self.kind, self.par, self.how = METHODS[name]
        self.Tanc = D[f"{cfg}_Tp"][anc].astype(float); self.Eanc = E[anc]
        self.eps = kth_threshold(self.scores_pool(cal), BETA)

    def dist_T(self, T):            # T: (n, L, 2) probe outputs
        return pdist(T, self.Tanc, self.kind, self.par)

    def dist_pool(self, idx):
        return self.dist_T(D[f"{self.cfg}_Tp"][idx].astype(float)) if self.space == "T" else d_eucl(E[idx], self.Eanc)

    def scores_pool(self, idx):
        return agg(self.dist_pool(idx), self.how)

    def score_field(self, Wf, e=None):
        if self.space == "E":
            return agg(d_eucl(e[None], self.Eanc), self.how)[0]
        return agg(self.dist_T(outputs(Wf, self.cfg, x_probe)[None]), self.how)[0]


def enforce_one(reg, j, feas_W, feas_E, feas_T):
    """Back-off of pool prompt j towards the nearest feasible safe field (two-prompt mixing for this affine model)."""
    Ws_j = fields(WST[reg.cfg], E[[j]])[0]; e_j = E[j]
    s0 = reg.scores_pool(np.array([j]))[0]
    if reg.space == "T":
        dnear = pdist(D[f"{reg.cfg}_Tp"][[j]].astype(float), feas_T, reg.kind, reg.par)[0]
    else:
        dnear = d_eucl(e_j[None], feas_E)[0]
    c = int(np.argmin(dnear)); Wc, ec = feas_W[c], feas_E[c]
    kap = 0.0
    if s0 > reg.eps:
        lo, hi = 0.0, 1.0
        for _ in range(18):
            mid = (lo + hi) / 2
            if reg.score_field((1 - mid) * Ws_j + mid * Wc, (1 - mid) * e_j + mid * ec) <= reg.eps: hi = mid
            else: lo = mid
        kap = hi
    Wf = (1 - kap) * Ws_j + kap * Wc
    return Wf, kap, s0, c


def certify(reg, Wf, Te_anc, q_anc):
    """Outputs of a final field on fresh noise, its P(A), and the certificate of its method (relative to its nearest anchor)."""
    Te = outputs(Wf, reg.cfg, x_eval); Tp = outputs(Wf, reg.cfg, x_probe)
    pA = float(np.mean(np.linalg.norm(Te, axis=1) <= r_A))
    if reg.space == "E":
        return Te, pA, dict()
    dp = reg.dist_T(Tp[None])[0]
    i = int(np.argmin(dp)) if reg.how == "kNN1" else None
    nrm = np.linalg.norm(Te[None] - Te_anc, axis=-1)                      # (M, N) fresh-noise distances to the anchors
    b = reg.eps
    if reg.how == "U":                                                    # uniform mixture of the anchors
        q, fresh = q_anc.mean(0), float(nrm.mean())                       # L^2 budget b dominates the L^1 coupling cost
        bound = np.min(q + b / R_GRID)
    else:
        q = q_anc[i]
        if reg.kind == "lp":
            fresh = float(nrm[i].mean()); bound = np.min(q + b / R_GRID)
        elif reg.kind == "trunc":
            fresh = float(np.minimum(nrm[i], reg.par).mean() / reg.par); bound = np.min(q + b * np.maximum(1, reg.par / R_GRID))
        else:
            fresh = float(np.mean(nrm[i] >= reg.par)); bound = np.min(np.where(R_GRID >= reg.par, q + b, np.inf))
    near = nrm if reg.how == "U" else nrm[[i]]                           # coupling distances actually certified
    tail_emp = np.min(q + np.array([np.mean(near >= r) for r in R_GRID]))  # a-posteriori bound, empirical tail
    return Te, pA, dict(bound=float(bound), budget=float(b), fresh=fresh, tail_emp=float(tail_emp))


def run_method(name, cfg, s, cand):
    anc, cal, te = split(1000 + s)
    reg = Region(name, cfg, anc, cal)
    sc_te, sc_un = reg.scores_pool(te), reg.scores_pool(np.arange(NS, NP))
    out = dict(method=name, cfg=cfg, split=s, eps=float(reg.eps), coverage=float(np.mean(sc_te <= reg.eps)))
    for m in CFG["masses"]:
        out[f"excl_{m}"] = float(np.mean(sc_un[Z[NS:, 2] == m] > reg.eps))
    # feasible targets of the back-off: anchors (and calibration fields) whose own score is at most eps
    pool_feas = np.r_[anc, cal]
    keep = reg.scores_pool(pool_feas) <= reg.eps
    feas = pool_feas[keep]
    feas_W = fields(WST[cfg], E[feas]); feas_E = E[feas]; feas_T = D[f"{cfg}_Tp"][feas].astype(float)
    Te_anc = D[f"{cfg}_Te"][anc].astype(float)
    q_anc = np.array([[np.mean(np.linalg.norm(Te_anc[i], axis=1) < r_A + r) for r in R_GRID] for i in range(M)])
    rows = []
    for j in cand:
        Wf, kap, s0, _ = enforce_one(reg, int(j), feas_W, feas_E, feas_T)
        Te, pA, cert = certify(reg, Wf, Te_anc, q_anc)
        rows.append(dict(j=int(j), m=float(MASS[j]), safe=bool(j < NS), kappa=float(kap), inside=bool(s0 <= reg.eps),
                         pA=pA, pA0=float(D[f"{cfg}_pA"][j]), mmd=float(mmd2(Te[:1000], D["true_safe_samples"][j]) * 1e3), **cert))
    out["rows"] = rows
    return out


# ------------------------------------------------------------------ white-box attack over embeddings
def attack(name, cfg, s, n_starts=12, iters=300, seed=0):
    """Random search over embeddings for the largest P(A) among fields admitted by the region (refusal defence).
    Starts from back-off results of m = 0.3 prompts (feasible); proposals are Gaussian steps in embedding space."""
    anc, cal, te = split(1000 + s)
    reg = Region(name, cfg, anc, cal)
    rng = np.random.default_rng(seed); x_att = x_eval[:600]
    sd = E.std(0)                                                        # per-coordinate scale of all pool embeddings
    pool_feas = np.r_[anc, cal]; feas = pool_feas[reg.scores_pool(pool_feas) <= reg.eps]
    feas_W, feas_E, feas_T = fields(WST[cfg], E[feas]), E[feas], D[f"{cfg}_Tp"][feas].astype(float)
    starts = rng.choice(np.arange(NS, NP)[Z[NS:, 2] == 0.3], n_starts, replace=False)
    best = []
    for j in starts:
        Wf, kap, _, c = enforce_one(reg, int(j), feas_W, feas_E, feas_T)
        e = (1 - kap) * E[j] + kap * feas_E[c]                          # the same two-prompt mixture, in embedding space
        W = fields(WST[cfg], e[None])[0]
        if reg.score_field(W, e) > reg.eps:                              # numerical safety: fall back to the back-off field
            W = Wf
        cur = np.mean(np.linalg.norm(outputs(W, cfg, x_att), axis=1) <= r_A); step = 0.3
        for it in range(iters):
            e2 = e + step * sd * rng.standard_normal(len(e)); W2 = fields(WST[cfg], e2[None])[0]
            if reg.score_field(W2, e2) > reg.eps:
                step *= 0.97; continue
            val = np.mean(np.linalg.norm(outputs(W2, cfg, x_att), axis=1) <= r_A)
            if val >= cur:
                e, W, cur = e2, W2, val; step *= 1.05
        pA_fresh = float(np.mean(np.linalg.norm(outputs(W, cfg, x_eval[600:]), axis=1) <= r_A))
        best.append(dict(start=int(j), pA_search=float(cur), pA_fresh=pA_fresh, score=float(reg.score_field(W, e) / reg.eps)))
    return dict(method=name, cfg=cfg, split=s, eps=float(reg.eps), runs=best)


def main():
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    cand_unsafe = np.arange(NS, NP)
    jobs = []
    for s in range(n_splits):
        anc, cal, te = split(1000 + s)
        cand = np.r_[te[:80], cand_unsafe]
        jobs += [(name, cfg, s, cand) for cfg in ["rect1", "rect2"] for name in METHODS]
    res = Parallel(n_jobs=14)(delayed(run_method)(*a) for a in jobs)
    print(f"[{time.time()-t0:.0f}s] enforcement done", flush=True)
    att = Parallel(n_jobs=14)(delayed(attack)(name, cfg, 0) for cfg in ["rect1", "rect2"] for name in METHODS)
    print(f"[{time.time()-t0:.0f}s] attacks done", flush=True)
    json.dump(dict(enforce=res, attack=att), open(os.path.join(RES, "phase2.json"), "w"))


if __name__ == "__main__":
    main()
