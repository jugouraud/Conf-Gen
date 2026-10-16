"""E1, Step 7 made rigorous (safe_prompt_fields_audit.tex, finding R3): a field is deployed only if its distance to the nearest
safe field is certified on fresh noise.

Methods: the improved region (nearest safe field, truncated L^1 distance d_1, Proposition 7.5) and the tail score, on the
1-rectified and the 2-rectified model (50 Euler steps), splits 0-4 of e1_phase2.py, the first 80 safe test prompts and the
unsafe prompts with m in {0.1, 0.3}. The region and its budget b (the conformal threshold, on the frozen probes) are unchanged.
The nearest anchor i* is always chosen on the frozen probes, and a "check" of a field computes an upper confidence bound at level
1 - delta_d / 2 for its distance to i* on L = 2000 fresh noises (Hoeffding for d_1, which has values in [0, 1]; Clopper-Pearson
for the tail, a proportion) and passes if the bound is at most b.
  1. If the request is admitted on the probes (score <= b), check its own field; deploy it unchanged if the check passes.
  2. Otherwise back off (bisection on the frozen probes towards the nearest feasible safe field, as in e1_phase2.enforce_one)
     until the probe score is at most a reduced budget b' = b - u(b) - 2 sqrt(b (1 - b) / L_probe), where u(b) is the margin
     of the check at the budget and the last term is two standard errors of a [0, 1]-valued probe average of mean b
     (Bhatia-Davis inequality); check the backed-off field; deploy it if the check passes, else deploy the anchor i* itself.
  At most two checks are made per request, so the deployed field is within b of its anchor with probability >= 1 - delta_d.
  3. Certificate: P(Y in A) <= q_{i*}(r) + b max(1, 1/r) (truncated, any r of the grid) or q_{i*}(r) + b (tail, r >= 0.5), with
     q_{i*}(r) = P(|Y_{i*}| < r_A + r) replaced by a Clopper-Pearson upper bound at level 1 - delta_q / |grid| computed on 10,000
     further noises. It holds with probability at least 1 - delta_d - delta_q.
Evaluation on the evaluation noise of e1_fit.py (independent of the probes and of the noise above): P(A) and MMD of the deployed
field; and the distance to i* on the 10,000 further noises, against the budget b.
Writes ../results/e1/verify.json.     Usage: python e1_verify.py
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
from scipy.stats import beta as betad
from joblib import Parallel, delayed
import e1_phase2 as P
from e1_common import fields, mmd2, r_A

D, Z, E, NS, NP, M = P.D, P.Z, P.E, P.NS, P.NP, P.M
DELTA_D, DELTA_Q = 0.025, 0.025
DELTA_CHECK = DELTA_D / 2                                       # two checks per request at most
L_FRESH, L_BIG = 2000, 10000
X_FRESH = np.random.default_rng(31).standard_normal((L_FRESH, 2))
X_BIG = np.random.default_rng(32).standard_normal((L_BIG, 2))
METHODS, CFGS, N_SPLITS = ["kNN1-tr1", "kNN1-tl.5"], ["rect1", "rect2"], 5


def cp_upper(k, n, delta):
    """Clopper-Pearson upper confidence bound at level 1 - delta for k successes out of n."""
    return 1.0 if k >= n else float(betad.ppf(1 - delta, k + 1, n - k))


def margin(kind, b):
    """Margin of a check at the budget: Hoeffding for d_1, Clopper-Pearson for the tail."""
    if kind == "trunc":
        return float(np.sqrt(np.log(1 / DELTA_CHECK) / (2 * L_FRESH)))
    k = int(np.floor(b * L_FRESH))
    return cp_upper(k, L_FRESH, DELTA_CHECK) - k / L_FRESH


def score(kind, par, Ya, Yb):
    nrm = np.linalg.norm(Ya - Yb, axis=-1)
    return float(np.minimum(nrm, par).mean() / par) if kind == "trunc" else float(np.mean(nrm >= par))


def ucb(kind, par, Ya, Yb):
    nrm = np.linalg.norm(Ya - Yb, axis=-1)
    if kind == "trunc":
        return float(np.minimum(nrm, par).mean() / par + np.sqrt(np.log(1 / DELTA_CHECK) / (2 * len(nrm))))
    return cp_upper(int(np.sum(nrm >= par)), len(nrm), DELTA_CHECK)


def run(name, cfg, s, limit=None):
    anc, cal, te = P.split(1000 + s)
    reg = P.Region(name, cfg, anc, cal); kind, par, b = reg.kind, reg.par, float(reg.eps)
    bprime = max(b - margin(kind, b) - 2 * np.sqrt(b * (1 - b) / len(P.x_probe)), 0.0)
    pool = np.r_[anc, cal]; feas = pool[reg.scores_pool(pool) <= bprime]                 # back-off targets (anchors included)
    feas_W, feas_T = fields(P.WST[cfg], E[feas]), D[f"{cfg}_Tp"][feas].astype(float)
    W_anc = fields(P.WST[cfg], E[anc])
    Tf_anc = np.stack([P.outputs(W, cfg, X_FRESH) for W in W_anc])
    Tb_anc = np.stack([P.outputs(W, cfg, X_BIG) for W in W_anc])
    nb = np.linalg.norm(Tb_anc, axis=-1)
    dq = DELTA_Q / len(P.R_GRID)
    q_up = np.array([[cp_upper(int(np.sum(nb[i] < r_A + r)), L_BIG, dq) for r in P.R_GRID] for i in range(M)])

    def check(W):
        i = int(np.argmin(reg.dist_T(P.outputs(W, cfg, P.x_probe)[None])[0]))           # nearest anchor, frozen probes
        return ucb(kind, par, P.outputs(W, cfg, X_FRESH), Tf_anc[i]), i

    cand = np.r_[te[:80], np.arange(NS, NP)[np.isin(Z[NS:, 2], [0.1, 0.3])]][:limit]
    rows = []
    for j in cand:
        Wj = fields(P.WST[cfg], E[[j]])[0]
        s0 = float(reg.scores_pool(np.array([j]))[0]); kap, W, stage, u = 0.0, Wj, "fallback", np.nan
        if s0 <= b:
            u, i = check(Wj)
            if u <= b:
                stage = "as is"
        if stage != "as is":
            if s0 > bprime:
                c = int(np.argmin(P.pdist(D[f"{cfg}_Tp"][[j]].astype(float), feas_T, kind, par)[0]))
                lo, hi = 0.0, 1.0
                for _ in range(18):
                    mid = (lo + hi) / 2
                    if reg.score_field((1 - mid) * Wj + mid * feas_W[c]) <= bprime: hi = mid
                    else: lo = mid
                kap = hi; W = (1 - hi) * Wj + hi * feas_W[c]
            u, i = check(W)
            if u <= b:
                stage = "backed off"
            else:
                W, kap = W_anc[i], 1.0
        Te = P.outputs(W, cfg, P.x_eval)
        bound = (np.min(q_up[i] + b * np.maximum(1, par / P.R_GRID)) if kind == "trunc"
                 else np.min(np.where(P.R_GRID >= par, q_up[i] + b, np.inf)))
        rows.append(dict(j=int(j), m=float(P.MASS[j]), safe=bool(j < NS), inside=bool(s0 <= b), stage=stage, kappa=float(kap),
                         ucb=float(u), d_big=score(kind, par, P.outputs(W, cfg, X_BIG), Tb_anc[i]),
                         pA=float(np.mean(np.linalg.norm(Te, axis=1) <= r_A)), bound=float(bound),
                         mmd=float(mmd2(Te[:1000], D["true_safe_samples"][j]) * 1e3)))
    return dict(method=name, cfg=cfg, split=s, b=b, bprime=float(bprime), margin=margin(kind, b), n_feas=int(len(feas)), rows=rows)


def main():
    t0 = time.time()
    res = Parallel(n_jobs=16)(delayed(run)(n, c, s) for n in METHODS for c in CFGS for s in range(N_SPLITS))
    json.dump(dict(delta_d=DELTA_D, delta_q=DELTA_Q, L_fresh=L_FRESH, L_big=L_BIG, runs=res),
              open(os.path.join(P.RES, "verify.json"), "w"), indent=0)
    for n in METHODS:
        for c in CFGS:
            B = [b for b in res if b["method"] == n and b["cfg"] == c]; R = [r for b in B for r in b["rows"]]
            saf = [r for r in R if r["safe"]]; u3 = [r for r in R if r["m"] == 0.3]; u1 = [r for r in R if r["m"] == 0.1]
            over = np.mean([r["d_big"] > b["b"] for b in B for r in b["rows"]])
            print(f"{n:10s} {c:6s} b {np.median([b['b'] for b in B]):.3f} b' {np.median([b['bprime'] for b in B]):.3f} | "
                  f"safe altered {np.mean([r['stage'] != 'as is' for r in saf]):.2f} (inside {np.mean([r['inside'] for r in saf]):.2f}) | "
                  f"fallback m=0.3 {np.mean([r['stage'] == 'fallback' for r in u3]):.2f} | P(A) m=0.1 "
                  f"{np.mean([r['pA'] for r in u1]):.3f} m=0.3 {np.mean([r['pA'] for r in u3]):.3f} | MMD m=0.3 "
                  f"{np.mean([r['mmd'] for r in u3]):.0f} | over budget (10k noises) {over:.4f} | bound median "
                  f"{np.median([r['bound'] for r in R]):.3f} | violations {sum(r['pA'] > r['bound'] for r in R)}/{len(R)}")
    print(f"[{time.time()-t0:.0f}s] saved verify.json")


if __name__ == "__main__":
    main()
