"""E2, step 3: enforcement (back-off) and a white-box attack for the static and tuned adaptive regions on the prompt
embeddings and on the fields, for the 1-rectified model and for the reflowed one-step model.

Every region is evaluated for the model it protects ("deployed"): its outputs decide P(A).
  deployed rect1 (1-rectified, 50 Euler steps):   E (prompt), PB:rect1 (prompt, pull-back of its slice at t = 0),
                                                   TR:rect1 (field, truncated map distance);
  deployed rect2_1step (reflowed, one step):        E, PB:rect2 (prompt, pull-back of its one-step metric = its field
                                                   metric), TR:rect2_1step (field).
Each with uniform weights (static) and with the reweighting tuned in e2_regions.py (adaptive).
Enforcement: back-off of the prompt towards the nearest feasible safe prompt (anchor or calibration prompt whose own
score is at most the threshold), by bisection on the mixing weight (Proposition 6.4); for this model, affine in the
prompt, the mixed prompt has the mixed field (Corollary 6.3). Attack: random search over embeddings for the largest
P(A) among prompts admitted by the region, as in e1_phase2.py (12 starts from enforced m = 0.3 prompts, 300 steps).
Writes ../results/e2/enforce_attack.json.    Usage: python e2_enforce_attack.py [n_splits]
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time
import numpy as np
from joblib import Parallel, delayed
import e2_regions as R
from e1_common import Features, fields, simulate, one_step, sample_target, mmd2, kth_threshold, r_A

FIT = np.load(os.path.join(R.RES1, "fit.npz"))
WST = {"rect1": FIT["W1"], "rect2_1step": FIT["W2"]}
JOF = {"rect1": R.CFG1["J"], "rect2_1step": 1}
X_PROBE, X_EVAL = FIT["x_probe"], FIT["x_eval"]
CASES = {"rect1": ["E", "PB:rect1", "TR:rect1"], "rect2_1step": ["E", "PB:rect2", "TR:rect2_1step"]}
DS = R.distances()
WHIT = {cfg: R.pullback_whitener(cfg) for cfg in ["rect1", "rect2"]}


def parse(vn):
    if vn == "U":
        return ("U", None)
    k, p = vn.split(":")
    return (k, int(p) if k == "N" else float(p))


def outputs(e, cfg, x0):
    feat = Features(R.CFG1["F"], seed=R.CFG1["feat_seed"])
    W = fields(WST[cfg], e[None])
    return (simulate(feat, W, x0, JOF[cfg]) if JOF[cfg] > 1 else one_step(feat, W, x0))[0]


class Region:
    """A region of e2_regions.py for one split, with a score for any embedding (prompt mixtures included)."""
    def __init__(self, space, vn, anc, cal):
        self.space, self.v, self.anc = space, parse(vn), anc
        self.Daa = DS[space][np.ix_(anc, anc)]
        self.Eaa = DS["E"][np.ix_(anc, anc)]
        sc_cal = R.score(DS[space][np.ix_(cal, anc)], self.v, self.Daa, DS["E"][np.ix_(cal, anc)], self.Eaa)
        self.eps = kth_threshold(sc_cal, R.BETA)
        if space.startswith("PB"):
            self.C = WHIT[space.split(":")[1]]; self.Za = R.E[anc] @ self.C
        if space[:2] in ("L2", "TR"):
            self.cfg = space.split(":")[1]; self.Ta = R.D[f"{self.cfg}_Tp"][anc].astype(float)

    def dists(self, e):
        if self.space == "E":
            return np.linalg.norm(R.E[self.anc] - e, axis=1)
        if self.space.startswith("PB"):
            return np.linalg.norm(self.Za - e @ self.C, axis=1)
        nrm = np.linalg.norm(outputs(e, self.cfg, X_PROBE)[None] - self.Ta, axis=-1)
        return np.sqrt((nrm ** 2).mean(-1)) if self.space.startswith("L2") else np.minimum(nrm, 1.0).mean(-1)

    def score(self, e):
        de = np.linalg.norm(R.E[self.anc] - e, axis=1)
        return float(R.score(self.dists(e)[None], self.v, self.Daa, de[None], self.Eaa)[0])


def feasible(reg, anc, cal):
    pool = np.r_[anc, cal]
    sc = R.score(DS[reg.space][np.ix_(pool, anc)], reg.v, reg.Daa, DS["E"][np.ix_(pool, anc)], reg.Eaa)
    return pool[sc <= reg.eps]


def backoff(reg, j, feas):
    """Mix prompt j with the nearest feasible safe prompt (in the region's distance) until the score reaches the threshold."""
    e = R.E[j]
    if reg.score(e) <= reg.eps:
        return e, 0.0
    c = feas[np.argmin(DS[reg.space][j, feas])]
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if reg.score((1 - mid) * e + mid * R.E[c]) <= reg.eps: hi = mid
        else: lo = mid
    return (1 - hi) * e + hi * R.E[c], hi


def enforce_job(cfg, space, tag, vn, s, cand):
    anc, cal, te = R.split(R.SAFE_EVAL, 30, 1000 + s)
    reg = Region(space, vn, anc, cal); feas = feasible(reg, anc, cal)
    rng = np.random.default_rng(7 + s)
    rows = []
    for j in cand(te):
        ef, kap = backoff(reg, int(j), feas)
        Y = outputs(ef, cfg, X_EVAL)
        z0 = np.r_[R.Z[j, :2], 0.0]
        rows.append(dict(j=int(j), m=float(R.Z[j, 2]), kappa=float(kap), pA=float(np.mean(np.linalg.norm(Y, axis=1) <= r_A)),
                         mmd=float(mmd2(Y[:1000], sample_target(z0, 1000, rng)) * 1e3)))
    return dict(cfg=cfg, space=space, weights=tag, variant=vn, split=s, eps=float(reg.eps), rows=rows)


def attack_job(cfg, space, tag, vn, n_starts=12, iters=300, seed=0):
    anc, cal, te = R.split(R.SAFE_EVAL, 30, 1000)
    reg = Region(space, vn, anc, cal); feas = feasible(reg, anc, cal)
    rng = np.random.default_rng(seed); sd = R.E.std(0)
    starts = rng.choice(R.UNS_EVAL[R.Z[R.UNS_EVAL, 2] == 0.3], n_starts, replace=False)
    runs = []
    for j in starts:
        e, _ = backoff(reg, int(j), feas)
        cur = np.mean(np.linalg.norm(outputs(e, cfg, X_EVAL[:600]), axis=1) <= r_A); step = 0.3
        for it in range(iters):
            e2 = e + step * sd * rng.standard_normal(len(e))
            if reg.score(e2) > reg.eps:
                step *= 0.97; continue
            val = np.mean(np.linalg.norm(outputs(e2, cfg, X_EVAL[:600]), axis=1) <= r_A)
            if val >= cur:
                e, cur = e2, val; step *= 1.05
        pf = float(np.mean(np.linalg.norm(outputs(e, cfg, X_EVAL[600:]), axis=1) <= r_A))
        runs.append(dict(start=int(j), pA_search=float(cur), pA_fresh=pf, e=e.tolist()))
    return dict(cfg=cfg, space=space, weights=tag, variant=vn, eps=float(reg.eps), runs=runs)


def main():
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.time()
    best = json.load(open(os.path.join(R.RES, "regions.json")))["best"]
    cells = [(cfg, sp, tag, "U" if tag == "static" else best[f"{sp}|30"]) for cfg, sps in CASES.items() for sp in sps
             for tag in ["static", "adaptive"]]

    def cand(te):                                           # 40 safe test prompts and the evaluation prompts with m >= 0.1
        return np.r_[te[:40], R.UNS_EVAL[R.Z[R.UNS_EVAL, 2] >= 0.1]]
    enf = Parallel(n_jobs=12)(delayed(enforce_job)(*c, s, cand) for c in cells for s in range(n_splits))
    print(f"[{time.time()-t0:.0f}s] enforcement", flush=True)
    att = Parallel(n_jobs=12)(delayed(attack_job)(*c) for c in cells)
    print(f"[{time.time()-t0:.0f}s] attack", flush=True)
    json.dump(dict(enforce=enf, attack=att), open(os.path.join(R.RES, "enforce_attack.json"), "w"))
    for c in cells:
        B = [b for b in enf if (b["cfg"], b["space"], b["weights"]) == c[:3]]
        rows = [r for b in B for r in b["rows"]]
        A = [a for a in att if (a["cfg"], a["space"], a["weights"]) == c[:3]][0]
        pa = {m: np.mean([r["pA"] for r in rows if r["m"] == m]) for m in [0.1, 0.3]}
        alt = np.mean([r["kappa"] > 0 for r in rows if r["m"] == 0])
        print(f"{c[0]:12s} {c[1]:15s} {c[2]:8s} {c[3]:6s} | enforced P(A) m=0.1 {pa[0.1]:.3f} m=0.3 {pa[0.3]:.3f} | "
              f"MMD(m=0.3) {np.mean([r['mmd'] for r in rows if r['m'] == 0.3]):.0f} | safe altered {alt:.2f} | "
              f"attack max {max(r['pA_fresh'] for r in A['runs']):.3f} median {np.median([r['pA_fresh'] for r in A['runs']]):.3f}")
    print(f"[{time.time()-t0:.0f}s] saved enforce_attack.json")


if __name__ == "__main__":
    main()
