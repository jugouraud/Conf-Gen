"""E2 robustness checks (safe_prompt_fields_audit.tex, findings F1-F3), on the pool and models of e2_pool.py; nothing is refitted.

(1) Probe sensitivity of the field regions of the 1-rectified model (L^2 and truncated L^1 map distances, static and nearest-
    anchor weights), with and without the probes that some pool prompt's sampler sends beyond the trimming radius 4.
(2) Attack 2 on the pull-back, field and Mahalanobis regions of both deployed models: random search in the 5 directions in
    which the region's own distance grows slowest, i.e. the eigenvectors with the smallest eigenvalues of its metric Gram
    matrix: Q = U^T G_0 U for a pull-back metric, J^T J for a field distance (J: Jacobian of the probe outputs at the start)
    and (S + d I)^{-1} for the Mahalanobis metric; from the first 12 admitted approved prompts of evaluation split 0 (anchors,
    then calibration prompts), 400 steps each, P(A) on fresh noise. The Euclidean prompt metric has no preferred direction.
(3) Mahalanobis prompt regions: the distance |e - e'| weighted by (S + d I)^{-1}, with S the covariance of the approved
    embeddings of the TUNING pool and d = rho tr(S) / D_e. rho and the adaptive weights are tuned on the tuning pool (50 splits,
    mean exclusion over the four masses) as for the regions of e2_regions.py; then coverage and exclusion (200 evaluation
    splits), enforcement by back-off (3 splits), the attack of e2_enforce_attack.py, attack 2, and the nuisance-token test of
    e2_nuisance.py (covariance and tuning redone for every token scale lambda).
Writes ../results/e2/robustness.json.    Usage: python e2_robustness.py
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
from joblib import Parallel, delayed
import e2_regions as R
import e2_enforce_attack as A
import e2_nuisance as N
from e1_common import kth_threshold, mmd2, sample_target, r_A

R_TRIM = 4.0
RHOS = [1e-6, 1e-4, 1e-2, 1.0]
BEST = json.load(open(os.path.join(R.RES, "regions.json")))["best"]
CASES = {"rect1": ["PB:rect1", "TR:rect1", "MH"], "rect2_1step": ["PB:rect2", "TR:rect2_1step", "MH"]}


def summarise(rows):
    r = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if k != "eps"}
    r["eps_median"] = float(np.median([x["eps"] for x in rows]))
    return r


def evaluate_many(DS, space, v, pool, uns, seeds):
    return summarise([R.evaluate(DS, space, v, *R.split(pool, 30, s), uns) for s in seeds])


# ------------------------------------------------------------------ (1) probe sensitivity
def probe_part():
    T = R.D["rect1_Tp"].astype(float)                                        # pool prompts only
    nrm = np.linalg.norm(T, axis=-1)
    f = np.where((nrm > R_TRIM).any(0))[0]
    keep = np.ones(T.shape[1], bool); keep[f] = False
    out = {"flagged": f.tolist(), "maxnorm_pool": float(nrm.max()),
           "maxnorm_grid": float(np.linalg.norm(R.D["rect1_Tpg"].astype(float), axis=-1).max()), "regions": {}}
    DS = {}
    for tag, kp in [("all", np.ones(T.shape[1], bool)), ("drop", keep)]:
        for kind in ["L2", "TR"]:
            DS[f"{kind}|{tag}"] = R.map_dist(T[:, kp], T[:R.NS, kp], kind)
    for key in DS:
        for v in [("U", None), ("N", 1)]:
            out["regions"][f"{key}|{R.vname(v)}"] = evaluate_many(DS, key, v, R.SAFE_EVAL, R.UNS_EVAL, [1000 + s for s in range(200)])
    return out


# ------------------------------------------------------------------ (3) Mahalanobis prompt regions
def whitener(Et, rho):
    """W with |(e - e') W| = Mahalanobis distance for the covariance of the rows of Et, regularised by rho tr(S) / D_e."""
    S = np.cov(Et.T); lam, V = np.linalg.eigh(S)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(lam, 0.0) + rho * np.trace(S) / len(S))) @ V.T


def mh_dist(Eall, Wm):
    Zw = Eall @ Wm
    return R.eucl(Zw, Zw[:R.NS]).astype(np.float32)


def tune(Eall):
    """Tuning of rho (static) and of (rho, adaptive weights) on the tuning pool; evaluation of both on the evaluation pool."""
    Dm, J = {}, {}
    for rho in RHOS:
        Dm[rho] = mh_dist(Eall, whitener(Eall[R.SAFE_TUNE], rho))
        for v in R.variants("E"):
            J[(rho, R.vname(v))] = evaluate_many({"MH": Dm[rho]}, "MH", v, R.SAFE_TUNE, R.UNS_TUNE, [5000 + s for s in range(50)])["J"]
    st = max(RHOS, key=lambda r: J[(r, "U")])
    ad = max([k for k in J if k[1] != "U"], key=lambda k: J[k])
    out = {"tune": {f"{k[0]:g}|{k[1]}": float(v) for k, v in J.items()}}
    for tag, (rho, vn) in [("static", (st, "U")), ("adaptive", ad)]:
        out[tag] = dict(rho=rho, variant=vn, **evaluate_many({"MH": Dm[rho]}, "MH", A.parse(vn), R.SAFE_EVAL, R.UNS_EVAL,
                                                             [1000 + s for s in range(200)]))
    return out


class MHRegion:
    """A Mahalanobis prompt region of one split, with a score for any embedding."""
    def __init__(self, rho, vn, anc, cal):
        self.Wm = whitener(R.E[R.SAFE_TUNE], rho); self.D = mh_dist(R.E, self.Wm)
        self.v, self.anc = A.parse(vn), anc
        self.Za = R.E[anc] @ self.Wm
        self.Daa = self.D[np.ix_(anc, anc)].astype(float)
        self.eps = kth_threshold(self.scores_idx(cal), R.BETA)

    def scores_idx(self, idx):
        return R.score(self.D[np.ix_(idx, self.anc)], self.v, self.Daa)

    def score(self, e):
        return float(R.score(np.linalg.norm(self.Za - e @ self.Wm, axis=1)[None], self.v, self.Daa)[0])

    def feasible(self, cal):
        pool = np.r_[self.anc, cal]
        return pool[self.scores_idx(pool) <= self.eps]


def mh_backoff(reg, j, feas):
    """e2_enforce_attack.backoff for the Mahalanobis region."""
    e = R.E[j]
    if reg.score(e) <= reg.eps:
        return e, 0.0
    c = feas[np.argmin(np.linalg.norm((R.E[feas] - e) @ reg.Wm, axis=1))]
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if reg.score((1 - mid) * e + mid * R.E[c]) <= reg.eps: hi = mid
        else: lo = mid
    return (1 - hi) * e + hi * R.E[c], hi


def mh_enforce(cfg, tag, rho, vn, s):
    """e2_enforce_attack.enforce_job for the Mahalanobis region."""
    anc, cal, te = R.split(R.SAFE_EVAL, 30, 1000 + s)
    reg = MHRegion(rho, vn, anc, cal); feas = reg.feasible(cal)
    rng = np.random.default_rng(7 + s); rows = []
    for j in np.r_[te[:40], R.UNS_EVAL[R.Z[R.UNS_EVAL, 2] >= 0.1]]:
        ef, kap = mh_backoff(reg, int(j), feas)
        Y = A.outputs(ef, cfg, A.X_EVAL); z0 = np.r_[R.Z[j, :2], 0.0]
        rows.append(dict(j=int(j), m=float(R.Z[j, 2]), kappa=float(kap), pA=float(np.mean(np.linalg.norm(Y, axis=1) <= r_A)),
                         mmd=float(mmd2(Y[:1000], sample_target(z0, 1000, rng)) * 1e3)))
    return dict(cfg=cfg, weights=tag, split=s, eps=float(reg.eps), rows=rows)


def mh_attack(cfg, tag, rho, vn, n_starts=12, iters=300, seed=0):
    """e2_enforce_attack.attack_job for the Mahalanobis region (same starts and proposals)."""
    anc, cal, te = R.split(R.SAFE_EVAL, 30, 1000)
    reg = MHRegion(rho, vn, anc, cal); feas = reg.feasible(cal)
    rng = np.random.default_rng(seed); sd = R.E.std(0)
    starts = rng.choice(R.UNS_EVAL[R.Z[R.UNS_EVAL, 2] == 0.3], n_starts, replace=False)
    runs = []
    for j in starts:
        e, _ = mh_backoff(reg, int(j), feas)
        cur = np.mean(np.linalg.norm(A.outputs(e, cfg, A.X_EVAL[:600]), axis=1) <= r_A); step = 0.3
        for _ in range(iters):
            e2 = e + step * sd * rng.standard_normal(len(e))
            if reg.score(e2) > reg.eps:
                step *= 0.97; continue
            val = np.mean(np.linalg.norm(A.outputs(e2, cfg, A.X_EVAL[:600]), axis=1) <= r_A)
            if val >= cur:
                e, cur = e2, val; step *= 1.05
        runs.append(dict(start=int(j), pA_search=float(cur),
                         pA_fresh=float(np.mean(np.linalg.norm(A.outputs(e, cfg, A.X_EVAL[600:]), axis=1) <= r_A))))
    return dict(cfg=cfg, space="MH", weights=tag, eps=float(reg.eps), runs=runs)


# ------------------------------------------------------------------ (2) attack 2
def make_region(space, vn, anc, cal, rho=None):
    return MHRegion(rho, vn, anc, cal) if space == "MH" else A.Region(space, vn, anc, cal)


def cheap_dirs(space, reg, e, k=5, h=1e-3):
    """The k embedding directions in which the region's distance grows slowest at e."""
    if space == "MH":
        G = reg.Wm @ reg.Wm.T
    elif space.startswith("PB"):
        C = A.WHIT[space.split(":")[1]]; G = C @ C.T
    else:
        cols = []
        for jj in range(len(e)):
            d = np.zeros(len(e)); d[jj] = h
            cols.append(((A.outputs(e + d, reg.cfg, A.X_PROBE) - A.outputs(e - d, reg.cfg, A.X_PROBE)) / (2 * h)).ravel())
        J = np.stack(cols, 1); G = J.T @ J
    lam, V = np.linalg.eigh(G)
    return V[:, :k], lam[:k]


def attack2_run(cfg, space, tag, vn, start, seed, rho=None, k=5, iters=400):
    anc, cal, te = R.split(R.SAFE_EVAL, 30, 1000)
    reg = make_region(space, vn, anc, cal, rho)
    pA = lambda e, x: float(np.mean(np.linalg.norm(A.outputs(e, cfg, x), axis=1) <= r_A))
    x_att, x_fr = A.X_EVAL[:600], A.X_EVAL[600:]
    e0 = R.E[start].copy()
    if reg.score(e0) > reg.eps:
        raise RuntimeError("starting prompt not admitted")
    V, lam = cheap_dirs(space, reg, e0, k)
    rng = np.random.default_rng(seed)
    e, cur, step = e0, pA(e0, x_att), 1.0
    for _ in range(iters):
        e2 = e + step * V @ rng.standard_normal(k)
        if reg.score(e2) > reg.eps:
            step *= 0.9; continue
        val = pA(e2, x_att)
        if val >= cur:
            e, cur = e2, val; step *= 1.1
    return dict(cfg=cfg, space=space, weights=tag, variant=vn, start=int(start), seed=int(seed), pA_start=pA(e0, x_fr),
                pA_search=float(cur), pA_fresh=pA(e, x_fr), score=float(reg.score(e) / reg.eps), dE=float(np.linalg.norm(e - e0)),
                eig=[float(x) for x in lam])


def attack2_jobs(mh):
    anc, cal, te = R.split(R.SAFE_EVAL, 30, 1000)
    jobs, idx = [], 0
    for cfg, spaces in CASES.items():
        for space in spaces:
            for tag in ["static", "adaptive"]:
                if space == "MH":
                    rho, vn = mh[tag]["rho"], mh[tag]["variant"]
                else:
                    rho, vn = None, ("U" if tag == "static" else BEST[f"{space}|30"])
                reg = make_region(space, vn, anc, cal, rho)
                adm = (reg.feasible(cal) if space == "MH" else A.feasible(reg, anc, cal))[:12]
                jobs += [(cfg, space, tag, vn, int(j), 9000 + 100 * idx + i, rho) for i, j in enumerate(adm)]
                idx += 1
    return jobs


def nuisance_part():
    s_pool = np.random.default_rng(13).uniform(-1, 1, len(R.Z))              # the tokens of e2_nuisance.main
    Nn = N.nuis(s_pool)
    return {f"{lam:g}": tune(R.E if lam == 0 else np.c_[R.E, lam * Nn]) for lam in [0.0] + N.LAMS}


def main():
    t0 = time.time()
    out = {"probe": probe_part()}
    print(f"[{time.time()-t0:.0f}s] probe sensitivity: flagged {out['probe']['flagged']}", flush=True)
    for cfg in ["rect1", "rect2"]:
        C = A.WHIT[cfg]; lam = np.linalg.eigvalsh(C @ C.T)
        out[f"Q_{cfg}"] = dict(min=float(lam.min()), max=float(lam.max()), cond=float(lam.max() / lam.min()))
    out["mh"] = tune(R.E)
    print(f"[{time.time()-t0:.0f}s] Mahalanobis regions tuned: static rho {out['mh']['static']['rho']:g}, adaptive "
          f"{out['mh']['adaptive']['rho']:g} {out['mh']['adaptive']['variant']}", flush=True)
    mh = out["mh"]
    cells = [(cfg, tag, mh[tag]["rho"], mh[tag]["variant"]) for cfg in CASES for tag in ["static", "adaptive"]]
    out["mh_enforce"] = Parallel(n_jobs=12)(delayed(mh_enforce)(*c, s) for c in cells for s in range(3))
    out["mh_attack"] = Parallel(n_jobs=4)(delayed(mh_attack)(*c) for c in cells)
    print(f"[{time.time()-t0:.0f}s] Mahalanobis enforcement and attack", flush=True)
    jobs = attack2_jobs(mh)
    out["attack2"] = Parallel(n_jobs=12)(delayed(attack2_run)(*j) for j in jobs)
    print(f"[{time.time()-t0:.0f}s] attack 2 ({len(jobs)} runs)", flush=True)
    out["mh_nuisance"] = nuisance_part()
    print(f"[{time.time()-t0:.0f}s] nuisance", flush=True)
    json.dump(out, open(os.path.join(R.RES, "robustness.json"), "w"), indent=0)
    # ---- summary
    for k, r in out["probe"]["regions"].items():
        print(f"probe {k:16s} cov {r['coverage']:.3f} excl " + " ".join(f"{r[f'excl_{m}']:.2f}" for m in R.MASSES))
    for tag in ["static", "adaptive"]:
        r = mh[tag]
        print(f"MH {tag:8s} rho {r['rho']:g} {r['variant']:6s}: cov {r['coverage']:.3f} excl " +
              " ".join(f"{r[f'excl_{m}']:.2f}" for m in R.MASSES))
    for b in out["mh_attack"]:
        v = [x["pA_fresh"] for x in b["runs"]]
        rows = [x for B in out["mh_enforce"] if (B["cfg"], B["weights"]) == (b["cfg"], b["weights"]) for x in B["rows"]]
        print(f"MH {b['cfg']:12s} {b['weights']:8s}: enforced P(A) m=0.3 {np.mean([x['pA'] for x in rows if x['m'] == 0.3]):.3f} "
              f"MMD {np.mean([x['mmd'] for x in rows if x['m'] == 0.3]):.0f} | attack worst {max(v):.3f}")
    agg = {}
    for r in out["attack2"]:
        agg.setdefault((r["cfg"], r["space"], r["weights"]), []).append(r["pA_fresh"])
    for k, v in agg.items():
        print(f"attack 2 {k[0]:12s} {k[1]:15s} {k[2]:8s}: worst {max(v):.3f} median {np.median(v):.3f} (n={len(v)})")
    for lam, r in out["mh_nuisance"].items():
        print(f"nuisance lam {lam}: MH static excl m=0.1 {r['static']['excl_0.1']:.2f} (rho {r['static']['rho']:g}), "
              f"adaptive {r['adaptive']['excl_0.1']:.2f} ({r['adaptive']['rho']:g} {r['adaptive']['variant']}), "
              f"coverage {r['static']['coverage']:.3f}/{r['adaptive']['coverage']:.3f}")
    print(f"[{time.time()-t0:.0f}s] saved robustness.json")


if __name__ == "__main__":
    main()
