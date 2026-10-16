"""E1 robustness checks (safe_prompt_fields_audit.tex, findings F1 and F2), on the pool and models of e1_fit.py; nothing is
refitted.

(1) Probe sensitivity. Some prompts' 50-step samplers send a probe noise beyond radius 4, the radius beyond which the reflow
    pairs are trimmed (e1_fit.py, Remark 3.10). The screen of e1_improve.py is recomputed with and without the probes
    flagged by that rule, for the unbounded distances (L^p) and, for comparison, the bounded ones (truncated L^1, tail).
(2) Attack 2. The attack of e1_phase2.py proposes Gaussian steps in all 43 coordinates of the embedding. Attack 2 searches
    only the 5 directions that move the probe outputs least at its starting point (right singular vectors of the Jacobian
    of the probe outputs with respect to the embedding, smallest singular values first), from the first 12 admitted
    approved prompts of split 0 (anchors, then calibration prompts), 400 steps each; P(A) is reported on fresh noise.
Writes ../results/e1/robustness.json.     Usage: python e1_robustness.py
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
from joblib import Parallel, delayed
import e1_phase2 as P
from e1_common import fields, kth_threshold, r_A

D, Z, E, NS, NP = P.D, P.Z, P.E, P.NS, P.NP
R_TRIM = 4.0                                   # e1_fit.CFG["reflow_trim"]
CFGS = ["rect1", "rect2"]
DISTS = [("L1", "lp", 1), ("L2", "lp", 2), ("L4", "lp", 4), ("L8", "lp", 8), ("trunc1", "trunc", 1.0), ("tail0.5", "tail", 0.5)]
ATTACK_REGIONS = ["U-L2", "kNN1-L1", "kNN1-tr1", "kNN1-tl.5"]


# ------------------------------------------------------------------ (1) probe sensitivity
def flagged(cfg):
    """Probes that some pool prompt's sampler sends beyond the trimming radius."""
    nrm = np.linalg.norm(D[f"{cfg}_Tp"].astype(float), axis=-1)
    return np.where((nrm > R_TRIM).any(0))[0]


def screen(cfg, keep, n_splits=100):
    T = D[f"{cfg}_Tp"].astype(float)[:, keep]
    out = {}
    for name, kind, par in DISTS:
        Dall = P.pdist(T, T[:NS], kind, par)                                   # (NP, NS)
        for agg in ["U", "kNN1"]:
            rows = []
            for s in range(n_splits):
                anc, cal, te = P.split(1000 + s)
                sc = P.agg(Dall[:, anc], agg); eps = kth_threshold(sc[cal], P.BETA)
                r = dict(eps=float(eps), coverage=float(np.mean(sc[te] <= eps)))
                for m in P.CFG["masses"]:
                    r[f"excl_{m}"] = float(np.mean(sc[NS:][Z[NS:, 2] == m] > eps))
                rows.append(r)
            res = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "eps"}
            res["eps_median"] = float(np.median([r["eps"] for r in rows]))
            out[f"{name}|{agg}"] = res
    return out


def probe_report():
    x = D["x_probe"]
    rep = {"flagged": {}, "x0_norm": {}, "x0_rank": {}, "maxnorm_probe": {}, "maxnorm_eval": {}, "share_L2": {}, "share_L4": {},
           "screen": {}}
    for cfg in CFGS + ["rect2_1step"]:
        rep["maxnorm_probe"][cfg] = float(np.linalg.norm(D[f"{cfg}_Tp"].astype(float), axis=-1).max())
        rep["maxnorm_eval"][cfg] = float(np.linalg.norm(D[f"{cfg}_Te"].astype(float), axis=-1).max())
    for cfg in CFGS:
        f = flagged(cfg); rep["flagged"][cfg] = f.tolist()
        rep["x0_norm"][cfg] = np.linalg.norm(x[f], axis=1).tolist()
        nx = np.linalg.norm(x, axis=1)                                         # one norm computation for all probes
        rep["x0_rank"][cfg] = [int((nx > nx[i]).sum()) for i in f]
        # share of the squared L^2 (and of the L^4) distance between safe prompts carried by the flagged probes
        T = D[f"{cfg}_Tp"].astype(float)[:NS]
        i, j = np.triu_indices(NS, 1)
        sel = np.random.default_rng(0).choice(len(i), 20000, replace=False)
        d2 = ((T[i[sel]] - T[j[sel]]) ** 2).sum(-1)
        for p, key in [(1, "share_L2"), (2, "share_L4")]:
            w = d2 ** p; sh = w[:, f].sum(1) / np.maximum(w.sum(1), 1e-300)
            rep[key][cfg] = dict(median=float(np.median(sh)), mean=float(sh.mean()), q90=float(np.quantile(sh, 0.9)))
        keep = np.ones(x.shape[0], bool); keep[f] = False
        rep["screen"][cfg] = {"all": screen(cfg, np.ones(x.shape[0], bool)), "drop": screen(cfg, keep)}
        print(f"{cfg}: flagged probes {f.tolist()}", flush=True)
    return rep


# ------------------------------------------------------------------ (2) attack 2
def jacobian_dirs(cfg, e, k=5, h=1e-3):
    """Embedding directions that move the probe outputs least (smallest singular values of the Jacobian first)."""
    cols = []
    for jj in range(len(e)):
        d = np.zeros(len(e)); d[jj] = h
        Tp = P.outputs(fields(P.WST[cfg], (e + d)[None])[0], cfg, P.x_probe)
        Tm = P.outputs(fields(P.WST[cfg], (e - d)[None])[0], cfg, P.x_probe)
        cols.append(((Tp - Tm) / (2 * h)).ravel())
    _, sv, Vt = np.linalg.svd(np.stack(cols, 1), full_matrices=False)
    return Vt[::-1][:k].T, sv[::-1][:k]


def attack2_run(name, cfg, start, seed, k=5, iters=400):
    anc, cal, te = P.split(1000)
    reg = P.Region(name, cfg, anc, cal)
    pA = lambda W, x: float(np.mean(np.linalg.norm(P.outputs(W, cfg, x), axis=1) <= r_A))
    x_att, x_fr = P.x_eval[:600], P.x_eval[600:]
    e0 = E[start].copy(); W0 = fields(P.WST[cfg], e0[None])[0]
    if reg.score_field(W0, e0) > reg.eps:
        raise RuntimeError("starting prompt not admitted")
    V, sv = jacobian_dirs(cfg, e0, k)
    rng = np.random.default_rng(seed)
    e, W, cur, step = e0, W0, pA(W0, x_att), 1.0
    for _ in range(iters):
        e2 = e + step * V @ rng.standard_normal(k); W2 = fields(P.WST[cfg], e2[None])[0]
        if reg.score_field(W2, e2) > reg.eps:
            step *= 0.9; continue
        val = pA(W2, x_att)
        if val >= cur:
            e, W, cur = e2, W2, val; step *= 1.1
    return dict(method=name, cfg=cfg, start=int(start), seed=int(seed), pA_start=pA(W0, x_fr), pA_search=cur,
                pA_fresh=pA(W, x_fr), score=float(reg.score_field(W, e) / reg.eps), dE=float(np.linalg.norm(e - e0)),
                sv=sv.tolist())


def attack2_jobs():
    anc, cal, te = P.split(1000)
    jobs = []
    for ci, cfg in enumerate(CFGS):
        for mi, name in enumerate(ATTACK_REGIONS):
            reg = P.Region(name, cfg, anc, cal)
            pool = np.r_[anc, cal]; adm = pool[reg.scores_pool(pool) <= reg.eps][:12]
            jobs += [(name, cfg, int(j), 7000 + 100 * (4 * ci + mi) + i) for i, j in enumerate(adm)]
    return jobs


def main():
    t0 = time.time()
    out = {"probe": probe_report()}
    print(f"[{time.time()-t0:.0f}s] probe sensitivity", flush=True)
    jobs = attack2_jobs()
    out["attack2"] = Parallel(n_jobs=12)(delayed(attack2_run)(*j) for j in jobs)
    print(f"[{time.time()-t0:.0f}s] attack 2 ({len(jobs)} runs)", flush=True)
    json.dump(out, open(os.path.join(P.RES, "robustness.json"), "w"), indent=0)
    for cfg in CFGS:
        S = out["probe"]["screen"][cfg]
        for key in S["all"]:
            a, d = S["all"][key], S["drop"][key]
            f = lambda r: f"cov {r['coverage']:.3f} excl " + " ".join(f"{r[f'excl_{m}']:.2f}" for m in P.CFG["masses"]) + \
                f" eps {r['eps_median']:.3f}"
            print(f"{cfg:6s} {key:14s} all: {f(a)} | dropped: {f(d)}")
    att = {}
    for r in out["attack2"]:
        att.setdefault((r["cfg"], r["method"]), []).append(r["pA_fresh"])
    for (cfg, name), v in att.items():
        print(f"attack 2 {cfg:6s} {name:10s}: worst P(A) {max(v):.3f}, median {np.median(v):.3f} (n={len(v)})")
    print(f"[{time.time()-t0:.0f}s] saved robustness.json")


if __name__ == "__main__":
    main()
