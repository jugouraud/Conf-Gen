"""E2b: a stress test of prompt-space regions, with an encoder that carries a nuisance ('style') token which does not change
what the prompt asks for.

A prompt has content z = (phi, rho, m) and a nuisance token s ~ U[-1, 1]; the target depends on z only. The encoder is
    e(z, s) = [e_E1(z), lam * sqrt(2 / Dn) cos(Om_n s + b_n)],
the E1 encoder followed by Dn nuisance coordinates. The 1-rectified model is refitted on 2000 training prompts with random
s, so it learns to ignore s as far as least squares can. The E2 pool keeps its content and each prompt gets its own s.
Least squares is invariant to the scale lam of the nuisance coordinates up to its 1e-7 ridge (the fitted weights scale by
1/lam), so the fields, the field distances and the pull-back metric are the same for every lam, while the Euclidean
embedding distance is not: lam is swept for the prompt-space region only (lam = 0 is the content embedding of E2).
Writes ../results/e2/nuisance.json and ../results/e2/nuisance_maps.npz.   Usage: python e2_nuisance.py
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json, time
import numpy as np
from joblib import Parallel, delayed
import e2_regions as R
from e1_common import Features, Encoder, fields, simulate, sample_target, fit_conditional, kth_threshold, mmd2, r_A

FIT = np.load(os.path.join(R.RES1, "fit.npz"))
CF = R.CFG1
DN, LAM_FIT, LAMS = 10, 1.0, [0.5, 1.0, 2.0, 4.0]
rng_n = np.random.default_rng(8)
OM_N, B_N = rng_n.standard_normal(DN) * 3.0, rng_n.uniform(0, 2 * np.pi, DN)


def nuis(s):
    return np.sqrt(2.0 / DN) * np.cos(np.outer(s, OM_N) + B_N)


def _sim(Ws, x0):
    return simulate(Features(CF["F"], seed=CF["feat_seed"]), Ws, x0, CF["J"])


def sim_parallel(Ws, x0, n_jobs=14):
    ch = [c for c in np.array_split(np.arange(len(Ws)), 4 * n_jobs) if len(c)]
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(_sim)(Ws[c], x0) for c in ch), 0)


def main():
    t0 = time.time()
    feat, enc = Features(CF["F"], seed=CF["feat_seed"]), Encoder(CF["Dr"], seed=CF["enc_seed"])
    # ---- refit of the 1-rectified model with the nuisance coordinates (lam = 1)
    Ztr = FIT["Ztr"]; s_tr = np.random.default_rng(12).uniform(-1, 1, len(Ztr))
    Etr = np.c_[enc(Ztr), LAM_FIT * nuis(s_tr)]
    rng = np.random.default_rng(CF["train_seed"])

    def indep(j, r):
        x1 = sample_target(Ztr[j], CF["n_x1"], r)
        return r.standard_normal(x1.shape), x1
    W = fit_conditional(feat, Etr, indep, rng, reps=CF["reps"], lam_rel=CF["lam_rel"])
    print(f"[{time.time()-t0:.0f}s] refit with nuisance coordinates", flush=True)
    # ---- the E2 pool with a nuisance token per prompt, and a (phi, s) grid at m = 0 and m = 0.1 for the maps
    Z = R.Z; s = np.random.default_rng(13).uniform(-1, 1, len(Z))
    phi_g, s_g = np.radians(np.linspace(-45, 45, 37)), np.linspace(-1, 1, 21)
    PG, SG = np.meshgrid(phi_g, s_g)
    Zg = np.concatenate([np.c_[PG.ravel(), np.full(PG.size, 2.0), np.full(PG.size, m)] for m in (0.0, 0.1)])
    sg = np.tile(SG.ravel(), 2)
    Ec, Ecg = R.E, enc(Zg)                                                  # content parts (the E2 embeddings)
    Nn, Nng = nuis(s), nuis(sg)
    Ws = fields(W, np.r_[np.c_[Ec, LAM_FIT * Nn], np.c_[Ecg, LAM_FIT * Nng]])
    Tp = sim_parallel(Ws, FIT["x_probe"]); Te = sim_parallel(Ws, FIT["x_eval"])
    pA = (np.linalg.norm(Te, axis=-1) <= r_A).mean(1)
    print(f"[{time.time()-t0:.0f}s] pool simulated", flush=True)
    n = len(Z); NS = R.NS
    fid = {"safe_pA": float(pA[:NS].mean()), **{f"pA_{m}": float(pA[NS:n][Z[NS:, 2] == m].mean()) for m in R.MASSES}}
    rr = np.random.default_rng(3)
    fid["mmd_safe"] = float(np.mean([mmd2(Te[i, :1000], sample_target(Z[i], 1000, rr)) for i in range(0, 300, 10)]) * 1e3)
    # ---- distances to the safe prompts: content and nuisance parts of the embedding, pull-back, truncated map distance
    Eall_c, Nall = np.r_[Ec, Ecg], np.r_[Nn, Nng]
    Dc = R.eucl(Eall_c, Ec[:NS]); Dn = R.eucl(Nall, Nn[:NS])
    DS = {f"E:{lam:g}": np.sqrt(Dc ** 2 + (lam * Dn) ** 2).astype(np.float32) for lam in LAMS}
    DS["E:0"] = Dc.astype(np.float32)
    feat0 = feat(FIT["x_probe"], 0.0)
    V = np.einsum("jcf,lf->jlc", W[1:], feat0); Q = np.einsum("jlc,klc->jk", V, V) / feat0.shape[0]
    C = np.linalg.cholesky(Q + 1e-12 * np.trace(Q) * np.eye(len(Q)))
    Zw = np.c_[Eall_c, LAM_FIT * Nall] @ C
    DS["PB:rect1"] = R.eucl(Zw, Zw[:NS]).astype(np.float32)
    DS["TR:rect1"] = R.map_dist(Tp, Tp[:NS], "TR")
    DS["E"] = DS["E:1"]                                                     # embedding distances for the X weights
    print(f"[{time.time()-t0:.0f}s] distances", flush=True)
    # how much the fields depend on the nuisance token: same content, other s (grid at m = 0, phi = 0) vs other content
    g0 = n + np.where((np.abs(Zg[:, 0]) < 1e-9) & (Zg[:, 2] == 0))[0]
    T_same = np.r_[Tp[g0]].astype(float)
    nrm = np.linalg.norm(T_same[:, None] - T_same[None], axis=-1); same = np.minimum(nrm, 1).mean(-1)[np.triu_indices(len(g0), 1)]
    iu = np.triu_indices(NS, 1); safe_pairs = DS["TR:rect1"][:NS][iu]
    fid["tr_same_content_median"] = float(np.median(same)); fid["tr_safe_pairs_median"] = float(np.median(safe_pairs))
    # ---- tuning and evaluation, M = 30 and M = 120
    res = {"fidelity": fid, "tune": {}, "eval": {}, "best": {}}
    spaces = [f"E:{lam:g}" for lam in [0] + LAMS] + ["PB:rect1", "TR:rect1"]
    for M in [30, 120]:
        for sp in spaces:
            for v in R.variants("E" if sp.startswith("E") or sp.startswith("PB") else "TR:x"):
                key = f"{sp}|{M}|{R.vname(v)}"
                T_ = [R.evaluate(DS, sp, v, *R.split(R.SAFE_TUNE, M, 5000 + k), R.UNS_TUNE) for k in range(50)]
                res["tune"][key] = {k: float(np.nanmean([r[k] for r in T_])) for k in T_[0]}
                E_ = [R.evaluate(DS, sp, v, *R.split(R.SAFE_EVAL, M, 1000 + k), R.UNS_EVAL) for k in range(200 if M == 30 else 100)]
                res["eval"][key] = {k: float(np.nanmean([r[k] for r in E_])) for k in E_[0]}
            ad = [v for v in R.variants("E" if sp.startswith("E") or sp.startswith("PB") else "TR:x") if v[0] != "U"]
            res["best"][f"{sp}|{M}"] = R.vname(max(ad, key=lambda v: res["tune"][f"{sp}|{M}|{R.vname(v)}"]["J"]))
    print(f"[{time.time()-t0:.0f}s] regions", flush=True)
    # ---- admission maps on the (phi, s) grid, evaluation split 0, M = 30
    anc, cal, te = R.split(R.SAFE_EVAL, 30, 1000)
    rows = np.arange(n, n + len(Zg)); maps = {}
    for sp in ["E:2", "PB:rect1", "TR:rect1"]:
        for tag, vn in [("static", "U"), ("adaptive", res["best"][f"{sp}|30"])]:
            k, p = (vn, None) if vn == "U" else (vn.split(":")[0], float(vn.split(":")[1]) if vn[0] != "N" else int(vn.split(":")[1]))
            Daa = DS[sp][np.ix_(anc, anc)]
            sc_cal = R.score(DS[sp][np.ix_(cal, anc)], (k, p), Daa, DS["E"][np.ix_(cal, anc)], DS["E"][np.ix_(anc, anc)])
            sc = R.score(DS[sp][np.ix_(rows, anc)], (k, p), Daa, DS["E"][np.ix_(rows, anc)], DS["E"][np.ix_(anc, anc)])
            maps[f"{sp}|{tag}"] = (sc / kth_threshold(sc_cal, R.BETA)).reshape(2, *PG.shape)
    np.savez_compressed(os.path.join(R.RES, "nuisance_maps.npz"), phi_deg=np.degrees(phi_g), s=s_g, anc=anc, s_pool=s, **maps)
    json.dump(res, open(os.path.join(R.RES, "nuisance.json"), "w"), indent=0)
    print("fidelity:", {k: round(v, 4) for k, v in fid.items()})
    for sp in spaces:
        b = res["best"][f"{sp}|30"]; u, a = res["eval"][f"{sp}|30|U"], res["eval"][f"{sp}|30|{b}"]
        fmt = lambda r: f"cov {r['coverage']:.3f} excl " + " ".join(f"{r[f'excl_{m}']:.2f}" for m in R.MASSES) + f" J {r['J']:.3f}"
        b2 = res["best"][f"{sp}|120"]; a2 = res["eval"][f"{sp}|120|{b2}"]
        print(f"{sp:10s} static: {fmt(u)} | adaptive {b:6s}: {fmt(a)} | M=120 adaptive {b2}: J {a2['J']:.3f} excl.03 {a2['excl_0.03']:.2f}")
    print(f"[{time.time()-t0:.0f}s] saved nuisance.json")


if __name__ == "__main__":
    main()
