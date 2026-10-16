"""E2, step 1: a fresh, larger prompt pool for the comparison of prompt-space and field-space regions, static and adaptive.

Uses the models fitted by e1_fit.py (1-rectified W1, reflowed W2) and the same frozen encoder; only the prompts are new.
  safe prompts:   1000, split once into a TUNING pool (300) and an EVALUATION pool (700);
  unsafe prompts: 200 bases x masses {0.01, 0.03, 0.1, 0.3}, bases 0-99 for tuning and 100-199 for evaluation;
  admission grid: rotation phi in [-45, 45] degrees x unsafe mass m in [0, 0.3], radius 2 (for the maps).
For every prompt: the embedding, the probe outputs (256 noises) of the three model/sampler configurations
(rect1 = 1-rectified, 50 Euler steps; rect2 = reflowed, 50 steps; rect2_1step = reflowed, one step) and P(A) on 2000 noises.
Writes ../results/e2/pool.npz.    Usage: python e2_pool.py
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, time
import numpy as np
from joblib import Parallel, delayed
from e1_common import Features, Encoder, fields, simulate, one_step, safe_prompts, r_A, PHI_MAX

HERE = os.path.dirname(os.path.abspath(__file__))
RES1 = os.path.join(HERE, "..", "results", "e1")
OUT = os.path.join(HERE, "..", "results", "e2"); os.makedirs(OUT, exist_ok=True)
CFG1 = json.load(open(os.path.join(RES1, "fit_config.json")))
POOL = dict(n_safe=1000, n_tune=300, n_bases=200, n_bases_tune=100, masses=[0.01, 0.03, 0.1, 0.3], seed=101,
            grid_phi_deg=np.linspace(-45, 45, 37).tolist(), grid_m=np.linspace(0, 0.3, 31).tolist(), grid_rho=2.0,
            n_jobs=12)
CONFIGS = {"rect1": ("W1", CFG1["J"]), "rect2": ("W2", CFG1["J"]), "rect2_1step": ("W2", 1)}


def _sim(Ws, x0, J):
    feat = Features(CFG1["F"], seed=CFG1["feat_seed"])
    return simulate(feat, Ws, x0, J) if J > 1 else one_step(feat, Ws, x0)


def sim_parallel(Ws, x0, J, n_jobs):
    chunks = [c for c in np.array_split(np.arange(len(Ws)), 4 * n_jobs) if len(c)]
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(_sim)(Ws[c], x0, J) for c in chunks), 0)


def main():
    t0 = time.time()
    D1 = np.load(os.path.join(RES1, "fit.npz"))
    enc = Encoder(CFG1["Dr"], seed=CFG1["enc_seed"])
    rng = np.random.default_rng(POOL["seed"])
    Zs = safe_prompts(POOL["n_safe"], rng)
    base = safe_prompts(POOL["n_bases"], rng)[:, :2]
    Zu = np.concatenate([np.c_[base, np.full(len(base), m)] for m in POOL["masses"]], 0)
    phi = np.radians(POOL["grid_phi_deg"]); mm = np.array(POOL["grid_m"])
    PH, MM = np.meshgrid(phi, mm)                                           # (n_m, n_phi)
    Zg = np.c_[PH.ravel(), np.full(PH.size, POOL["grid_rho"]), MM.ravel()]
    Z = np.r_[Zs, Zu]; E = enc(Z); Eg = enc(Zg)
    out = dict(Z=Z, E=E, Zg=Zg, Eg=Eg, n_safe=POOL["n_safe"], n_tune=POOL["n_tune"], base=base,
               masses=np.array(POOL["masses"]), base_idx=np.tile(np.arange(POOL["n_bases"]), len(POOL["masses"])),
               grid_shape=np.array(PH.shape))
    x_probe, x_eval = D1["x_probe"], D1["x_eval"]
    for name, (wkey, J) in CONFIGS.items():
        Wst = D1[wkey]
        for tag, ZZ in [("", E), ("g", Eg)]:
            Ws = fields(Wst, ZZ)
            Tp = sim_parallel(Ws, x_probe, J, POOL["n_jobs"])
            Te = sim_parallel(Ws, x_eval, J, POOL["n_jobs"])
            out[f"{name}_Tp{tag}"] = Tp.astype(np.float32)
            out[f"{name}_pA{tag}"] = (np.linalg.norm(Te, axis=-1) <= r_A).mean(1)
        print(f"[{time.time()-t0:.0f}s] {name}: pool and grid simulated", flush=True)
    np.savez_compressed(os.path.join(OUT, "pool.npz"), **out)
    json.dump(POOL, open(os.path.join(OUT, "pool_config.json"), "w"), indent=1)
    NS = POOL["n_safe"]
    for name in CONFIGS:
        pA = out[f"{name}_pA"]
        print(f"{name:12s} P(A) safe {pA[:NS].mean():.4f} | " + " ".join(
            f"m={m}: {pA[NS:][Zu[:, 2] == m].mean():.3f}" for m in POOL["masses"]))
    print(f"[{time.time()-t0:.0f}s] saved pool.npz")


if __name__ == "__main__":
    main()
