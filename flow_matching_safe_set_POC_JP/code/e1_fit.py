"""E1, step 1: fit the prompt-conditioned linear flow-matching model (1-rectified), reflow it (2-rectified),
and simulate a pool of safe and unsafe prompts for three model/sampler configurations:
    rect1 = 1-rectified, 50 Euler steps;  rect2 = 2-rectified, 50 Euler steps;  rect2_1step = 2-rectified, one step.
Writes ../results/e1/fit.npz and ../results/e1/fit_config.json.   Usage: python e1_fit.py
"""
import os, json, time, hashlib
import numpy as np
from joblib import Parallel, delayed
from e1_common import *

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "e1"); os.makedirs(OUT, exist_ok=True)
CFG = dict(F=256, feat_seed=0, Dr=40, enc_seed=7, n_train=2000, train_seed=11, n_x1=300, reps=4,
           lam_rel=1e-7, J=50, reflow_pairs=1200, reflow_seed=13, reflow_trim=4.0, reflow_lam_rel=1e-5,
           L_probe=256, probe_seed=1, N_eval=2000, eval_seed=2,
           n_safe=300, n_unsafe_base=100, masses=[0.01, 0.03, 0.1, 0.3], pool_seed=5,
           gram_n=4000, gram_seed=17, n_jobs=12)
TAG = hashlib.sha1(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:10]


def chunks(n, k):
    b = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(b[i], b[i + 1]) for i in range(k) if b[i + 1] > b[i]]


def _reflow_chunk(idx, Ws, seed):
    feat = Features(CFG["F"], seed=CFG["feat_seed"])
    r = np.random.default_rng([seed, int(idx[0])])
    x0 = r.standard_normal((len(idx), CFG["reflow_pairs"], 2))
    return x0, simulate(feat, Ws, x0, CFG["J"])


def _sim_chunk(Ws, x0, J):
    feat = Features(CFG["F"], seed=CFG["feat_seed"])
    return simulate(feat, Ws, x0, J) if J > 1 else one_step(feat, Ws, x0)


def sim_parallel(Ws, x0, J):
    parts = Parallel(n_jobs=CFG["n_jobs"])(delayed(_sim_chunk)(Ws[c], x0, J) for c in chunks(len(Ws), CFG["n_jobs"] * 2))
    return np.concatenate(parts, 0)


def _gram_chunk(Z, seed):
    feat = Features(CFG["F"], seed=CFG["feat_seed"])
    return np.stack([path_gram(feat, z, CFG["gram_n"], np.random.default_rng([seed, i])) for i, z in enumerate(Z)])


def main():
    t0 = time.time()
    feat, enc = Features(CFG["F"], seed=CFG["feat_seed"]), Encoder(CFG["Dr"], seed=CFG["enc_seed"])
    rng = np.random.default_rng(CFG["train_seed"])
    Ztr = train_prompts(CFG["n_train"], rng); Etr = enc(Ztr)

    def indep(j, r):
        x1 = sample_target(Ztr[j], CFG["n_x1"], r)
        return r.standard_normal(x1.shape), x1
    W1 = fit_conditional(feat, Etr, indep, rng, reps=CFG["reps"], lam_rel=CFG["lam_rel"])
    print(f"[{time.time()-t0:.0f}s] 1-rectified fit", flush=True)

    # reflow: couple each training prompt's noise with the 1-rectified model's own output, refit
    Wtr = fields(W1, Etr)
    parts = Parallel(n_jobs=CFG["n_jobs"])(delayed(_reflow_chunk)(c, Wtr[c], CFG["reflow_seed"])
                                           for c in chunks(len(Ztr), CFG["n_jobs"] * 2))
    X0r = np.concatenate([p[0] for p in parts], 0); X1r = np.concatenate([p[1] for p in parts], 0)
    # a few training noises (about 0.1%) are sent far away by the 1-rectified sampler (|x1| up to 1e5); these pairs
    # dominate the least-squares refit and make the 2-rectified sampler unstable, so they are dropped
    keep = np.linalg.norm(X1r, axis=-1) <= CFG["reflow_trim"]
    print(f"reflow: dropped {1 - keep.mean():.4%} of the pairs (|x1| > {CFG['reflow_trim']}; max |x1| = {np.linalg.norm(X1r, axis=-1).max():.3g})")
    W2 = fit_conditional(feat, Etr, lambda j, r: (X0r[j][keep[j]], X1r[j][keep[j]]), np.random.default_rng(CFG["reflow_seed"] + 1),
                         reps=4, lam_rel=CFG["reflow_lam_rel"])
    print(f"[{time.time()-t0:.0f}s] reflow pairs + 2-rectified fit", flush=True)

    # prompt pool: safe prompts, and unsafe prompts built on independent safe bases
    rp = np.random.default_rng(CFG["pool_seed"])
    Zs = safe_prompts(CFG["n_safe"], rp)
    base = safe_prompts(CFG["n_unsafe_base"], rp)[:, :2]
    Zu = np.concatenate([np.c_[base, np.full(len(base), m)] for m in CFG["masses"]], 0)
    Z = np.r_[Zs, Zu]; E = enc(Z)
    x_probe = np.random.default_rng(CFG["probe_seed"]).standard_normal((CFG["L_probe"], 2))
    x_eval = np.random.default_rng(CFG["eval_seed"]).standard_normal((CFG["N_eval"], 2))
    out = dict(Z=Z, E=E, n_safe=CFG["n_safe"], masses=np.array(CFG["masses"]), base=base,
               x_probe=x_probe, x_eval=x_eval, W1=W1, W2=W2, Ztr=Ztr)
    for name, Wst, J in [("rect1", W1, CFG["J"]), ("rect2", W2, CFG["J"]), ("rect2_1step", W2, 1)]:
        Ws = fields(Wst, E)
        Tp, Te = sim_parallel(Ws, x_probe, J), sim_parallel(Ws, x_eval, J)
        T1p = one_step(feat, Ws, x_probe)
        out[f"{name}_Tp"], out[f"{name}_Te"], out[f"{name}_T1p"] = Tp.astype(np.float32), Te.astype(np.float32), T1p.astype(np.float32)
        out[f"{name}_pA"] = (np.linalg.norm(Te, axis=-1) <= r_A).mean(1)
        out[f"{name}_zeta"] = np.sqrt(((Tp - T1p) ** 2).sum(-1).mean(-1))
        print(f"[{time.time()-t0:.0f}s] simulated pool for {name}", flush=True)

    # interpolation-path Gram matrices of the true safe targets (for the pooled-path distance)
    Gp = Parallel(n_jobs=CFG["n_jobs"])(delayed(_gram_chunk)(Zs[c], CFG["gram_seed"] + int(c[0]))
                                        for c in chunks(len(Zs), CFG["n_jobs"]))
    out["Gp_safe"] = np.concatenate(Gp, 0)
    # true samples: safe counterpart (m = 0) of every pool prompt, for fidelity; and true P(A) of every prompt
    rt = np.random.default_rng(23)
    out["true_safe_samples"] = np.stack([sample_target(np.r_[z[:2], 0.0], 1000, rt) for z in Z]).astype(np.float32)
    out["true_pA"] = np.array([(np.linalg.norm(sample_target(z, 20000, rt), axis=1) <= r_A).mean() for z in Z])
    np.savez_compressed(os.path.join(OUT, "fit.npz"), **out)
    json.dump(dict(CFG, tag=TAG), open(os.path.join(OUT, "fit_config.json"), "w"), indent=1)

    # model quality summary
    for name in ["rect1", "rect2", "rect2_1step"]:
        Te, pA = out[f"{name}_Te"], out[f"{name}_pA"]
        mm = [mmd2(Te[i, :1000].astype(float), out["true_safe_samples"][i].astype(float)) * 1e3 for i in range(0, CFG["n_safe"], 10)]
        msg = f"{name:12s} safe: MMD2x1e3 {np.mean(mm):.2f}  P(A) {pA[:CFG['n_safe']].mean():.4f}  zeta {out[f'{name}_zeta'][:CFG['n_safe']].mean():.3f} |"
        for k, m in enumerate(CFG["masses"]):
            s = slice(CFG["n_safe"] + k * CFG["n_unsafe_base"], CFG["n_safe"] + (k + 1) * CFG["n_unsafe_base"])
            msg += f" m={m}: P(A) {pA[s].mean():.4f} (true {out['true_pA'][s].mean():.4f})"
        print(msg, flush=True)
    print(f"[{time.time()-t0:.0f}s] saved fit.npz (tag {TAG})")


if __name__ == "__main__":
    main()
