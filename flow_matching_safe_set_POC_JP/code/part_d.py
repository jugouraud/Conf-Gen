"""Pilot D: the SAME linear flow-matching fits, sampled with the marginal-preserving SDE
    dX = [(1 + c t) v_W(X, t) - c X] dt + sqrt(2 c (1 - t)) dB      (score s = (t v - x)/(1 - t)).
Euler-Maruyama chains with a common noise law give, exactly,
    KL(P^W || P^W') = 1/2 (W - W')^T G_KL(W) (W - W'),
    G_KL(W) = sum_k h (1 + c t_k)^2 / (2 c (1 - t_k)) E_{P^W_k}[psi psi^T]   (Gram along W's OWN chain).
Order-1 regions with reverse-KL cost (on-policy) and forward-KL cost (historical chains)."""
import os
import numpy as np, time, sys
from scipy.stats import beta as betad
from lfm_common import *
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
t_start = time.time()
F_, lam, c = 256, 1e-5, 1.0
feat = Features(F_, seed=0); D = feat.dim
S = np.load(f"{OUT}/part_c_state.npz")
W_all, L_means = S["W_all"], S["L_means"]; P = W_all.shape[0]
masses = list(S["masses"])
# rebuild the contaminated fits exactly as in part_c_pre (same RNG stream)
rng = np.random.default_rng(11)
A_all, B_all = np.zeros((P, D, D)), np.zeros((P, 2, D))
for a in range(P):
    A_all[a], B_all[a] = fm_moments(feat, sample_task(L_means[a], 20000, rng), rng)
A_bad, B_bad = fm_moments(feat, bad_std * rng.standard_normal((20000, 2)), rng)
assert np.allclose(np.stack([ridge(A_all[a], B_all[a], lam) for a in range(3)]), W_all[:3])
W_bad = {m: np.stack([ridge((1 - m) * A_all[a] + m * A_bad, (1 - m) * B_all[a] + m * B_bad, lam) for a in range(P)]) for m in masses}
del A_all, B_all

h = 1.0 / K
wk = h * (1 + c * tgrid[:K]) ** 2 / (2 * c * (1 - tgrid[:K]))        # KL time weights
rp = np.random.default_rng(21)
Z0p, Zp = rp.standard_normal((256, 2)), rp.standard_normal((K, 256, 2))
Z0b, Zb = rp.standard_normal((4000, 2)), rp.standard_normal((K, 4000, 2))

def sde(W, Z0, Z, gram=False):
    Y = Z0.copy(); G = np.zeros((D, D)) if gram else None
    for k in range(K):
        t = tgrid[k]; Psi = feat(Y, t)
        if gram: G += wk[k] * Psi.T @ Psi / Y.shape[0]
        Y = Y + h * ((1 + c * t) * (Psi @ W.T) - c * Y) + np.sqrt(2 * c * (1 - t) * h) * Z[k]
    return (Y, G) if gram else Y

def qf(Delta, G): return np.einsum("...cd,de,...ce->...", Delta, G, Delta)
G_L = np.zeros((P, D, D)); pA_L = np.zeros(P)
for a in range(P):
    _, G_L[a] = sde(W_all[a], Z0p, Zp, gram=True)
    pA_L[a] = (np.linalg.norm(sde(W_all[a], Z0b, Zb), axis=1) <= r_A).mean()
# KL matrices: REV[a, i] = KL(P^a || P^i) (Gram of a), FWD[a, i] = KL(P^i || P^a) (Gram of i)
REV_LL = np.stack([0.5 * qf(W_all[a][None] - W_all, G_L[a]) for a in range(P)])
FWD_LL = REV_LL.T.copy()
print(f"legit chains {time.time()-t_start:.0f}s; SDE P(A) legit mean {pA_L.mean():.4f} max {pA_L.max():.4f}", flush=True)
REV_BL, FWD_BL, pA_B = {}, {}, {}
for m in masses:
    REV_BL[m] = np.zeros((P, P)); FWD_BL[m] = np.zeros((P, P)); pA_B[m] = np.zeros(P)
    for a in range(P):
        _, Gb = sde(W_bad[m][a], Z0p, Zp, gram=True)
        REV_BL[m][a] = 0.5 * qf(W_bad[m][a][None] - W_all, Gb)
        FWD_BL[m][a] = 0.5 * np.einsum("icd,ide,ice->i", W_bad[m][a][None] - W_all, G_L, W_bad[m][a][None] - W_all)
        pA_B[m][a] = (np.linalg.norm(sde(W_bad[m][a], Z0b, Zb), axis=1) <= r_A).mean()
    print(f"m={m}: SDE P(A) {pA_B[m].mean():.4f}; median KL to own clean fit: reverse {np.median(REV_BL[m][np.arange(P), np.arange(P)]):.3f} "
          f"forward {np.median(FWD_BL[m][np.arange(P), np.arange(P)]):.4f}  ({time.time()-t_start:.0f}s)", flush=True)
off = ~np.eye(P, dtype=bool)
print(f"legit pairwise KL: median {np.median(REV_LL[off]):.3f}, 90% {np.quantile(REV_LL[off], .9):.3f}")

def kl_bin(p, q):
    p = np.clip(p, 1e-300, 1 - 1e-16); q = np.clip(q, 1e-300, 1 - 1e-16)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
def kl_inv(q, kappa):
    lo, hi = q, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if kl_bin(mid, q) <= kappa: lo = mid
        else: hi = mid
    return lo
def phi(cc): return np.minimum(cc / 2, np.sqrt(1 - np.exp(-cc ** 2 / 2)))

M, Kc, beta = 20, 40, 0.1
q_idx = int(np.ceil((Kc + 1) * (1 - beta))); norm = 1.0 / (M * (M + 1))
reps = 200; rs = np.random.default_rng(5)
out = {k: dict(cov=[], excl={m: [] for m in masses}, bound=[], inside=[]) for k in ["rev", "fwd"]}
for r in range(reps):
    perm = rs.permutation(P); tr, ca, te = perm[:M], perm[M:M + Kc], perm[M + Kc:]
    cnt = int(round(pA_L[tr].sum() * 4000)); q_ucb = betad.ppf(0.95, cnt + 1, M * 4000 - cnt)
    for k, LL, BL in [("rev", REV_LL, REV_BL), ("fwd", FWD_LL, FWD_BL)]:
        eps = np.sort(LL[np.ix_(ca, tr)].sum(1) * norm)[q_idx - 1]
        s_te = LL[np.ix_(te, tr)].sum(1) * norm
        out[k]["cov"].append((s_te <= eps).mean())
        ins = [pA_L[te][s_te <= eps]]
        for m in masses:
            sb = BL[m][np.ix_(te, tr)].sum(1) * norm
            out[k]["excl"][m].append((sb > eps).mean()); ins.append(pA_B[m][te][sb <= eps])
        out[k]["inside"].append(np.concatenate(ins).max())
        # rev: KL(P^W || Pbar) <= (M+1) eps  -> kl-inversion;  fwd: KL(Pbar-components) -> TV via phi(sqrt(2 KL))
        out[k]["bound"].append(kl_inv(q_ucb, (M + 1) * eps) if k == "rev"
                               else q_ucb + phi(np.sqrt(2 * (M + 1) * eps)))
for k, name in [("rev", "reverse KL (on-policy Gram)"), ("fwd", "forward KL (historical Grams, convex)")]:
    o = out[k]
    print(f"[{name}] coverage {np.mean(o['cov']):.3f};  exclusion " + "  ".join(f"m={m}: {np.mean(o['excl'][m]):.2f}" for m in masses) +
          f";  guaranteed P(A) bound median {np.median(o['bound']):.3f};  max P(A) inside: median {np.median(o['inside']):.4f} max {np.max(o['inside']):.4f};"
          f"  violations {int(np.sum(np.array(o['inside']) > np.array(o['bound'])))}/{reps}")
print(f"total {time.time()-t_start:.0f}s")

# ---- save arrays for the figures ----
import json
np.savez(f"{OUT}/part_d_cache.npz", REV_LL=REV_LL, pA_L=pA_L, masses=np.array(masses),
         REV_BL=np.stack([REV_BL[m] for m in masses]), FWD_BL=np.stack([FWD_BL[m] for m in masses]),
         pA_B=np.stack([pA_B[m] for m in masses]), wk=wk)
json.dump({k: dict(cov=[float(x) for x in o["cov"]], excl={str(m): [float(x) for x in v] for m, v in o["excl"].items()},
                   bound=[float(x) for x in o["bound"]], inside=[float(x) for x in o["inside"]]) for k, o in out.items()},
          open(f"{OUT}/part_d_splits.json", "w"), indent=1)
print("saved part_d_cache.npz and part_d_splits.json")
