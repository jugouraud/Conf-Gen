"""Pilot C (precompute): linear flow matching v_W(x,t) = W psi(x,t), ample-data least-squares fits,
order-1 conformal regions in (i) the fixed-reference metric ||(W-W')||_{G_ref} and (ii) the on-policy
metric ||(W-W')||_{G_on(W)}; coverage and exclusion of contaminated fields over random splits."""
import os
import numpy as np, time, sys
from lfm_common import *
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
t_start = time.time()
F_, lam, delta = 256, 1e-5, 1e-6
feat = Features(F_, seed=0); D = feat.dim
cache = np.load(f"{OUT}/part_a_cache.npz")
phi_all, rho_all = cache["phi"], cache["rho"]
P = phi_all.shape[0]; L_means = ring(phi_all, rho_all)
rng = np.random.default_rng(11)

n_mc = 20000
A_all, B_all = np.zeros((P, D, D)), np.zeros((P, 2, D))
for a in range(P):
    A_all[a], B_all[a] = fm_moments(feat, sample_task(L_means[a], n_mc, rng), rng)
A_bad, B_bad = fm_moments(feat, bad_std * rng.standard_normal((n_mc, 2)), rng)
W_all = np.stack([ridge(A_all[a], B_all[a], lam) for a in range(P)])
masses = [0.01, 0.03, 0.1, 0.3]
W_bad = {m: np.stack([ridge((1 - m) * A_all[a] + m * A_bad, (1 - m) * B_all[a] + m * B_bad, lam) for a in range(P)])
         for m in masses}
print(f"fits {time.time()-t_start:.0f}s", flush=True)

def reg(G): return G + delta * np.trace(G) / D * np.eye(D)
n_p = 256
X0p = np.random.default_rng(1).standard_normal((n_p, 2))
def onpolicy_gram(W):
    _, feats = simulate(feat, W, X0p, store_feats=True)
    Psi = np.concatenate(feats, 0)
    return reg(Psi.T @ Psi / Psi.shape[0])
def qf(Delta, G): return np.einsum("...cd,de,...ce->...", Delta, G, Delta)

# on-policy cost matrices (split-independent): C_on[a, i] = ||W_a - W_i||_{G_on(W_a)}
C_on_LL = np.zeros((P, P)); C_on_BL = {m: np.zeros((P, P)) for m in masses}
for a in range(P):
    C_on_LL[a] = np.sqrt(np.maximum(qf(W_all[a][None] - W_all, onpolicy_gram(W_all[a])), 0))
    for m in masses:
        C_on_BL[m][a] = np.sqrt(np.maximum(qf(W_bad[m][a][None] - W_all, onpolicy_gram(W_bad[m][a])), 0))
print(f"on-policy costs {time.time()-t_start:.0f}s", flush=True)

X0b = np.random.default_rng(2).standard_normal((10000, 2))
def p_unsafe(W): return float((np.linalg.norm(simulate(feat, W, X0b), axis=1) <= r_A).mean())
pA_all = np.array([p_unsafe(W_all[a]) for a in range(P)])
pA_bad = {m: np.array([p_unsafe(W_bad[m][a]) for a in range(P)]) for m in masses}
print(f"P(A) {time.time()-t_start:.0f}s; legit mean {pA_all.mean():.4f} max {pA_all.max():.4f}; " +
      "  ".join(f"m={m}: {pA_bad[m].mean():.4f}" for m in masses), flush=True)

M, Kc, beta = 20, 40, 0.1
q_idx = int(np.ceil((Kc + 1) * (1 - beta))); norm = 1.0 / (M * (M + 1))
def ref_costs(Wc, tr, Cw):      # ||(W_c - W_i) Cw||_F for all c, i in tr
    Z, Zt = Wc @ Cw, W_all[tr] @ Cw
    return np.linalg.norm((Z[:, None] - Zt[None]).reshape(len(Wc), len(tr), -1), axis=2)

reps = 100; rs = np.random.default_rng(5)
cov = {"ref": [], "on": []}; excl = {k: {m: [] for m in masses} for k in cov}; inside_pA = {k: [] for k in cov}
for r in range(reps):
    perm = rs.permutation(P); tr, ca, te = perm[:M], perm[M:M + Kc], perm[M + Kc:]
    Cw = np.linalg.cholesky(reg(A_all[tr].mean(0)))
    eps = np.sort(ref_costs(W_all[ca], tr, Cw).sum(1) * norm)[q_idx - 1]
    s_te = ref_costs(W_all[te], tr, Cw).sum(1) * norm; cov["ref"].append((s_te <= eps).mean())
    ins = [pA_all[te][s_te <= eps]]
    for m in masses:
        sb = ref_costs(W_bad[m][te], tr, Cw).sum(1) * norm
        excl["ref"][m].append((sb > eps).mean()); ins.append(pA_bad[m][te][sb <= eps])
    inside_pA["ref"].append(np.concatenate(ins).max())
    eps = np.sort(C_on_LL[np.ix_(ca, tr)].sum(1) * norm)[q_idx - 1]
    s_te = C_on_LL[np.ix_(te, tr)].sum(1) * norm; cov["on"].append((s_te <= eps).mean())
    ins = [pA_all[te][s_te <= eps]]
    for m in masses:
        sb = C_on_BL[m][np.ix_(te, tr)].sum(1) * norm
        excl["on"][m].append((sb > eps).mean()); ins.append(pA_bad[m][te][sb <= eps])
    inside_pA["on"].append(np.concatenate(ins).max())
for k in cov:
    print(f"[{k}] coverage {np.mean(cov[k]):.3f} (5-95%: {np.quantile(cov[k],.05):.3f}-{np.quantile(cov[k],.95):.3f}); exclusion " +
          "  ".join(f"m={m}: {np.mean(excl[k][m]):.2f}" for m in masses) +
          f"; max P(A) of a field inside: median {np.median(inside_pA[k]):.3f}", flush=True)

# fixed split for the fine-tuning study
perm = np.random.default_rng(7).permutation(P); tr, ca, te = perm[:M], perm[M:M + Kc], perm[M + Kc:]
G = reg(A_all[tr].mean(0)); Cw = np.linalg.cholesky(G)
eps_ref = np.sort(ref_costs(W_all[ca], tr, Cw).sum(1) * norm)[q_idx - 1]
eps_on = np.sort(C_on_LL[np.ix_(ca, tr)].sum(1) * norm)[q_idx - 1]
np.savez(f"{OUT}/part_c_state.npz", W_all=W_all, L_means=L_means, tr=tr, ca=ca, te=te, G=G,
         eps_ref=eps_ref, eps_on=eps_on, pA_all=pA_all,
         cov_ref=cov["ref"], cov_on=cov["on"],
         excl_ref=[np.mean(excl["ref"][m]) for m in masses], excl_on=[np.mean(excl["on"][m]) for m in masses],
         masses=masses, pA_bad=np.stack([pA_bad[m] for m in masses]))
print(f"saved state; eps_ref={eps_ref:.4f} eps_on={eps_on:.4f}; total {time.time()-t_start:.0f}s")
