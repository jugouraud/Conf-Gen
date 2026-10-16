"""Pilot C (fine-tuning worker): few-shot linear-FM fine-tuning on held-out tasks j in [j0, j1).
Usage: python part_c_ft.py j0 j1   (reads results/part_c_state.npz, writes results/part_c_ft_{j0}_{j1}.json).
Importable: few_shot_data(j) and fit_methods(A_n, B_n) are reused by example_fits.py."""
import os, numpy as np, cvxpy as cp, time, json, sys
from lfm_common import *
OUT = os.environ.get("POC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
F_, lam, delta = 256, 1e-5, 1e-6
feat = Features(F_, seed=0); D = feat.dim
S = np.load(f"{OUT}/part_c_state.npz")
W_all, L_means, tr, te, G = S["W_all"], S["L_means"], S["tr"], S["te"], S["G"]
eps_ref, eps_on = float(S["eps_ref"]), float(S["eps_on"])
M = len(tr); norm = 1.0 / (M * (M + 1))
Cw = np.linalg.cholesky(G); Cwi = np.linalg.inv(Cw)

def reg(G): return G + delta * np.trace(G) / D * np.eye(D)
X0p = np.random.default_rng(1).standard_normal((256, 2))
def onpolicy_gram(W):
    _, feats = simulate(feat, W, X0p, store_feats=True)
    Psi = np.concatenate(feats, 0)
    return reg(Psi.T @ Psi / Psi.shape[0])
def qf(Delta, G_): return np.einsum("...cd,de,...ce->...", Delta, G_, Delta)
def s_ref(W): return np.sqrt(np.maximum(qf(W[None] - W_all[tr], G), 0)).sum() * norm
def s_on(W): return np.sqrt(np.maximum(qf(W[None] - W_all[tr], onpolicy_gram(W)), 0)).sum() * norm
X0b = np.random.default_rng(2).standard_normal((10000, 2))
def p_unsafe(W): return float((np.linalg.norm(simulate(feat, W, X0b), axis=1) <= r_A).mean())

Ztr = W_all[tr] @ Cw; Zc = Ztr.mean(0)          # geometric median of the anchors (Weiszfeld, whitened)
for _ in range(1000):
    d = np.linalg.norm((Ztr - Zc[None]).reshape(M, -1), axis=1); w = 1 / np.maximum(d, 1e-12)
    Zc = (w[:, None, None] * Ztr).sum(0) / w.sum()
W_c = Zc @ Cwi

def socp(A, B, C_, budget):
    """min tr(W (A + lam I) W^T) - 2 tr(B W^T) s.t. sum_i ||(W - W_i) C_||_F <= budget, solved in Z = W C_."""
    Ci = np.linalg.inv(C_)
    Q = Ci @ (A + lam * np.eye(D)) @ Ci.T; Q = (Q + Q.T) / 2
    R = np.linalg.cholesky(Q + 1e-12 * np.trace(Q) / D * np.eye(D))
    Bz, Zi = B @ Ci.T, W_all[tr] @ C_
    Z = cp.Variable((2, D))
    obj = cp.sum_squares(Z @ R) - 2 * cp.sum(cp.multiply(Bz, Z))
    cp.Problem(cp.Minimize(obj), [sum(cp.norm(Z - Zi[i], "fro") for i in range(M)) <= budget]).solve(solver=cp.CLARABEL)
    return Z.value @ Ci

def mmd2(x, y, bws=(0.1, 0.3, 1.0)):
    def k(a, b):
        d2 = ((a[:, None] - b[None]) ** 2).sum(-1)
        return sum(np.exp(-d2 / (2 * s * s)) for s in bws)
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y); n, m = len(x), len(y)
    return (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()

X0q = np.random.default_rng(3).standard_normal((2500, 2))
def evaluate(W, true_means, seed):
    y = simulate(feat, W, X0q); xt = sample_task(true_means, 2500, np.random.default_rng(seed))
    return dict(mmd=float(mmd2(y, xt)), pA=p_unsafe(W), s_ref=float(s_ref(W) / eps_ref), s_on=float(s_on(W) / eps_on))

n_clean = 12

def few_shot_data(j):
    """Few-shot data of held-out task te[j] (same RNG sequence for n_bad = 0 then 3)."""
    a = te[j]; rd = np.random.default_rng(100 + j)
    clean = sample_task(L_means[a], n_clean, rd)
    for n_bad in [0, 3]:
        data = np.r_[clean, bad_std * rd.standard_normal((n_bad, 2))]
        A_n, B_n = fm_moments(feat, data, rd, reps=3000)
        yield n_bad, data, A_n, B_n

def fit_methods(A_n, B_n):
    """All few-shot estimators; returns ({name: W}, {name: info})."""
    W, info = {}, {}
    for l in [1e-5, 1e-3, 1e-1]:
        W[f"U_lam{l:g}"] = ridge(A_n, B_n, l)
    for g_ in [0.1, 1.0, 10.0]:     # shrinkage to the centre: min L + g ||(W - W_c)||_G^2
        W[f"S_gam{g_:g}"] = np.linalg.solve(A_n + lam * np.eye(D) + g_ * G, (B_n + g_ * W_c @ G).T).T
    t0 = time.time(); W["C_ref"] = socp(A_n, B_n, Cw, M * (M + 1) * eps_ref); info["C_ref"] = dict(time=time.time() - t0)
    t0 = time.time(); Wk = W["C_ref"]
    for it in range(4):            # sequential convex programming on the on-policy metric
        Wk = socp(A_n, B_n, np.linalg.cholesky(onpolicy_gram(Wk)), M * (M + 1) * eps_on)
    W["C_on_scp"] = Wk.copy()
    backoff = 0.0
    if s_on(Wk) > eps_on:          # radial back-off towards the centre (gauge-type retraction)
        lo, hi = 0.0, 1.0
        for _ in range(16):
            mid = (lo + hi) / 2
            if s_on((1 - mid) * Wk + mid * W_c) <= eps_on: hi = mid
            else: lo = mid
        backoff = hi; Wk = (1 - hi) * Wk + hi * W_c
    W["C_on"] = Wk; info["C_on"] = dict(time=time.time() - t0, backoff=backoff)
    W["centre"] = W_c
    return W, info

def run(j0, j1):
    rows = []
    for j in range(j0, j1):
        a = te[j]
        for n_bad, data, A_n, B_n in few_shot_data(j):
            W, info = fit_methods(A_n, B_n)
            res = {k: evaluate(W[k], L_means[a], 10_000 + j) for k in W if k != "C_on_scp"}
            for k, v in info.items():
                res[k].update(v)
            res["oracle"] = evaluate(W_all[a], L_means[a], 10_000 + j)
            rows.append(dict(j=j, task=int(a), n_bad=n_bad, res=res))
            print(f"j={j} n_bad={n_bad}: " + "  ".join(f"{k}: mmd {v['mmd']*1e3:.1f} pA {v['pA']:.3f} sref {v['s_ref']:.2f} son {v['s_on']:.2f}"
                  for k, v in res.items() if k in ("U_lam0.001", "S_gam1", "C_ref", "C_on", "oracle")), flush=True)
            json.dump(rows, open(f"{OUT}/part_c_ft_{j0}_{j1}.json", "w"), indent=1)
    print("done")

if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]))
