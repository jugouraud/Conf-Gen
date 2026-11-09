"""Pilot A: order-1 conformal regions on score fields of a discretised reverse SDE.

Fields are exact scores of 2-D Gaussian mixtures (tasks = rotated/scaled 4-blob rings).
Generative chain (VP, beta = 1, Euler-Maruyama, same noise law for every field):
    Y_{k+1} = Y_k + h_k (Y_k / 2 + s(Y_k, t_k)) + sqrt(h_k) xi_k,   Y_0 ~ N(0, I).
For two such chains the path KL is exact:
    KL(P^a || P^b) = 1/2 sum_k h_k E_{P^a} || s_a(Y_k, t_k) - s_b(Y_k, t_k) ||^2 =: G[a, b] / 2.
Costs between a candidate a and a historical field b:
    forward  c_F = sqrt(G[b, a])   (probes from the historical chain, convex in s_a)
    reverse  c_R = sqrt(G[a, b])   (probes from the candidate's own chain)
    rev-KL   c_K = G[a, b] / 2
Order-1 score: S(a) = 1/(M(M+1)) sum_{b in train} c(a, b); epsilon by split conformal.
Unsafe event A = {||Y_K|| <= r_A} (the ring's centre).
"""
import os
import numpy as np, time, json, sys

rng_master = np.random.default_rng(0)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# ---------------- process ----------------
T, t_min, K = 6.0, 1e-2, 50
ts = t_min + (T - t_min) * np.linspace(1.0, 0.0, K + 1) ** 2   # t_0 = T > ... > t_K = t_min
hs = ts[:-1] - ts[1:]
alpha = np.exp(-0.5 * ts)
sig2 = 1.0 - alpha ** 2

# ---------------- task family ----------------
P = 200                     # pool of legit tasks
sd_data = 0.30              # blob std
s_phi, s_rho = 0.05, 0.05   # rotation / radius spread
r_A = 0.6                   # unsafe disk radius
bad_std = 0.25              # std of the unsafe blob at the origin
masses = [0.002, 0.01, 0.03, 0.1, 0.3]
n_bad_bases = 100

def ring(phi, rho):
    ang = phi[:, None] + np.arange(4)[None, :] * np.pi / 2
    return np.stack([rho[:, None] * np.cos(ang), rho[:, None] * np.sin(ang)], -1)

phi = rng_master.normal(0, s_phi, P)
rho = rng_master.normal(2.0, s_rho, P)
L_means = ring(phi, rho)                       # (P, 4, 2)
L_stds = np.full((P, 4), sd_data)
L_logw = np.full((P, 4), np.log(0.25))

# contaminated fields: (1 - m) * task + m * N(0, bad_std^2 I)
B_means, B_stds, B_logw, B_base, B_mass = [], [], [], [], []
for a in range(n_bad_bases):
    for m in masses:
        B_means.append(np.vstack([L_means[a], np.zeros((1, 2))]))
        B_stds.append(np.r_[L_stds[a], bad_std])
        B_logw.append(np.log(np.r_[np.full(4, (1 - m) / 4), m]))
        B_base.append(a); B_mass.append(m)
B_means, B_stds, B_logw = map(np.array, (B_means, B_stds, B_logw))
B_base, B_mass = np.array(B_base), np.array(B_mass)

# ---------------- scores ----------------
def score_all(x, k, means, stds, logw):
    """Every field evaluated at the same points x (N, 2) -> (F, N, 2)."""
    a, s2 = alpha[k], sig2[k]
    v = a ** 2 * stds ** 2 + s2                              # (F, J)
    diff = a * means[:, :, None, :] - x[None, None]          # (F, J, N, 2)
    lp = logw[:, :, None] - np.log(v)[:, :, None] - (diff ** 2).sum(-1) / (2 * v[:, :, None])
    lp -= lp.max(1, keepdims=True)
    r = np.exp(lp); r /= r.sum(1, keepdims=True)
    return (r[..., None] * diff / v[:, :, None, None]).sum(1)

def score_diag(Y, k, means, stds, logw):
    """Field f evaluated at its own points Y[f] (F, N, 2) -> (F, N, 2)."""
    a, s2 = alpha[k], sig2[k]
    v = a ** 2 * stds ** 2 + s2
    diff = a * means[:, :, None, :] - Y[:, None]
    lp = logw[:, :, None] - np.log(v)[:, :, None] - (diff ** 2).sum(-1) / (2 * v[:, :, None])
    lp -= lp.max(1, keepdims=True)
    r = np.exp(lp); r /= r.sum(1, keepdims=True)
    return (r[..., None] * diff / v[:, :, None, None]).sum(1)

def simulate(means, stds, logw, Z0, Z, store=False, chunk=50):
    """Chains for every field with common random numbers Z0 (n,2), Z (K,n,2)."""
    F, n = means.shape[0], Z0.shape[0]
    finals = np.empty((F, n, 2))
    traj = np.empty((K, F, n, 2)) if store else None
    for f0 in range(0, F, chunk):
        sl = slice(f0, min(F, f0 + chunk))
        Y = np.broadcast_to(Z0, (sl.stop - sl.start, n, 2)).copy()
        for k in range(K):
            if store:
                traj[k, sl] = Y
            s = score_diag(Y, k, means[sl], stds[sl], logw[sl])
            Y = Y + hs[k] * (0.5 * Y + s) + np.sqrt(hs[k]) * Z[k][None]
        finals[sl] = Y
    return finals, traj

n_probe, n_big = 256, 20000
rng = np.random.default_rng(1)
Z0p, Zp = rng.standard_normal((n_probe, 2)), rng.standard_normal((K, n_probe, 2))
Z0b, Zb = rng.standard_normal((n_big, 2)), rng.standard_normal((K, n_big, 2))

t0 = time.time()
_, trajL = simulate(L_means, L_stds, L_logw, Z0p, Zp, store=True)
_, trajB = simulate(B_means, B_stds, B_logw, Z0p, Zp, store=True)
print(f"probe chains {time.time()-t0:.1f}s", flush=True)

def G_cross(traj_src, src_fields, tgt_fields, chunk=10):
    """G[a, b] = sum_k h_k E_{chain a} ||s_a - s_b||^2 for a in src, b in tgt (probes of a)."""
    Ms, Ss, Ws = src_fields; Mt, St, Wt = tgt_fields
    Fa, Fb = Ms.shape[0], Mt.shape[0]
    G = np.zeros((Fa, Fb))
    for a0 in range(0, Fa, chunk):
        sl = slice(a0, min(Fa, a0 + chunk)); na = sl.stop - sl.start
        for k in range(K):
            X = traj_src[k, sl]                                  # (na, n, 2)
            sa = score_diag(X, k, Ms[sl], Ss[sl], Ws[sl])        # (na, n, 2)
            sb = score_all(X.reshape(-1, 2), k, Mt, St, Wt).reshape(Fb, na, -1, 2)
            G[sl] += hs[k] * ((sb - sa[None]) ** 2).sum(-1).mean(-1).T
    return G

def G_fwd(traj_hist, hist_fields, cand_fields, chunk=10):
    """F[b, f] = sum_k h_k E_{chain b} ||s_f - s_b||^2 (probes of the historical chain b)."""
    Mh, Sh, Wh = hist_fields; Mc, Sc, Wc = cand_fields
    Fh, Fc = Mh.shape[0], Mc.shape[0]
    G = np.zeros((Fh, Fc))
    for b0 in range(0, Fh, chunk):
        sl = slice(b0, min(Fh, b0 + chunk)); nb = sl.stop - sl.start
        for k in range(K):
            X = traj_hist[k, sl]
            sb = score_diag(X, k, Mh[sl], Sh[sl], Wh[sl])
            sc = score_all(X.reshape(-1, 2), k, Mc, Sc, Wc).reshape(Fc, nb, -1, 2)
            G[sl] += hs[k] * ((sc - sb[None]) ** 2).sum(-1).mean(-1).T
    return G

Lf, Bf = (L_means, L_stds, L_logw), (B_means, B_stds, B_logw)
t0 = time.time(); G_LL = G_cross(trajL, Lf, Lf); print(f"G_LL {time.time()-t0:.1f}s", flush=True)
t0 = time.time(); G_BL = G_cross(trajB, Bf, Lf); print(f"G_BL (reverse, bad->legit) {time.time()-t0:.1f}s", flush=True)
t0 = time.time(); G_LB = G_fwd(trajL, Lf, Bf); print(f"G_LB (forward, legit probes) {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
finL, _ = simulate(L_means, L_stds, L_logw, Z0b, Zb)
finB, _ = simulate(B_means, B_stds, B_logw, Z0b, Zb)
pA_L = (np.linalg.norm(finL, axis=-1) <= r_A).mean(1)
pA_B = (np.linalg.norm(finB, axis=-1) <= r_A).mean(1)
print(f"big chains {time.time()-t0:.1f}s", flush=True)

np.savez(f"{OUT}/part_a_cache.npz", G_LL=G_LL, G_BL=G_BL, G_LB=G_LB, pA_L=pA_L, pA_B=pA_B,
         B_base=B_base, B_mass=B_mass, ts=ts, hs=hs, phi=phi, rho=rho)
print("saved")
