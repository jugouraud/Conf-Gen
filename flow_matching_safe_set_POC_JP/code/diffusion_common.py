"""Pilot A process and exact Gaussian-mixture scores (verbatim copies of the definitions in part_a.py),
for the figure scripts.  VP forward SDE with beta = 1; Euler-Maruyama reverse chain on a quadratic grid."""
import numpy as np

T, t_min, K = 6.0, 1e-2, 50
ts = t_min + (T - t_min) * np.linspace(1.0, 0.0, K + 1) ** 2   # t_0 = T > ... > t_K = t_min
hs = ts[:-1] - ts[1:]
alpha = np.exp(-0.5 * ts)
sig2 = 1.0 - alpha ** 2
sd_data, r_A, bad_std = 0.30, 0.6, 0.25

def ring(phi, rho):
    ang = phi[:, None] + np.arange(4)[None, :] * np.pi / 2
    return np.stack([rho[:, None] * np.cos(ang), rho[:, None] * np.sin(ang)], -1)

def fields_legit(means):
    P = means.shape[0]
    return means, np.full((P, 4), sd_data), np.full((P, 4), np.log(0.25))

def fields_bad(means, m):
    P = means.shape[0]
    Mb = np.concatenate([means, np.zeros((P, 1, 2))], 1)
    Sb = np.concatenate([np.full((P, 4), sd_data), np.full((P, 1), bad_std)], 1)
    Wb = np.log(np.concatenate([np.full((P, 4), (1 - m) / 4), np.full((P, 1), m)], 1))
    return Mb, Sb, Wb

def score_all(x, k, means, stds, logw):
    a, s2 = alpha[k], sig2[k]
    v = a ** 2 * stds ** 2 + s2
    diff = a * means[:, :, None, :] - x[None, None]
    lp = logw[:, :, None] - np.log(v)[:, :, None] - (diff ** 2).sum(-1) / (2 * v[:, :, None])
    lp -= lp.max(1, keepdims=True)
    r = np.exp(lp); r /= r.sum(1, keepdims=True)
    return (r[..., None] * diff / v[:, :, None, None]).sum(1)

def score_diag(Y, k, means, stds, logw):
    a, s2 = alpha[k], sig2[k]
    v = a ** 2 * stds ** 2 + s2
    diff = a * means[:, :, None, :] - Y[:, None]
    lp = logw[:, :, None] - np.log(v)[:, :, None] - (diff ** 2).sum(-1) / (2 * v[:, :, None])
    lp -= lp.max(1, keepdims=True)
    r = np.exp(lp); r /= r.sum(1, keepdims=True)
    return (r[..., None] * diff / v[:, :, None, None]).sum(1)

def simulate(means, stds, logw, Z0, Z, store=False, chunk=50):
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
