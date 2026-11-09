"""Linear flow matching: v_theta(x, t) = W psi(x, t), W in R^{2 x F}, linear interpolant
x_t = (1 - t) x0 + t x1, x0 ~ N(0, I), target u = x1 - x0, Euler ODE sampler on [0, 1]."""
import numpy as np

sd_data, r_A, bad_std = 0.30, 0.6, 0.25
K = 50
tgrid = np.linspace(0.0, 1.0, K + 1)

def ring(phi, rho):
    ang = np.asarray(phi)[:, None] + np.arange(4)[None, :] * np.pi / 2
    return np.stack([np.asarray(rho)[:, None] * np.cos(ang), np.asarray(rho)[:, None] * np.sin(ang)], -1)

def sample_task(means, n, rng, m_bad=0.0):
    """(1 - m_bad) * equal-weight GMM(means, sd_data) + m_bad * N(0, bad_std^2 I)."""
    comp = rng.integers(0, means.shape[0], n)
    x = means[comp] + sd_data * rng.standard_normal((n, 2))
    if m_bad > 0:
        bad = rng.random(n) < m_bad
        x[bad] = bad_std * rng.standard_normal((bad.sum(), 2))
    return x

class Features:
    def __init__(self, F=512, seed=0):
        r = np.random.default_rng(seed)
        scales = r.choice([0.6, 1.2, 2.4, 4.8], F)
        self.Wx = r.standard_normal((F, 2)) * scales[:, None]
        self.wt = r.standard_normal(F) * 3.0
        self.b = r.uniform(0, 2 * np.pi, F)
        self.F = F
        self.dim = F + 6
    def __call__(self, x, t):
        t = np.broadcast_to(np.asarray(t, float), (x.shape[0],))
        rff = np.sqrt(2.0 / self.F) * np.cos(x @ self.Wx.T + t[:, None] * self.wt[None] + self.b[None])
        aff = np.c_[np.ones_like(t), t, x, t[:, None] * x]
        return np.c_[rff, aff]

def fm_moments(feat, x1, rng, reps=1):
    """Monte-Carlo moments of the flow-matching least squares for data x1 (each point used `reps` times)."""
    x1 = np.repeat(x1, reps, 0)
    n = x1.shape[0]
    x0 = rng.standard_normal((n, 2)); t = rng.random(n)
    xt = (1 - t)[:, None] * x0 + t[:, None] * x1
    Psi = feat(xt, t)
    return Psi.T @ Psi / n, (x1 - x0).T @ Psi / n        # A = E[psi psi^T] (D x D), B = E[u psi^T] (2 x D)

def ridge(A, B, lam):
    return np.linalg.solve(A + lam * np.eye(A.shape[0]), B.T).T   # W (2 x D)

def simulate(feat, W, x0, store_feats=False):
    """Euler ODE from t = 0 to 1. W: (2, D) or (F, 2, D) batch with x0 (n, 2) shared."""
    batched = W.ndim == 3
    Y = np.broadcast_to(x0, (W.shape[0],) + x0.shape).copy() if batched else x0.copy()
    feats = []
    for k in range(K):
        t, h = tgrid[k], tgrid[k + 1] - tgrid[k]
        if batched:
            Psi = feat(Y.reshape(-1, 2), t).reshape(Y.shape[0], Y.shape[1], -1)
            V = np.einsum("fnd,fcd->fnc", Psi, W)
        else:
            Psi = feat(Y, t); V = Psi @ W.T
        if store_feats: feats.append(Psi)
        Y = Y + h * V
    return (Y, feats) if store_feats else Y
