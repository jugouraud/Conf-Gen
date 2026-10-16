"""E1: prompt-conditioned linear flow matching on the 2-D toy (safe_prompt_fields.tex, Section 9).

Prompt z = (phi, rho, m): rotation, radius and unsafe mass of the target
    p_z = (1 - m) Ring(phi, rho) + m N(0, 0.25^2 I),
    Ring(phi, rho) = equal mixture of N(rho (cos, sin)(phi + j pi / 2), 0.3^2 I), j = 0..3.
A frozen "text encoder" maps z to an embedding e(z) in R^{D_e}; the model is affine in it,
    v(x, t; e) = W(e) psi(x, t),   W(e) = W_0 + sum_j e_j W_j       (Definition 3.11(b)),
fitted once by least squares on (prompt, sample) pairs (independent coupling: 1-rectified),
then refitted on its own couplings (reflow: 2-rectified).  Field coordinates are W(e) in R^{2 x Fpsi}.
"""
import numpy as np
from lfm_common import Features, ring, sample_task, r_A

PHI_MAX = np.pi / 8          # safe rotations: phi ~ U[-pi/8, pi/8]
RHO_RANGE = (1.75, 2.25)     # safe radii
BAD_STD = 0.25               # unsafe component N(0, 0.25^2 I)


class Encoder:
    """Frozen text encoder e(z) = [c, sqrt(2/Dr) cos(Om c + beta)], c = (phi/PHI_MAX, (rho-2)/0.25, m/0.2)."""
    def __init__(self, Dr=20, seed=7, scale=1.0):
        r = np.random.default_rng(seed)
        self.Om = r.standard_normal((Dr, 3)) * scale
        self.beta = r.uniform(0.0, 2 * np.pi, Dr)
        self.Dr, self.dim = Dr, 3 + Dr

    def __call__(self, z):
        z = np.atleast_2d(z)
        c = np.c_[z[:, 0] / PHI_MAX, (z[:, 1] - 2.0) / 0.25, z[:, 2] / 0.2]
        return np.c_[c, np.sqrt(2.0 / self.Dr) * np.cos(c @ self.Om.T + self.beta)]


def sample_target(z, n, rng):
    """n samples of p_z."""
    return sample_task(ring([z[0]], [z[1]])[0], n, rng, m_bad=float(z[2]))


def safe_prompts(n, rng):
    return np.c_[rng.uniform(-PHI_MAX, PHI_MAX, n), rng.uniform(*RHO_RANGE, n), np.zeros(n)]


def train_prompts(n, rng, unsafe_frac=0.5, m_max=0.4, widen=1.1):
    """Training prompts cover the safe family (slightly widened) and unsafe masses m ~ U[0, m_max]."""
    phi = rng.uniform(-widen * PHI_MAX, widen * PHI_MAX, n)
    rho = rng.uniform(RHO_RANGE[0] - 0.05, RHO_RANGE[1] + 0.05, n)
    m = np.where(rng.random(n) < unsafe_frac, rng.uniform(0.0, m_max, n), 0.0)
    return np.c_[phi, rho, m]


def fit_conditional(feat, E, pairs, rng, reps=4, lam_rel=1e-7, chunk=100):
    """Least squares for v(x, t; e) = W(e) psi(x, t) on (prompt, sample) pairs.
    pairs(j, rng) -> (x0, x1), each (n, 2): a coupling sample for prompt j; each pair gets `reps` times t.
    Returns Wstack (D_e + 1, 2, Fpsi) with W(e) = Wstack[0] + sum_j e_j Wstack[j]."""
    P, De1, Fp = E.shape[0], E.shape[1] + 1, feat.dim
    Et = np.c_[np.ones(P), E]
    A = np.zeros((De1 * De1, Fp * Fp)); B = np.zeros((De1, Fp, 2)); N = 0
    for j0 in range(0, P, chunk):
        idx = np.arange(j0, min(P, j0 + chunk))
        S = np.zeros((len(idx), Fp, Fp))
        for jj, j in enumerate(idx):
            x0, x1 = pairs(j, rng)
            x0, x1 = np.repeat(x0, reps, 0), np.repeat(x1, reps, 0)
            t = rng.random(len(x0))
            Psi = feat((1 - t)[:, None] * x0 + t[:, None] * x1, t)
            S[jj] = Psi.T @ Psi
            B += Et[j][:, None, None] * (Psi.T @ (x1 - x0))[None]
            N += len(x0)
        E2 = (Et[idx][:, :, None] * Et[idx][:, None, :]).reshape(len(idx), -1)
        A += E2.T @ S.reshape(len(idx), -1)
    A = A.reshape(De1, De1, Fp, Fp).transpose(0, 2, 1, 3).reshape(De1 * Fp, De1 * Fp) / N
    b = B.reshape(De1 * Fp, 2) / N
    lam = lam_rel * np.trace(A) / A.shape[0]
    w = np.linalg.solve(A + lam * np.eye(A.shape[0]), b)
    return w.reshape(De1, Fp, 2).transpose(0, 2, 1)


def fields(Wstack, E):
    """W(e) for a batch of embeddings E (P, D_e) -> (P, 2, Fpsi)."""
    Et = np.c_[np.ones(len(E)), E]
    return np.einsum("pa,acf->pcf", Et, Wstack)


def simulate(feat, Ws, x0, J, chunk=16):
    """J-step Euler sampler (Definition 3.4) for fields Ws (P, 2, Fpsi) from noise x0 ((n, 2) shared, or (P, n, 2))."""
    grid = np.linspace(0.0, 1.0, J + 1)
    P = Ws.shape[0]; shared = x0.ndim == 2
    out = np.empty((P,) + (x0.shape if shared else x0.shape[1:]))
    for p0 in range(0, P, chunk):
        W = Ws[p0:p0 + chunk]
        Y = (np.broadcast_to(x0, (W.shape[0],) + x0.shape) if shared else x0[p0:p0 + chunk]).copy()
        WT = W.transpose(0, 2, 1)
        for k in range(J):
            Psi = feat(Y.reshape(-1, 2), grid[k]).reshape(Y.shape[0], Y.shape[1], -1)
            Y += (grid[k + 1] - grid[k]) * np.matmul(Psi, WT)
        out[p0:p0 + chunk] = Y
    return out


def one_step(feat, Ws, x0):
    """One-step map T^1(x) = x + v(x, 0) for fields Ws (P, 2, Fpsi) and shared noise x0 (n, 2)."""
    return x0[None] + np.matmul(feat(x0, 0.0)[None], Ws.transpose(0, 2, 1))


def path_gram(feat, z, n, rng):
    """E[psi psi^T] under the interpolation path of the true target p_z (independent coupling)."""
    x1 = sample_target(z, n, rng); x0 = rng.standard_normal((n, 2)); t = rng.random(n)
    Psi = feat((1 - t)[:, None] * x0 + t[:, None] * x1, t)
    return Psi.T @ Psi / n


def mmd2(x, y, bws=(0.1, 0.3, 1.0)):
    """Unbiased squared MMD, kernel sum_s exp(-|x - y|^2 / (2 s^2)) (as in the POC)."""
    def k(a, b):
        d2 = ((a[:, None] - b[None]) ** 2).sum(-1)
        return sum(np.exp(-d2 / (2 * s * s)) for s in bws)
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y); n, m = len(x), len(y)
    return (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()


def kth_threshold(cal_scores, beta):
    """C_{1-beta}: the ceil((K+1)(1-beta))-th smallest calibration score, +inf if that index exceeds K."""
    K = len(cal_scores); k = int(np.ceil((K + 1) * (1 - beta)))
    return np.sort(cal_scores)[k - 1] if k <= K else np.inf
