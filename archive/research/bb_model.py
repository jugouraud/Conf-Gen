"""Blackbox meta-learning score: a small network predicts the embedding.

This is g_d^BB from adaptation.tex (eq:bb_score, pr:bb_coverage), the one
estimator family in the manuscript not previously ported:

    g_d^BB(tau, theta; D_train)  :=  d( theta,  psi_omega(tau) )

with psi_omega : T -> Theta a meta-model trained on D_train, and the conformal
radius calibrated on a disjoint D_cal. Unlike the Wasserstein scores it needs
no reference set at inference: the whole corpus is compressed into the weights.

WHAT THE NETWORK MAY SEE
------------------------
psi must NOT take theta as input. theta is CLIP's pooled [EOS] state, a
deterministic function of the prompt, so a network given it would learn the
identity and the score would be zero everywhere. The faithful input is the TASK
DESCRIPTOR tau of README section 2.2 -- the bag of CONTENT-token hidden states.
That is a genuine prediction problem: tau is unordered and unprocessed, theta is
the output of twelve layers of causal attention, and section 11.5 measured how
much they differ.

The score is then a reconstruction residual. Trained only on safe prompts, psi
learns the token-bag -> summary map *restricted to the safe manifold*; it
extrapolates badly off it, and the residual is the novelty signal. Capacity is
what makes this work, so the network is deliberately small.

GEOMETRY OF THE SCORE
---------------------
The safe set is { theta : ||theta - psi(tau)|| <= eps }: a single BALL centred
on the prediction. That is the simplest region in the taxonomy of section 5 --
projection onto it is one line, with none of the union-of-balls or
sum-of-norms machinery.

ARCHITECTURE
------------
A DeepSets encoder, which is the right inductive bias for an unordered bag:

    per token  ->  phi  ->  mean & max pool  ->  rho  ->  predicted theta

Inputs and outputs live in whitened coordinates, so the Euclidean distance the
score uses IS the Mahalanobis metric the rest of the project uses. The output is
the top-m whitened coordinates only; the score adds back theta's own
off-subspace residual, which section 11's metric sweep showed is what catches
out-of-domain prompts (truncating it collapsed degenerate detection to 0.00).

    ~58k parameters, 0.007% of the 860M-parameter U-Net, <1 ms per prompt.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SetPredictor(nn.Module):
    """DeepSets psi_omega: a bag of token states -> the predicted embedding.

    Permutation-invariant by construction (mean and max are), which matches the
    fact that tau carries no order. Mean and max are concatenated because mean
    alone loses the "is any token extreme" signal that novelty depends on.
    """

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x, mask):
        """x: (B, L, d_in) padded token features; mask: (B, L) 1 for real tokens."""
        h = self.phi(x)                                   # (B, L, H)
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        mx = h.masked_fill(m == 0, -1e9).max(dim=1).values
        return self.rho(torch.cat([mean, mx], dim=-1))

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class BBScore:
    """The fitted blackbox score g_d^BB, ready to evaluate on new prompts."""

    def __init__(self, net, tok_wm, th_wm, m_out, L_max, mu_in, sd_in, device="cpu"):
        self.net, self.tok_wm, self.th_wm = net, tok_wm, th_wm
        self.m_out, self.L_max = m_out, L_max
        self.mu_in, self.sd_in = mu_in, sd_in
        self.device = device

    # ---- feature construction -------------------------------------------

    def _token_features(self, clouds):
        """Bag of tokens -> padded (B, L_max, d_in) plus mask.

        Per token: its top-m whitened coordinates, plus log1p of the whitened
        off-subspace residual norm so the network can see energy in directions
        the coordinates discard.
        """
        B, L = len(clouds), self.L_max
        V = self.tok_wm.V
        d_in = V.shape[1] + 1
        X = np.zeros((B, L, d_in), dtype=np.float32)
        M = np.zeros((B, L), dtype=np.float32)
        for i, c in enumerate(clouds):
            w = self.tok_wm(np.atleast_2d(c))[:L]
            coef = w @ V
            resid = np.sqrt(np.maximum((w ** 2).sum(1) - (coef ** 2).sum(1), 0.0))
            f = np.concatenate([coef, np.log1p(resid)[:, None]], axis=1)
            X[i, :len(f)] = (f - self.mu_in) / self.sd_in
            M[i, :len(f)] = 1.0
        return torch.tensor(X), torch.tensor(M)

    def _theta_parts(self, theta):
        """theta -> (top-m whitened coords, off-subspace whitened residual norm)."""
        w = self.th_wm(np.atleast_2d(np.asarray(theta, dtype=float)))
        V = self.th_wm.V
        coef = w @ V
        resid = np.sqrt(np.maximum((w ** 2).sum(1) - (coef ** 2).sum(1), 0.0))
        return coef, resid

    # ---- the score -------------------------------------------------------

    @torch.no_grad()
    def predict(self, clouds):
        X, M = self._token_features(clouds)
        self.net.eval()
        return self.net(X.to(self.device), M.to(self.device)).cpu().numpy()

    def __call__(self, theta, clouds):
        """g_d^BB = || theta - psi(tau) || in the whitened (Mahalanobis) metric.

        The prediction lives in the retained subspace, so theta's own
        off-subspace residual enters the distance in full -- that term is what
        detects prompts that leave the safe manifold entirely.
        """
        coef, resid = self._theta_parts(theta)
        pred = self.predict(clouds)
        return np.sqrt(((coef - pred) ** 2).sum(1) + resid ** 2)

    def in_subspace_error(self, theta, clouds):
        """The learned part of the residual alone, for diagnostics."""
        coef, _ = self._theta_parts(theta)
        return np.linalg.norm(coef - self.predict(clouds), axis=1)


def fit_bb(theta_train, clouds_train, tok_wm, th_wm, epochs=400, lr=3e-3,
           d_hidden=128, val_frac=0.15, weight_decay=1e-4, patience=40,
           batch=64, seed=0, device="cpu", verbose=True):
    """Train psi_omega on (task, weight) pairs from the SAFE corpus only.

    theta_train  : (N, 768) pooled embeddings
    clouds_train : list of (L_i, 768) content-token clouds
    tok_wm/th_wm : whiteners for the token space and the embedding space,
                   fitted on data disjoint from calibration and test

    Early stopping on a held-out slice of the training set. The calibration set
    must NOT be used here -- pr:bb_coverage needs D_train and D_cal disjoint.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = len(theta_train)
    L_max = max(len(c) for c in clouds_train)

    # provisional scorer just to build features, then standardise the inputs
    probe = BBScore(None, tok_wm, th_wm, th_wm.V.shape[1], L_max,
                    np.zeros(tok_wm.V.shape[1] + 1), np.ones(tok_wm.V.shape[1] + 1))
    X, M = probe._token_features(clouds_train)
    flat = X[M.bool()].numpy()
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    probe.mu_in, probe.sd_in = mu, sd
    X, M = probe._token_features(clouds_train)
    Y = torch.tensor(probe._theta_parts(theta_train)[0], dtype=torch.float32)

    perm = rng.permutation(N)
    n_val = max(8, int(val_frac * N))
    va, tr = perm[:n_val], perm[n_val:]

    net = SetPredictor(X.shape[-1], Y.shape[-1], d_hidden=d_hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, Mt, Yt = X[tr].to(device), M[tr].to(device), Y[tr].to(device)
    Xv, Mv, Yv = X[va].to(device), M[va].to(device), Y[va].to(device)

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        idx = torch.randperm(len(tr), device=device)
        for k in range(0, len(tr), batch):
            b = idx[k:k + batch]
            opt.zero_grad()
            loss = ((net(Xt[b], Mt[b]) - Yt[b]) ** 2).sum(1).mean()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(Xv, Mv) - Yv) ** 2).sum(1).mean())
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 50 == 0:
            print(f"      epoch {ep:>4}  train {float(loss):.4f}  val {v:.4f}")

    if best_state is not None:
        net.load_state_dict(best_state)
    if verbose:
        print(f"      stopped at epoch {ep}, best val MSE {best:.4f}, "
              f"{net.n_params} parameters")
    return BBScore(net, tok_wm, th_wm, Y.shape[-1], L_max, mu, sd, device=device)
