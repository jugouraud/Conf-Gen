"""
Order-1 Empirical Christoffel Function and ellipsoidal projection.

Uses a spectrally-regularized covariance: keep the top-m PCA eigenvalues intact
and model the remaining (d-m) directions as ISOTROPIC with variance γ (a
probabilistic-PCA covariance). This gives a Mahalanobis distance that penalizes
BOTH outliers within the safe subspace AND points that live outside it (the key
failure mode of pure PCA reduction).

Relation to the manuscript (meta_full_paper/ecf.tex). The paper regularizes the
moment matrix as M̃_{m,σ} = σI + M̃_m (uniform Tikhonov). Here the tail is
instead REPLACED by γ, which is a different estimator: with the paper's
σ ≈ 1e-11 an unobserved direction costs ~1e11 per unit², whereas γ (≈2.6 on the
CLIP prompt data) makes off-subspace travel merely expensive. The conformal
guarantee is unaffected — any score function fixed on the fit split is valid —
but Theorem 3.4 of devonport2023data is stated for M̃_{m,σ} and is NOT inherited.

The score is 1 + d_M², an affine, strictly increasing transform of the paper's
order-1 inverse ECF z₁ᵀM̃⁻¹z₁ = 1/N + ‖μ̂-x‖²_Σ̂⁻¹, so it induces exactly the same
family of sublevel sets and the same conformally-calibrated safe set.

The sublevel set is an ellipsoid in R^d. Projection onto it in the MAHALANOBIS
metric is closed-form (a radial contraction toward the mean); the EUCLIDEAN
projection is not closed-form and is not what this module computes.
"""

import warnings

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class EllipsoidalSafeSet:
    mean: torch.Tensor        # (d,)
    V: torch.Tensor           # (d, r) top-r eigenvectors (columns), r = min(m, rank)
    eig_reg: torch.Tensor     # (d,) regularized eigenvalues (tail = gamma)
    eig_inv: torch.Tensor     # (d,) 1/eig_reg
    alpha: float
    epsilon: float
    n_cal: int
    d: int                    # ambient dimension
    m: int                    # PCA cutoff
    gamma: float              # isotropic tail variance


def fit_regularized_covariance(
    embeddings: torch.Tensor,
    m: int,
    gamma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Fit the probabilistic-PCA covariance: top-m eigenvalues kept, tail set to γ.

    If γ=None, uses the mean of the discarded eigenvalues (Ledoit-Wolf-like).

    Returns (mean, V_top, eig_reg, eig_inv, gamma) where V_top is (d, r) with
    r = min(m, rank). The remaining d-r directions are isotropic at γ and are
    handled analytically, so no orthonormal completion of V is ever formed.
    """
    N, d = embeddings.shape
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean

    # Full SVD (truncated at min(N, d)). Note centering costs one degree of
    # freedom, so the last singular value is numerically zero.
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    eigvals = (S ** 2) / (N - 1)
    V = Vt.T  # (d, rank) eigenvectors as columns

    # Effective numerical rank. Centring costs one degree of freedom, so the
    # last singular value is ~0; keeping it as a "signal" direction would give
    # it eigenvalue ~0 and a Mahalanobis weight of 1/lambda ~ 1e16.
    tol = max(N, d) * float(torch.finfo(eigvals.dtype).eps) * eigvals[0]
    eff_rank = int((eigvals > tol).sum())
    rank = len(eigvals)

    # At least one direction must remain for the tail, or gamma is undefined.
    m_eff = min(m, max(eff_rank - 1, 1))
    if m_eff < m:
        warnings.warn(
            f"m={m} exceeds the usable rank of the fit sample (effective rank "
            f"{eff_rank} from N={N} points); capping at m={m_eff}. Without this "
            f"the isotropic tail gamma would collapse onto a numerically-zero "
            f"eigenvalue and inflate the Mahalanobis metric by orders of "
            f"magnitude.", RuntimeWarning, stacklevel=2)

    if gamma is None:
        tail = eigvals[m_eff:eff_rank]
        gamma = tail.mean().item() if len(tail) else eigvals[eff_rank - 1].item()

    gamma = max(gamma, 1e-8)

    r = m_eff
    eig_reg = torch.full((d,), gamma, dtype=embeddings.dtype)
    eig_reg[:r] = torch.maximum(eigvals[:r], torch.tensor(gamma, dtype=embeddings.dtype))
    eig_inv = 1.0 / eig_reg

    return mean, V[:, :r].contiguous(), eig_reg, eig_inv, gamma


def mahalanobis_sq(z: torch.Tensor, mean: torch.Tensor, V: torch.Tensor,
                   eig_inv: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
    """Mahalanobis² under Σ⁻¹ = γ⁻¹I + Σ_{i<r} (λ_i⁻¹ − γ⁻¹) v_i v_iᵀ.

    Exact for the isotropic-tail covariance, using only the r retained
    eigenvectors — no orthonormal completion, hence no Gram-Schmidt error.
    """
    diff = z - mean                          # (*, d)
    r = V.shape[1]
    if gamma is None:                        # tail value lives past the kept block
        gamma = float(eig_inv[-1].reciprocal())
    g_inv = 1.0 / gamma
    coeff = diff @ V                         # (*, r) coordinates in the retained eigenbasis
    iso = (diff ** 2).sum(dim=-1) * g_inv
    corr = (coeff ** 2 * (eig_inv[:r] - g_inv)).sum(dim=-1)
    return iso + corr


def ecf_score(z: torch.Tensor, safe_set: EllipsoidalSafeSet) -> torch.Tensor:
    """Order-1 empirical inverse Christoffel function: 1 + d_M²(z)."""
    return 1.0 + mahalanobis_sq(z, safe_set.mean, safe_set.V,
                                safe_set.eig_inv, safe_set.gamma)


def calibrate_conformal(scores_cal: torch.Tensor, epsilon: float) -> float:
    """Split-conformal threshold: the ⌈(K+1)(1−ε)⌉-th smallest calibration score.

    Follows content.tex:325 — sort the K calibration scores, append +∞ as the
    (K+1)-th element, and take r_{⌈(K+1)(1−ε)⌉}. When ⌈(K+1)(1−ε)⌉ > K the
    correct threshold IS +∞ (the safe set is all of R^d: vacuous but valid).
    Clamping to the largest finite score instead would silently void the
    coverage guarantee, which needs K ≥ ⌈1/ε⌉ − 1 to avoid.
    """
    n = len(scores_cal)
    sorted_scores, _ = torch.sort(scores_cal)
    rank = int(np.ceil((1 - epsilon) * (n + 1)))
    if rank > n:
        return float("inf")
    return sorted_scores[rank - 1].item()


def build_safe_set(
    embeddings_fit: torch.Tensor,
    embeddings_cal: torch.Tensor,
    epsilon: float = 0.05,
    m: int = 16,
    gamma: float | None = None,
) -> EllipsoidalSafeSet:
    """Build the ellipsoidal safe set with conformal coverage guarantee.

    Args:
        m: number of PCA components to keep at full variance
        gamma: isotropic tail variance for the remaining dimensions (None = auto)
    """
    mean, V, eig_reg, eig_inv, gamma_used = fit_regularized_covariance(embeddings_fit, m, gamma)

    ss = EllipsoidalSafeSet(
        mean=mean, V=V, eig_reg=eig_reg, eig_inv=eig_inv,
        alpha=0, epsilon=epsilon, n_cal=len(embeddings_cal),
        d=embeddings_fit.shape[1], m=m, gamma=gamma_used,
    )

    cal_scores = ecf_score(embeddings_cal, ss)
    ss.alpha = calibrate_conformal(cal_scores, epsilon)
    return ss


def is_safe(embeddings: torch.Tensor, safe_set: EllipsoidalSafeSet) -> torch.Tensor:
    return ecf_score(embeddings, safe_set) <= safe_set.alpha


def project_onto_ellipsoid(
    embeddings: torch.Tensor,
    safe_set: EllipsoidalSafeSet,
) -> torch.Tensor:
    """Project onto the safe ellipsoid in the MAHALANOBIS metric (closed-form).

    The ellipsoid is {h : (h−μ)ᵀΣ_reg⁻¹(h−μ) ≤ r²}, r² = α − 1. Minimising
    ‖h − z‖_{Σ⁻¹} over it is a radial contraction toward μ: scaling (z−μ) by s
    scales the score by s², so s = r / d_M(z) lands exactly on the boundary.

    This is NOT the Euclidean projection, which has no closed form on an
    ellipsoid (it needs a secular-equation root).
    """
    r_sq = safe_set.alpha - 1.0
    if not np.isfinite(r_sq):
        return embeddings.clone()          # vacuous safe set: everything is already inside
    if r_sq <= 0:
        raise ValueError(f"Degenerate safe set: α={safe_set.alpha} ≤ 1")
    r = np.sqrt(r_sq)

    d_m = torch.sqrt(mahalanobis_sq(embeddings, safe_set.mean, safe_set.V,
                                    safe_set.eig_inv, safe_set.gamma))
    # Inside → unchanged. Outside → contract to radius r(1−1e-6) so the
    # projected point lands strictly inside despite floating-point error.
    scale = torch.where(d_m > r, (r * (1 - 1e-6)) / d_m.clamp_min(1e-12),
                        torch.ones_like(d_m))
    return safe_set.mean + (embeddings - safe_set.mean) * scale.unsqueeze(-1)


def barrier(embeddings: torch.Tensor, safe_set: EllipsoidalSafeSet) -> torch.Tensor:
    return safe_set.alpha - ecf_score(embeddings, safe_set)
