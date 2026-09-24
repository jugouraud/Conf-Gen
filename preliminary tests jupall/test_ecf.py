"""
Tests for the order-1 ECF safety filter.

Verifies: score correctness, conformal coverage, projection geometry,
barrier sign convention, and edge cases.
"""

import torch
import numpy as np
import pytest
from ecf import (
    build_safe_set, ecf_score, is_safe, project_onto_ellipsoid,
    barrier, mahalanobis_sq, fit_regularized_covariance,
)


def make_gaussian_data(n, d, seed=0):
    torch.manual_seed(seed)
    mean = torch.randn(d) * 0.5
    A = torch.randn(d, d) * 0.3
    cov = A @ A.T + 0.1 * torch.eye(d)
    L = torch.linalg.cholesky(cov)
    data = torch.randn(n, d) @ L.T + mean
    return data, mean, cov


class TestECFScore:
    def test_score_at_mean_is_one(self):
        data, _, _ = make_gaussian_data(100, 10)
        ss = build_safe_set(data[:70], data[70:], epsilon=0.05, m=5)
        score = ecf_score(ss.mean.unsqueeze(0), ss)
        assert abs(score.item() - 1.0) < 1e-5

    def test_score_increases_with_distance(self):
        data, _, _ = make_gaussian_data(200, 10)
        ss = build_safe_set(data[:140], data[140:], epsilon=0.05, m=5)
        close = ss.mean + 0.1 * torch.randn(10)
        far = ss.mean + 10.0 * torch.randn(10)
        assert ecf_score(close.unsqueeze(0), ss).item() < ecf_score(far.unsqueeze(0), ss).item()

    def test_score_nonnegative(self):
        data, _, _ = make_gaussian_data(100, 10)
        ss = build_safe_set(data[:70], data[70:], epsilon=0.05, m=5)
        scores = ecf_score(data, ss)
        assert (scores >= 1.0 - 1e-5).all()

    def test_batch_consistency(self):
        data, _, _ = make_gaussian_data(100, 10)
        ss = build_safe_set(data[:70], data[70:], epsilon=0.05, m=5)
        batch_scores = ecf_score(data[:5], ss)
        for i in range(5):
            single = ecf_score(data[i:i+1], ss)
            assert abs(batch_scores[i].item() - single.item()) < 1e-4


class TestConformalCoverage:
    def test_coverage_guarantee(self):
        """The conformal guarantee: P[h_new ∈ S] ≥ 1 - ε, verified empirically."""
        torch.manual_seed(42)
        n, d = 500, 20
        data, _, _ = make_gaussian_data(n, d)
        epsilon = 0.10
        n_trials = 200
        coverages = []

        for _ in range(n_trials):
            perm = torch.randperm(n)
            fit = data[perm[:250]]
            cal = data[perm[250:375]]
            test = data[perm[375:]]

            ss = build_safe_set(fit, cal, epsilon=epsilon, m=10)
            covered = is_safe(test, ss).float().mean().item()
            coverages.append(covered)

        mean_cov = np.mean(coverages)
        # Empirical coverage should be ≥ 1 - ε on average
        assert mean_cov >= 1 - epsilon - 0.02, f"Coverage {mean_cov:.3f} < {1-epsilon-0.02}"

    def test_alpha_monotone_in_epsilon(self):
        """Larger ε → smaller α (tighter set)."""
        data, _, _ = make_gaussian_data(200, 10)
        fit, cal = data[:140], data[140:]
        alphas = []
        for eps in [0.01, 0.05, 0.10, 0.20]:
            ss = build_safe_set(fit, cal, epsilon=eps, m=5)
            alphas.append(ss.alpha)
        for i in range(len(alphas) - 1):
            assert alphas[i] >= alphas[i+1], f"α not monotone: {alphas}"


class TestProjection:
    def test_interior_points_unchanged(self):
        data, _, _ = make_gaussian_data(200, 10)
        ss = build_safe_set(data[:140], data[140:], epsilon=0.05, m=5)
        inside_mask = is_safe(data, ss)
        if inside_mask.any():
            inside_pts = data[inside_mask]
            projected = project_onto_ellipsoid(inside_pts, ss)
            diff = (inside_pts - projected).norm(dim=-1)
            assert diff.max().item() < 1e-5, f"Interior points moved: max_diff={diff.max()}"

    def test_projected_points_are_safe(self):
        data, _, _ = make_gaussian_data(200, 10)
        ss = build_safe_set(data[:140], data[140:], epsilon=0.05, m=5)
        # Generate far-out points
        outliers = ss.mean + 50.0 * torch.randn(20, 10)
        projected = project_onto_ellipsoid(outliers, ss)
        assert is_safe(projected, ss).all(), "Some projected points are not safe"

    def test_projection_lands_near_boundary(self):
        """Projected exterior points should have score ≈ α."""
        data, _, _ = make_gaussian_data(200, 10)
        ss = build_safe_set(data[:140], data[140:], epsilon=0.05, m=5)
        outliers = ss.mean + 50.0 * torch.randn(20, 10)
        projected = project_onto_ellipsoid(outliers, ss)
        scores = ecf_score(projected, ss)
        # Should be close to α (within the 1e-3 tolerance)
        rel_error = (scores - ss.alpha).abs() / ss.alpha
        assert rel_error.max().item() < 0.01, f"Max relative error: {rel_error.max()}"

    def test_projection_reduces_score(self):
        data, _, _ = make_gaussian_data(200, 10)
        ss = build_safe_set(data[:140], data[140:], epsilon=0.05, m=5)
        outliers = ss.mean + 50.0 * torch.randn(20, 10)
        orig_scores = ecf_score(outliers, ss)
        projected = project_onto_ellipsoid(outliers, ss)
        proj_scores = ecf_score(projected, ss)
        assert (proj_scores <= orig_scores + 1e-3).all()

    def test_projection_is_closest_point(self):
        """Projection should be approximately the closest point on the ellipsoid boundary."""
        torch.manual_seed(7)
        data, _, _ = make_gaussian_data(300, 5)
        ss = build_safe_set(data[:200], data[200:], epsilon=0.05, m=3)
        outlier = ss.mean + 20.0 * torch.randn(5)
        projected = project_onto_ellipsoid(outlier.unsqueeze(0), ss).squeeze(0)

        # Check that small perturbations of the projected point are farther from the outlier
        dist_proj = (outlier - projected).norm()
        for _ in range(100):
            perturb = projected + 0.01 * torch.randn(5)
            if is_safe(perturb.unsqueeze(0), ss).item():
                dist_perturb = (outlier - perturb).norm()
                # Projected point should be at least as close (up to perturbation size)
                assert dist_proj <= dist_perturb + 0.02


class TestBarrier:
    def test_positive_inside_negative_outside(self):
        data, _, _ = make_gaussian_data(200, 10)
        ss = build_safe_set(data[:140], data[140:], epsilon=0.05, m=5)
        inside_mask = is_safe(data, ss)
        b = barrier(data, ss)
        if inside_mask.any():
            assert (b[inside_mask] >= -1e-5).all(), "Barrier negative inside safe set"
        outside_mask = ~inside_mask
        if outside_mask.any():
            assert (b[outside_mask] <= 1e-5).all(), "Barrier positive outside safe set"


class TestSpectralRegularization:
    def test_eigenvalue_floor(self):
        data, _, _ = make_gaussian_data(100, 50)
        m, gamma = 10, 0.5
        mean, V, eig_reg, eig_inv, g = fit_regularized_covariance(data, m, gamma)
        assert (eig_reg[m:] == gamma).all(), "Eigenvalues below m not floored at γ"
        assert (eig_reg[:m] >= gamma).all(), "Top eigenvalues below γ"

    def test_auto_gamma(self):
        data, _, _ = make_gaussian_data(100, 50)
        mean, V, eig_reg, eig_inv, g = fit_regularized_covariance(data, 10, gamma=None)
        assert g > 0, "Auto γ should be positive"

    def test_full_rank_even_underdetermined(self):
        """With N < d, the regularized covariance should still be full rank."""
        data = torch.randn(20, 100)
        ss = build_safe_set(data[:14], data[14:], epsilon=0.10, m=5, gamma=0.1)
        assert (ss.eig_reg > 0).all()
        scores = ecf_score(data, ss)
        assert torch.isfinite(scores).all()


class TestEdgeCases:
    def test_single_calibration_point(self):
        data, _, _ = make_gaussian_data(100, 10)
        ss = build_safe_set(data[:99], data[99:], epsilon=0.05, m=5)
        assert ss.alpha > 0

    def test_high_epsilon(self):
        """ε close to 1 should give a very tight set."""
        data, _, _ = make_gaussian_data(200, 10)
        ss_tight = build_safe_set(data[:140], data[140:], epsilon=0.50, m=5)
        ss_loose = build_safe_set(data[:140], data[140:], epsilon=0.01, m=5)
        assert ss_tight.alpha <= ss_loose.alpha

    def test_identical_points(self):
        """All identical points: covariance is zero, regularization should save us."""
        data = torch.ones(50, 10)
        data += 1e-8 * torch.randn_like(data)  # tiny perturbation
        ss = build_safe_set(data[:35], data[35:], epsilon=0.05, m=3, gamma=1.0)
        assert torch.isfinite(ecf_score(data[:1], ss)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
