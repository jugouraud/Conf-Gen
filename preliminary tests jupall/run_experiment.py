"""
End-to-end experiment: build a conformal safe set on CLIP conditioning
embeddings, test detection of degenerate prompts, and demonstrate
the ellipsoidal projection.

Uses spectrally-regularized covariance so the ECF detects both
within-subspace outliers AND out-of-subspace (degenerate) embeddings.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ecf import build_safe_set, ecf_score, is_safe, project_onto_ellipsoid, barrier, mahalanobis_sq
from embeddings import load_text_encoder, encode_prompts
from prompts import SAFE_PROMPTS, DEGENERATE_PROMPTS, BORDERLINE_PROMPTS


OUT_DIR = Path(__file__).parent / "outputs"
DEVICE = "mps"


def run():
    torch.manual_seed(42)
    np.random.seed(42)
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    # ── 1. Load text encoder ──
    print("\n[1/7] Loading CLIP text encoder...")
    tokenizer, text_encoder = load_text_encoder(device=DEVICE)

    # ── 2. Encode all prompts ──
    print("[2/7] Encoding prompts...")
    safe_emb = encode_prompts(SAFE_PROMPTS, tokenizer, text_encoder, device=DEVICE, pooling="eos")
    degen_emb = encode_prompts(DEGENERATE_PROMPTS, tokenizer, text_encoder, device=DEVICE, pooling="eos")
    border_emb = encode_prompts(BORDERLINE_PROMPTS, tokenizer, text_encoder, device=DEVICE, pooling="eos")
    print(f"  Safe: {safe_emb.shape}, Degenerate: {degen_emb.shape}, Borderline: {border_emb.shape}")

    # ── 3. Diagnostic: raw embedding geometry ──
    print("\n[3/7] Raw embedding geometry...")
    diagnose_geometry(safe_emb, degen_emb, border_emb)

    # ── 4. PCA cutoff + gamma sweep ──
    print("\n[4/7] Hyperparameter sweep (m, γ)...")
    sweep_results = sweep_hyperparams(safe_emb, degen_emb, border_emb)

    best_key = max(sweep_results, key=lambda k: sweep_results[k]["separation"])
    best_m, best_gamma_label = best_key
    print(f"\n  Best config: m={best_m}, γ={best_gamma_label} "
          f"(separation={sweep_results[best_key]['separation']:.3f})")

    # ── 5. Build final safe set ──
    n_safe = len(safe_emb)
    n_fit = int(0.6 * n_safe)
    n_cal = int(0.2 * n_safe)
    perm = torch.randperm(n_safe)
    fit_emb = safe_emb[perm[:n_fit]]
    cal_emb = safe_emb[perm[n_fit:n_fit + n_cal]]
    test_emb = safe_emb[perm[n_fit + n_cal:]]

    best_gamma = float(best_gamma_label) if best_gamma_label != "auto" else None
    print(f"\n[5/7] Building safe set (ε=0.05, m={best_m}, γ={best_gamma_label})...")
    print(f"  Split: {n_fit} fit, {n_cal} cal, {len(test_emb)} test")

    safe_set = build_safe_set(fit_emb, cal_emb, epsilon=0.05, m=best_m, gamma=best_gamma)
    print(f"  Calibrated α = {safe_set.alpha:.4f}")
    print(f"  γ (used) = {safe_set.gamma:.6f}")
    print(f"  Top eigenvalues: {safe_set.eig_reg[:5].tolist()}")
    print(f"  Floor eigenvalue: {safe_set.eig_reg[best_m:best_m+3].tolist()}")

    # ── 6. Evaluate ──
    print("\n[6/7] Evaluating...")
    evaluate_and_print(safe_emb, test_emb, degen_emb, border_emb, safe_set)

    # ── 7. Projection demo ──
    projection_demo(degen_emb, safe_set)

    # ── Plots ──
    print("\n[7/7] Generating plots...")
    all_safe_scores = ecf_score(safe_emb, safe_set)
    degen_scores = ecf_score(degen_emb, safe_set)
    border_scores = ecf_score(border_emb, safe_set)
    plot_score_distributions(all_safe_scores, degen_scores, border_scores, safe_set)
    plot_pca_projection(safe_emb, degen_emb, border_emb, safe_set)
    plot_conformal_coverage(safe_emb, best_m, best_gamma)
    plot_sweep(sweep_results)
    plot_eigenspectrum(safe_set)

    print(f"\nDone. Outputs in {OUT_DIR}/")


def diagnose_geometry(safe_emb, degen_emb, border_emb):
    """Check raw distances to understand the embedding space geometry."""
    safe_mean = safe_emb.mean(dim=0)

    safe_dist = (safe_emb - safe_mean).norm(dim=-1)
    degen_dist = (degen_emb - safe_mean).norm(dim=-1)
    border_dist = (border_emb - safe_mean).norm(dim=-1)

    print(f"  Euclidean distance from safe centroid:")
    print(f"    Safe:       mean={safe_dist.mean():.3f}, std={safe_dist.std():.3f}")
    print(f"    Degenerate: mean={degen_dist.mean():.3f}, std={degen_dist.std():.3f}")
    print(f"    Borderline: mean={border_dist.mean():.3f}, std={border_dist.std():.3f}")

    # Cosine similarity to safe centroid
    safe_cos = torch.nn.functional.cosine_similarity(safe_emb, safe_mean.unsqueeze(0))
    degen_cos = torch.nn.functional.cosine_similarity(degen_emb, safe_mean.unsqueeze(0))
    border_cos = torch.nn.functional.cosine_similarity(border_emb, safe_mean.unsqueeze(0))

    print(f"  Cosine similarity to safe centroid:")
    print(f"    Safe:       mean={safe_cos.mean():.4f}, min={safe_cos.min():.4f}")
    print(f"    Degenerate: mean={degen_cos.mean():.4f}, min={degen_cos.min():.4f}")
    print(f"    Borderline: mean={border_cos.mean():.4f}, min={border_cos.min():.4f}")

    # PCA spectrum
    centered = safe_emb - safe_mean
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    eigvals = S ** 2 / (len(safe_emb) - 1)
    cum_var = eigvals.cumsum(0) / eigvals.sum()
    for k in [2, 4, 8, 16, 32]:
        if k < len(cum_var):
            print(f"  PCA: top-{k} captures {cum_var[k-1]:.1%} of variance")


def sweep_hyperparams(safe_emb, degen_emb, border_emb, n_trials=30, epsilon=0.05):
    """Sweep over (m, γ) configurations."""
    configs = [
        (8, "auto"), (16, "auto"), (32, "auto"),
        (8, "0.001"), (8, "0.01"), (8, "0.1"),
        (16, "0.001"), (16, "0.01"), (16, "0.1"),
    ]
    results = {}
    n = len(safe_emb)

    for m, gamma_label in configs:
        gamma = float(gamma_label) if gamma_label != "auto" else None
        safe_coverages, degen_blocks, border_blocks = [], [], []

        for _ in range(n_trials):
            perm = torch.randperm(n)
            n_fit = int(0.6 * n)
            n_cal = int(0.2 * n)
            fit_idx = perm[:n_fit]
            cal_idx = perm[n_fit:n_fit + n_cal]
            test_idx = perm[n_fit + n_cal:]
            if len(test_idx) == 0:
                continue

            ss = build_safe_set(safe_emb[fit_idx], safe_emb[cal_idx],
                                epsilon=epsilon, m=m, gamma=gamma)
            safe_coverages.append(is_safe(safe_emb[test_idx], ss).float().mean().item())
            degen_blocks.append((~is_safe(degen_emb, ss)).float().mean().item())
            border_blocks.append((~is_safe(border_emb, ss)).float().mean().item())

        r = {
            "safe_coverage": np.mean(safe_coverages),
            "safe_coverage_std": np.std(safe_coverages),
            "degen_block": np.mean(degen_blocks),
            "degen_block_std": np.std(degen_blocks),
            "border_block": np.mean(border_blocks),
            "separation": np.mean(degen_blocks) - (1 - np.mean(safe_coverages)),
        }
        results[(m, gamma_label)] = r
        print(f"  m={m:>2}, γ={gamma_label:>5}: safe_cov={r['safe_coverage']:.3f}, "
              f"degen_block={r['degen_block']:.3f}, border_block={r['border_block']:.3f}, "
              f"sep={r['separation']:.3f}")

    return results


def evaluate_and_print(safe_emb, test_emb, degen_emb, border_emb, safe_set):
    test_scores = ecf_score(test_emb, safe_set)
    degen_scores = ecf_score(degen_emb, safe_set)
    border_scores = ecf_score(border_emb, safe_set)

    test_in = is_safe(test_emb, safe_set)
    degen_in = is_safe(degen_emb, safe_set)
    border_in = is_safe(border_emb, safe_set)

    print(f"\n  {'Category':<20} {'N':>4} {'Inside %':>10} {'Mean score':>12} {'Max score':>12}")
    print(f"  {'─'*58}")
    for name, scores, mask in [
        ("Safe (held-out)", test_scores, test_in),
        ("Degenerate", degen_scores, degen_in),
        ("Borderline", border_scores, border_in),
    ]:
        pct = mask.float().mean().item() * 100
        print(f"  {name:<20} {len(scores):>4} {pct:>9.1f}% "
              f"{scores.mean().item():>12.2f} {scores.max().item():>12.2f}")

    print(f"\n  α (threshold) = {safe_set.alpha:.2f}")

    print("\n  Degenerate prompt detail:")
    for p, s, inside in zip(DEGENERATE_PROMPTS, degen_scores, degen_in):
        tag = "SAFE" if inside else "BLOCKED"
        print(f"    [{tag:>7}] score={s.item():>10.2f}  {p[:60]}")

    print("\n  Borderline prompt detail:")
    for p, s, inside in zip(BORDERLINE_PROMPTS, border_scores, border_in):
        tag = "SAFE" if inside else "BLOCKED"
        print(f"    [{tag:>7}] score={s.item():>10.2f}  {p[:60]}")


def projection_demo(degen_emb, safe_set):
    print("\n  Projection demo...")
    degen_in = is_safe(degen_emb, safe_set)
    blocked_mask = ~degen_in

    if not blocked_mask.any():
        print("  No blocked embeddings to project (all degenerate passed)")
        return

    blocked_emb = degen_emb[blocked_mask]
    projected = project_onto_ellipsoid(blocked_emb, safe_set)

    proj_scores = ecf_score(projected, safe_set)
    proj_in = is_safe(projected, safe_set)
    displacement = (blocked_emb - projected).norm(dim=-1)

    orig_mah = mahalanobis_sq(blocked_emb, safe_set.mean, safe_set.V, safe_set.eig_inv).sqrt()
    proj_mah = mahalanobis_sq(projected, safe_set.mean, safe_set.V, safe_set.eig_inv).sqrt()

    print(f"  Projected {blocked_mask.sum().item()} blocked embeddings onto safe set")
    print(f"  All projected inside? {proj_in.all().item()}")
    print(f"  Mahalanobis dist: {orig_mah.mean().item():.2f} → {proj_mah.mean().item():.2f}")
    print(f"  Embedding displacement: mean={displacement.mean().item():.4f}, "
          f"max={displacement.max().item():.4f}")

    score_error = (proj_scores - safe_set.alpha).abs()
    print(f"  Projected score: mean={proj_scores.mean().item():.4f} "
          f"(target α={safe_set.alpha:.4f}, max |err|={score_error.max().item():.6f})")

    blocked_prompts = [p for p, b in zip(DEGENERATE_PROMPTS, blocked_mask) if b]
    print(f"\n  Example projections (first 8):")
    for i in range(min(8, len(blocked_prompts))):
        orig_s = ecf_score(blocked_emb[i:i+1], safe_set).item()
        print(f"    '{blocked_prompts[i][:55]}...'")
        print(f"      score: {orig_s:.2f} → {proj_scores[i].item():.2f}, "
              f"disp={displacement[i].item():.4f}")


# ── Plots ──

def plot_score_distributions(safe_s, degen_s, border_s, safe_set):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    all_scores = torch.cat([safe_s, degen_s, border_s])
    bins = np.linspace(0, all_scores.max().item() * 1.05, 80)
    ax.hist(safe_s.numpy(), bins=bins, alpha=0.6, label="Safe (nature)", color="tab:green", density=True)
    ax.hist(degen_s.numpy(), bins=bins, alpha=0.6, label="Degenerate", color="tab:red", density=True)
    ax.hist(border_s.numpy(), bins=bins, alpha=0.6, label="Borderline", color="tab:orange", density=True)
    ax.axvline(safe_set.alpha, color="black", linestyle="--", linewidth=2,
               label=f"α = {safe_set.alpha:.1f}")
    ax.set_xlabel("ECF score Λ₁(h)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution (full range)")
    ax.legend(fontsize=8)

    ax = axes[1]
    zoom_max = safe_set.alpha * 2
    bins_z = np.linspace(0, zoom_max, 60)
    ax.hist(safe_s.numpy(), bins=bins_z, alpha=0.6, label="Safe", color="tab:green", density=True)
    ax.hist(degen_s.numpy(), bins=bins_z, alpha=0.6, label="Degenerate", color="tab:red", density=True)
    ax.hist(border_s.numpy(), bins=bins_z, alpha=0.6, label="Borderline", color="tab:orange", density=True)
    ax.axvline(safe_set.alpha, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("ECF score Λ₁(h)")
    ax.set_title("Near threshold (zoomed)")

    fig.suptitle(f"Order-1 ECF — m={safe_set.m}, γ={safe_set.gamma:.4f}, ε={safe_set.epsilon}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "score_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved score_distributions.png")


def plot_pca_projection(safe_emb, degen_emb, border_emb, safe_set):
    """2D visualization using whitened Mahalanobis norm + discriminant direction.

    The top PCA eigenvectors capture within-safe variance, NOT the safe-vs-degenerate
    separation (which lives in the floored dimensions). So instead we plot:
    - x-axis: ECF score (the scalar that determines safe/unsafe)
    - y-axis: first PC of within-safe variation (for spread)
    """
    safe_scores = ecf_score(safe_emb, safe_set)
    degen_scores = ecf_score(degen_emb, safe_set)
    border_scores = ecf_score(border_emb, safe_set)

    # y-axis: first PC within safe distribution
    V1 = safe_set.V[:, 0]
    safe_y = ((safe_emb - safe_set.mean) @ V1).numpy()
    degen_y = ((degen_emb - safe_set.mean) @ V1).numpy()
    border_y = ((border_emb - safe_set.mean) @ V1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: score vs PC1 with threshold line
    ax = axes[0]
    ax.scatter(safe_scores.numpy(), safe_y, c="tab:green", s=20, alpha=0.5,
               label="Safe (nature)", zorder=2)
    ax.scatter(border_scores.numpy(), border_y, c="tab:orange", s=60, marker="D",
               alpha=0.8, label="Borderline", zorder=3)
    ax.scatter(degen_scores.numpy(), degen_y, c="tab:red", s=60, marker="x",
               linewidths=2, alpha=0.8, label="Degenerate", zorder=3)

    # Projection arrows
    blocked = ~is_safe(degen_emb, safe_set)
    if blocked.any():
        projected = project_onto_ellipsoid(degen_emb[blocked], safe_set)
        proj_scores = ecf_score(projected, safe_set)
        proj_y = ((projected - safe_set.mean) @ V1).numpy()
        blk_x = degen_scores[blocked].numpy()
        blk_y = degen_y[blocked.numpy()]
        for i in range(len(blk_x)):
            ax.annotate("", xy=(proj_scores[i].item(), proj_y[i]),
                         xytext=(blk_x[i], blk_y[i]),
                         arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1, alpha=0.4))

    ax.axvline(safe_set.alpha, color="black", linestyle="--", linewidth=2,
               label=f"α = {safe_set.alpha:.1f}")
    ax.set_xlabel("ECF score Λ₁(h)")
    ax.set_ylabel("PC 1 (within-safe variation)")
    ax.set_title("ECF Score vs. Principal Variation")
    ax.legend(fontsize=8, loc="upper left")

    # Right: whitened 2D view — use discriminant direction + orthogonal safe direction
    ax = axes[1]
    # Discriminant: direction from safe centroid to degenerate centroid in whitened space
    # V holds only the r = min(m, rank) retained eigenvectors (the tail is
    # isotropic at gamma and handled analytically), so slice eig_reg to match.
    sqrt_eig = safe_set.eig_reg[:safe_set.V.shape[1]].sqrt()
    safe_w = (safe_emb - safe_set.mean) @ safe_set.V / sqrt_eig
    degen_w = (degen_emb - safe_set.mean) @ safe_set.V / sqrt_eig
    border_w = (border_emb - safe_set.mean) @ safe_set.V / sqrt_eig

    disc = degen_w.mean(dim=0) - safe_w.mean(dim=0)
    disc = disc / disc.norm()

    # Orthogonal direction: largest variance direction in safe_w orthogonal to disc
    safe_w_orth = safe_w - (safe_w @ disc).unsqueeze(-1) * disc
    _, _, Vt_orth = torch.linalg.svd(safe_w_orth, full_matrices=False)
    orth = Vt_orth[0]

    safe_2d = torch.stack([safe_w @ disc, safe_w @ orth], dim=-1)
    degen_2d = torch.stack([degen_w @ disc, degen_w @ orth], dim=-1)
    border_2d = torch.stack([border_w @ disc, border_w @ orth], dim=-1)

    # Draw circle (safe set boundary in whitened space is a ball of radius r)
    r = np.sqrt(safe_set.alpha - 1.0)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(theta), r * np.sin(theta), "k--", lw=2,
            label=f"Safe boundary (r={r:.1f})")
    ax.fill(r * np.cos(theta), r * np.sin(theta), alpha=0.05, color="tab:green")

    ax.scatter(*safe_2d.T.numpy(), c="tab:green", s=20, alpha=0.5, label="Safe", zorder=2)
    ax.scatter(*border_2d.T.numpy(), c="tab:orange", s=60, marker="D", alpha=0.8,
               label="Borderline", zorder=3)
    ax.scatter(*degen_2d.T.numpy(), c="tab:red", s=60, marker="x", linewidths=2,
               alpha=0.8, label="Degenerate", zorder=3)

    # Projection arrows in whitened space
    if blocked.any():
        proj_w = (projected - safe_set.mean) @ safe_set.V / sqrt_eig
        proj_2d_w = torch.stack([proj_w @ disc, proj_w @ orth], dim=-1)
        blk_2d = degen_2d[blocked]
        for i in range(len(blk_2d)):
            ax.annotate("", xy=proj_2d_w[i].numpy(), xytext=blk_2d[i].numpy(),
                         arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1, alpha=0.4))

    ax.set_xlabel("Discriminant direction (whitened)")
    ax.set_ylabel("Within-safe direction (whitened)")
    ax.set_title("Whitened Space (safe set = ball)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    fig.suptitle(f"Embedding Space Geometry — m={safe_set.m}, γ={safe_set.gamma:.2f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "embedding_space.png", dpi=150)
    plt.close(fig)
    print(f"  Saved embedding_space.png")


def plot_conformal_coverage(safe_emb, m, gamma):
    epsilons = np.linspace(0.01, 0.30, 25)
    coverages = []
    n = len(safe_emb)

    for eps in epsilons:
        cover_runs = []
        for _ in range(50):
            perm = torch.randperm(n)
            n_fit = int(0.5 * n)
            n_cal = int(0.25 * n)
            fit_idx = perm[:n_fit]
            cal_idx = perm[n_fit:n_fit + n_cal]
            test_idx = perm[n_fit + n_cal:]
            if len(test_idx) == 0:
                continue

            ss = build_safe_set(safe_emb[fit_idx], safe_emb[cal_idx],
                                epsilon=eps, m=m, gamma=gamma)
            test_scores = ecf_score(safe_emb[test_idx], ss)
            cover_runs.append((test_scores <= ss.alpha).float().mean().item())

        coverages.append(np.mean(cover_runs) if cover_runs else np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epsilons, coverages, "o-", color="tab:blue", label="Empirical coverage", markersize=4)
    ax.plot(epsilons, 1 - epsilons, "k--", label="Target 1 − ε")
    ax.fill_between(epsilons, 1 - epsilons, 1.0, alpha=0.1, color="green", label="Valid region")
    ax.set_xlabel("ε (miscoverage rate)")
    ax.set_ylabel("Coverage P[h ∈ Ŝ]")
    ax.set_title(f"Conformal Coverage Verification (m={m})")
    ax.legend()
    ax.set_ylim(0.5, 1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "conformal_coverage.png", dpi=150)
    plt.close(fig)
    print(f"  Saved conformal_coverage.png")


def plot_sweep(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = sorted(results.keys())
    x_labels = [f"m={m}\nγ={g}" for m, g in keys]
    x = range(len(keys))

    safe_cov = [results[k]["safe_coverage"] for k in keys]
    degen_blk = [results[k]["degen_block"] for k in keys]
    border_blk = [results[k]["border_block"] for k in keys]

    width = 0.25
    ax.bar([i - width for i in x], safe_cov, width, label="Safe coverage", color="tab:green", alpha=0.7)
    ax.bar(list(x), degen_blk, width, label="Degen block rate", color="tab:red", alpha=0.7)
    ax.bar([i + width for i in x], border_blk, width, label="Border block rate", color="tab:orange", alpha=0.7)

    ax.axhline(0.95, color="black", linestyle=":", alpha=0.5, label="Target 1−ε")
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylabel("Rate")
    ax.set_title("Hyperparameter Sweep: (m, γ)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweep.png", dpi=150)
    plt.close(fig)
    print(f"  Saved sweep.png")


def plot_eigenspectrum(safe_set):
    fig, ax = plt.subplots(figsize=(8, 4))
    eigs = safe_set.eig_reg.numpy()
    ax.semilogy(eigs, "o-", markersize=3, color="tab:blue")
    ax.axhline(safe_set.gamma, color="red", linestyle="--", label=f"γ = {safe_set.gamma:.4f}")
    ax.axvline(safe_set.m - 0.5, color="gray", linestyle=":", label=f"m = {safe_set.m} (PCA cutoff)")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Regularized Covariance Eigenspectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eigenspectrum.png", dpi=150)
    plt.close(fig)
    print(f"  Saved eigenspectrum.png")


if __name__ == "__main__":
    run()
