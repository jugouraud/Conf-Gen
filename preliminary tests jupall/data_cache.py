"""Build (and cache) the CLIP embeddings and the task-space distance matrix.

Everything downstream needs the same three objects:

    theta   (N, 768)  pooled [EOS] embeddings for every prompt in every tier
    labels  (N,)      tier name per row: safe / degenerate / borderline / subtle
    Dtask   (N, 120)  order-1 Wasserstein distance from each prompt's content-token
                      cloud to each SAFE prompt's cloud -- the task-space metric
                      d_T that the adaptive scores reweight with

Dtask is the only expensive part (N x 120 exact EMD solves), so it is cached to
an .npz and reused. Run `python data_cache.py --force` to rebuild.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_embeddings.npz")
DEVICE = "mps"
TIERS = ["safe", "degenerate", "borderline", "subtle",
         "P1_studio", "P2_nature_bg", "P3_environmental", "P4_figure_in_landscape"]


def _build():
    import ot as pot
    from scipy.spatial.distance import cdist
    from embeddings import load_text_encoder, encode_prompts_with_tasks
    from prompts import (SAFE_PROMPTS, DEGENERATE_PROMPTS,
                         BORDERLINE_PROMPTS, SUBTLE_PROMPTS, PORTRAIT_TIERS)

    tiers = {
        "safe": SAFE_PROMPTS,
        "degenerate": DEGENERATE_PROMPTS,
        "borderline": BORDERLINE_PROMPTS,
        "subtle": SUBTLE_PROMPTS,
        **PORTRAIT_TIERS,
    }
    all_prompts, labels = [], []
    for name in TIERS:
        all_prompts += tiers[name]
        labels += [name] * len(tiers[name])

    print(f"[1/3] loading CLIP text encoder on {DEVICE} ...")
    tok, enc = load_text_encoder(device=DEVICE)

    print(f"[2/3] encoding {len(all_prompts)} prompts (theta + task clouds) ...")
    theta, clouds = encode_prompts_with_tasks(all_prompts, tok, enc, device=DEVICE)
    theta_np = theta.numpy().astype("float64")

    n_safe = len(tiers["safe"])
    safe_clouds = clouds[:n_safe]

    print(f"[3/3] task-space W1 matrix: {len(clouds)} x {n_safe} EMD solves ...")
    D = np.empty((len(clouds), n_safe))
    for i, ci in enumerate(clouds):
        for j, cj in enumerate(safe_clouds):
            C = np.ascontiguousarray(cdist(ci, cj, metric="euclidean"))
            a = np.ones(len(ci)) / len(ci)
            b = np.ones(len(cj)) / len(cj)
            D[i, j] = pot.emd2(a, b, C, numThreads=1)
        if (i + 1) % 40 == 0:
            print(f"      {i + 1}/{len(clouds)}")

    np.savez_compressed(CACHE, theta=theta_np, labels=np.array(labels),
                        Dtask=D, prompts=np.array(all_prompts, dtype=object),
                        allow_pickle=True)
    print(f"saved -> {CACHE}")
    return theta_np, np.array(labels), D, np.array(all_prompts, dtype=object)


def load(force=False):
    if force or not os.path.exists(CACHE):
        return _build()
    z = np.load(CACHE, allow_pickle=True)
    return z["theta"], z["labels"], z["Dtask"], z["prompts"]


def masks(labels):
    """Boolean index per tier."""
    return {t: (labels == t) for t in TIERS}


if __name__ == "__main__":
    theta, labels, D, prompts = load(force="--force" in sys.argv)
    print(f"\ntheta  {theta.shape}")
    print(f"Dtask  {D.shape}   min={D.min():.4f} med={np.median(D):.4f} max={D.max():.4f}")
    for t in TIERS:
        print(f"  {t:<11} {(labels == t).sum():>4}")
