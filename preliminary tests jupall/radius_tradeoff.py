"""How tight must the safe set be before projection actually changes the image?

The conformal radius at eps=0.05 is 13.98 whitened units, while the safe anchors
sit only ~11 apart: the calibrated region is larger than the data manifold, so
the projection stops on a shell that contains no data and the image is
unchanged. This sweeps the ball radius r from the calibrated value down to 0 and
reports, for each r:

    coverage   the fraction of held-out SAFE prompts still inside  (r is a
               conformal radius for SOME level; this is the level it buys)
    move       how far a degenerate prompt is displaced
    d(proj,NN) how far the projected point ends up from a real safe prompt

At r = 0 the region degenerates to the anchor set itself and the projection maps
onto the nearest safe prompt exactly -- guaranteed semantics, zero coverage.
Everything in between is the trade-off.

    python radius_tradeoff.py            # ~4 min on MPS (generates images)
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import encode_sequences as es
from conformal_guard import ConformalGuard, pooled_from
from conformal_regions import UnionOfBallsRegion

OUT = Path(__file__).parent / "outputs"
STEPS, GUIDE, SEED = 30, 7.5, 12345
PROMPT = "a modern glass skyscraper in a busy downtown financial district"
GEN_RADII_FRAC = [1.0, 0.65, 0.40, 0.20, 0.0]


def main():
    seq, eos, labels, prompts = es.load()
    seq = seq.astype(np.float64)
    safe = np.flatnonzero(labels == "safe")
    rng = np.random.default_rng(0)
    p = rng.permutation(len(safe))
    fit_i, tst_i = safe[p[:-20]], safe[p[-20:]]

    guard = ConformalGuard.fit(seq[fit_i], eos[fit_i], epsilon=0.05,
                               score="orderinf", lift="pooled", n_ref=60)
    A = guard.region.centers
    eps = float(guard.region.radii[0])

    W = lambda idx: guard.whiten(pooled_from(seq[idx], eos[idx], "eos"))
    Z_tst = W(tst_i)
    Z_deg = W(np.flatnonzero(labels == "degenerate"))

    print(f"calibrated eps = {eps:.3f}; median anchor spacing = "
          f"{np.median(np.linalg.norm(A[:, None] - A[None], axis=2)[np.triu_indices(len(A), 1)]):.3f}")
    print(f"\n{'r':>8}{'r/eps':>7}{'safe cov':>10}{'degen move':>12}{'d(proj,NN)':>12}")
    print("-" * 49)
    curve = []
    for frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        r = eps * frac
        reg = UnionOfBallsRegion(A, np.full(len(A), max(r, 1e-9)))
        cov = float(reg.contains(Z_tst).mean())
        P = reg.project(Z_deg)
        move = float(np.linalg.norm(P - Z_deg, axis=1).mean())
        dnn = float(np.linalg.norm(P[:, None] - A[None], axis=2).min(1).mean())
        curve.append((r, frac, cov, move, dnn))
        print(f"{r:>8.2f}{frac:>7.2f}{cov:>10.2f}{move:>12.2f}{dnn:>12.2f}")

    # ---------- generate at a few radii ----------
    from diffusers import StableDiffusionPipeline
    print("\nloading SD 1.5 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False,
    ).to("mps")
    pipe.set_progress_bar_config(disable=True)

    ids = pipe.tokenizer([PROMPT], padding="max_length", max_length=77,
                         truncation=True, return_tensors="pt")
    with torch.no_grad():
        H = pipe.text_encoder(ids.input_ids.to("mps")).last_hidden_state
        neg = pipe.text_encoder(pipe.tokenizer([""], padding="max_length",
              max_length=77, return_tensors="pt").input_ids.to("mps")).last_hidden_state
    e = int(ids.input_ids[0].argmax())
    H_np = H.cpu().numpy().astype(np.float64)
    pooled = pooled_from(H_np, [e], "eos")
    zw = guard.whiten(pooled)

    fig, axes = plt.subplots(1, len(GEN_RADII_FRAC), figsize=(3.1 * len(GEN_RADII_FRAC), 3.7))
    for c, frac in enumerate(GEN_RADII_FRAC):
        r = eps * frac
        reg = UnionOfBallsRegion(A, np.full(len(A), max(r, 1e-9)))
        proj = guard.unwhiten(reg.project(zw))
        delta = proj - pooled
        seq_new = torch.as_tensor(H_np + delta[:, None, :], dtype=H.dtype, device=H.device)
        img = pipe(prompt_embeds=seq_new, negative_prompt_embeds=neg,
                   num_inference_steps=STEPS, guidance_scale=GUIDE,
                   generator=torch.Generator(device="cpu").manual_seed(SEED)).images[0]
        cov = [c_ for c_ in curve if abs(c_[1] - frac) < 1e-9]
        cov = cov[0][2] if cov else float("nan")
        axes[c].imshow(img); axes[c].axis("off")
        axes[c].set_title(f"r = {r:.1f}  ({frac:.0%} of ε)\nsafe coverage {cov:.2f}",
                          fontsize=9)
        print(f"  generated at r = {r:6.2f}  (coverage {cov:.2f})")

    fig.suptitle(f'"{PROMPT[:58]}"\nprojected onto the union of balls at shrinking radius',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .90])
    fig.savefig(OUT / "w_radius_tradeoff_images.png", dpi=120); plt.close(fig)

    # ---------- trade-off curve ----------
    r_, fr, cov, mv, dnn = map(np.array, zip(*curve))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(fr, cov, "o-", color="tab:blue")
    ax[0].axhline(.95, ls=":", c="k", lw=1)
    ax[0].set_xlabel("ball radius as a fraction of the calibrated ε")
    ax[0].set_ylabel("coverage of held-out safe prompts")
    ax[0].set_title("what you give up"); ax[0].grid(alpha=.3)
    ax[1].plot(fr, mv, "o-", color="tab:red", label="displacement of a degenerate prompt")
    ax[1].plot(fr, dnn, "s-", color="tab:green", label="distance from a real safe prompt")
    ax[1].set_xlabel("ball radius as a fraction of the calibrated ε")
    ax[1].set_ylabel("whitened units")
    ax[1].set_title("what you get"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.suptitle("Conformal projection: coverage vs semantic reach", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "w_radius_tradeoff.png", dpi=150)
    print(f"\nfigures -> {OUT}/w_radius_tradeoff*.png")


if __name__ == "__main__":
    main()
