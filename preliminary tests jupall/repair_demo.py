"""Certified repair: sweep repair_t under two lifts and generate.

The plain metric projection (t=0) lands on the region's boundary, about one
anchor-spacing beyond the data, and leaves the image unchanged. repair_t
interpolates from there toward the nearest REAL safe point; every value of t is
still certified inside the conformal region (ConformalRegion.repair).

The two lifts behave completely differently under that interpolation:

  pooled    only the pooled statistic is forced, by a rigid translation of all
            77 rows. Large t means a large uniform shift, which takes the
            sequence off-manifold -> texture noise.
  sequence  the whole (77, 768) tensor is interpolated toward a real safe
            prompt's own conditioning, which is exactly prompt interpolation
            -> at t = 1 the output IS a real safe prompt.

    python repair_demo.py            # ~4 min on MPS
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
from conformal_guard import ConformalGuard

OUT = Path(__file__).parent / "outputs"
STEPS, GUIDE, SEED = 30, 7.5, 12345
T_VALUES = [0.0, 0.35, 0.70, 1.0]
PROMPT = "a modern glass skyscraper in a busy downtown financial district"


def main():
    from diffusers import StableDiffusionPipeline

    seq, eos, labels, prompts = es.load()
    seq = seq.astype(np.float64)
    safe = np.flatnonzero(labels == "safe")

    print("[1/3] fitting guards (pooled and sequence lifts) ...")
    guards = {lift: ConformalGuard.fit(seq[safe], eos[safe], epsilon=0.05,
                                       score="orderinf", lift=lift, n_ref=60)
              for lift in ("pooled", "sequence")}
    for lift, g in guards.items():
        print(f"      {lift:<9} radius {g.radius:.3f}, region "
              f"{type(g.region).__name__} of {len(g.region.centers)} balls")

    print("[2/3] loading Stable Diffusion 1.5 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False,
    ).to("mps")
    pipe.set_progress_bar_config(disable=True)

    ids = pipe.tokenizer([PROMPT], padding="max_length", max_length=77,
                         truncation=True, return_tensors="pt")
    e = int(ids.input_ids[0].argmax())
    with torch.no_grad():
        H = pipe.text_encoder(ids.input_ids.to("mps")).last_hidden_state
        neg = pipe.text_encoder(pipe.tokenizer([""], padding="max_length",
              max_length=77, return_tensors="pt").input_ids.to("mps")).last_hidden_state
    H_np = H.cpu().numpy().astype(np.float64)

    print(f"[3/3] generating {len(guards)} x {len(T_VALUES)} images ...")
    fig, axes = plt.subplots(len(guards), len(T_VALUES),
                             figsize=(3.1 * len(T_VALUES), 3.5 * len(guards)))
    for r, (lift, g) in enumerate(guards.items()):
        for c, t in enumerate(T_VALUES):
            g.repair_t = t
            new, info = g.guard(H_np, [e])
            ok = bool(np.atleast_1d(info["score_after"])[0] <= g.region.radius + 1e-6)
            rel = info["displacement"][0] / np.linalg.norm(H_np)
            img = pipe(prompt_embeds=torch.as_tensor(new, dtype=H.dtype, device=H.device),
                       negative_prompt_embeds=neg, num_inference_steps=STEPS,
                       guidance_scale=GUIDE,
                       generator=torch.Generator(device="cpu").manual_seed(SEED)).images[0]
            axes[r, c].imshow(img); axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"repair_t = {t:.2f}", fontsize=11)
            axes[r, c].set_xlabel("")
            print(f"  {lift:<9} t={t:.2f}  certified={ok}  rel.disp={rel:.3f}")
        axes[r, 0].text(-.04, .5, f"lift = {lift}", rotation=90, fontsize=9,
                        transform=axes[r, 0].transAxes, va="center", ha="right")

    fig.suptitle(f'Certified repair (every panel is inside the conformal safe set)\n'
                 f'"{PROMPT[:56]}"', fontsize=11)
    fig.tight_layout(rect=[.02, 0, 1, .93])
    fig.savefig(OUT / "w_repair_demo.png", dpi=115)
    print(f"\nsaved -> {OUT}/w_repair_demo.png")


if __name__ == "__main__":
    main()
