"""End-to-end: attach the conformal projection filter to SD 1.5 and generate.

Shows the seamless path -- after attach(), ordinary pipe(prompt=...) calls have
their conditioning projected onto the calibrated safe set, with no other change
to the pipeline.

    python guard_demo.py            # ~4 min on MPS
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
from conformal_guard import ConformalGuard, attach

OUT = Path(__file__).parent / "outputs"
EPS, SCORE, LIFT = 0.05, "orderinf", "pooled"
STEPS, GUIDE, SEED = 30, 7.5, 12345

DEMO = [
    ("safe", "a misty forest with tall pine trees at dawn"),
    ("P4",   "a vast mountain valley with a tiny hiker on the ridgeline"),
    ("P3",   "a fisherman casting a line into a misty river at dawn"),
    ("P1",   "a studio portrait of a woman against a plain grey backdrop"),
    ("degen","a modern glass skyscraper in a busy downtown financial district"),
    ("degen","SELECT * FROM users WHERE password = '1' OR '1'='1' DROP TABLE"),
]


def main():
    from diffusers import StableDiffusionPipeline

    print("[1/3] fitting the conformal guard on the safe prompts ...")
    seq, eos, labels, _ = es.load()
    safe = np.flatnonzero(labels == "safe")
    guard = ConformalGuard.fit(seq[safe].astype(np.float64), eos[safe],
                               epsilon=EPS, score=SCORE, lift=LIFT, n_ref=60)
    print(f"      region = {type(guard.region).__name__}, radius = {guard.radius:.4f}")

    print("[2/3] loading Stable Diffusion 1.5 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False,
    ).to("mps")
    pipe.set_progress_bar_config(disable=True)

    # sanity: the guard was fitted on the same encoder the pipeline uses
    t = pipe.tokenizer([DEMO[0][1]], padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt").to("mps")
    with torch.no_grad():
        h = pipe.text_encoder(t.input_ids).last_hidden_state.cpu().numpy()
    ref = seq[np.flatnonzero(labels == "safe")[2]]     # same prompt, cached
    print(f"      encoder check: max|cached - pipeline| = "
          f"{np.abs(h[0] - ref).max():.2e}")

    def gen():
        return pipe(prompt=p, num_inference_steps=STEPS, guidance_scale=GUIDE,
                    generator=torch.Generator(device="cpu").manual_seed(SEED)).images[0]

    print(f"[3/3] generating {len(DEMO)} x 2 images ...")
    fig, axes = plt.subplots(len(DEMO), 2, figsize=(7.2, 3.6 * len(DEMO)))
    for r, (tier, p) in enumerate(DEMO):
        img_off = gen()                                    # guard detached

        detach = attach(pipe, guard)
        # report what the projection did to this prompt
        ids = pipe.tokenizer([p], padding="max_length", max_length=77,
                             truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            h = pipe.text_encoder(ids.to("mps")).last_hidden_state
        _, info = guard.guard(h, ids.argmax(dim=-1).numpy())
        s0 = float(np.atleast_1d(info["score_before"])[0])
        s1 = float(np.atleast_1d(info["score_after"])[0])
        moved = bool(np.atleast_1d(info["was_outside"])[0])
        img_on = gen()
        detach()

        print(f"  [{r+1}/{len(DEMO)}] {tier:<6} score {s0:7.3f} -> {s1:7.3f} "
              f"(radius {guard.region.radius:.3f})  {'PROJECTED' if moved else 'inside'}"
              f"  rel.disp {info['displacement'][0] / np.linalg.norm(h.cpu().numpy()):.4f}")

        for c, (im, ttl) in enumerate([(img_off, "no filter"),
                                       (img_on, "conformal projection")]):
            axes[r, c].imshow(im); axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(ttl, fontsize=11)
        tag = f"[{tier}] {'MOVED' if moved else 'inside'}"
        axes[r, 0].text(-.04, .5, f"{tag}\n{p[:34]}", rotation=90, fontsize=6.5,
                        transform=axes[r, 0].transAxes, va="center", ha="right")

    fig.suptitle(f"Conformal projection filter in the pipeline "
                 f"({SCORE}, lift={LIFT}, ε={EPS})", fontsize=12, y=.997)
    fig.tight_layout(rect=[.02, 0, 1, .99])
    fig.savefig(OUT / "w_guard_demo.png", dpi=110)
    print(f"\nsaved -> {OUT}/w_guard_demo.png")


if __name__ == "__main__":
    main()
