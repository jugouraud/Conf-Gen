"""Generate the images behind §17's numbers.

The tables say the filter blocks other landscape types (0.94-1.00), lets
`alpine` through a quarter of the time, and accepts almost every GAP prompt.
What they cannot say is whether those decisions are *right*.

The GAP tier is the case that matters. Those prompts mix desert and polar
vocabulary, and the filter accepts them. Two possibilities the numbers cannot
distinguish:

  * they render as coherent scenes that are neither desert nor polar -- the
    filter is genuinely failing;
  * they collapse onto one of the two safe modes, in which case accepting them
    is defensible and the "failure" is an artefact of judging prompts by their
    words rather than their images.

This picks examples by measured block rate, generates them, and labels each
with the decision and whether that decision is correct.

    python multimodal_gallery.py            # ~4 min on MPS
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
from scipy.spatial.distance import cdist

from run_multimodal_experiment import load, features, M_REF, K_CAL, BETA, OTHERS
from wasserstein import conformal_radius

OUT = Path(__file__).parent / "outputs"
N_SPLITS = 40
STEPS, GUIDE, SEED = 30, 7.5, 1234


def block_rates():
    """Per-prompt fraction of splits in which the prompt is blocked."""
    theta, labels, prompts = load()
    a = np.flatnonzero(labels == "mode_a_desert")
    b = np.flatnonzero(labels == "mode_b_polar")
    rate = np.zeros(len(theta))
    seen = np.zeros(len(theta))
    for sp in range(N_SPLITS):
        r = np.random.default_rng(sp)
        pa, pb = r.permutation(len(a)), r.permutation(len(b))
        h = M_REF // 2
        ref = np.concatenate([a[pa[:h]], b[pb[:h]]])
        cal = np.concatenate([a[pa[h:h + K_CAL // 2]], b[pb[h:h + K_CAL // 2]]])
        W, A_, Bt_ = features(theta, ref, "full")
        s = np.sqrt(Bt_)                       # s_perp: the deployed screening score
        e = conformal_radius(s[cal], BETA)
        held = np.setdiff1d(np.arange(len(theta)), np.concatenate([ref, cal]))
        rate[held] += (s[held] > e)
        seen[held] += 1
    return rate / np.maximum(seen, 1), labels, prompts


def pick(rate, labels, prompts):
    """Choose examples spanning correct-accept, correct-block, and failures."""
    def best(mask, target, n=1):
        idx = np.flatnonzero(mask)
        return idx[np.argsort(np.abs(rate[idx] - target))][:n]

    sel = []
    for lab, tgt, tag in [("mode_a_desert", 0.0, "SAFE desert"),
                          ("mode_b_polar", 0.0, "SAFE polar")]:
        for i in best(labels == lab, tgt):
            sel.append((i, tag, "accept"))
    for k, tgt in [("forest", 1.0), ("tropical", 1.0)]:
        for i in best(labels == f"other_{k}", tgt):
            sel.append((i, f"other: {k}", "block"))
    # the subtle one: alpine, blocked ~0.73 -- show both outcomes
    for i in best(labels == "other_alpine", 1.0):
        sel.append((i, "other: alpine", "block"))
    for i in best(labels == "other_alpine", 0.0):
        sel.append((i, "other: alpine", "block"))
    for i in best(labels == "other_coast", 0.0):
        sel.append((i, "other: coast", "block"))
    # the payload: GAP prompts the filter accepts
    for i in best(labels == "gap", 0.0, n=3):
        sel.append((i, "GAP (desert x polar)", "block"))
    return sel


def main():
    from diffusers import StableDiffusionPipeline

    print("[1/3] computing per-prompt block rates ...")
    rate, labels, prompts = block_rates()
    sel = pick(rate, labels, prompts)

    print("[2/3] loading Stable Diffusion 1.5 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False).to("mps")
    pipe.set_progress_bar_config(disable=True)

    print(f"[3/3] generating {len(sel)} images ...")
    cols = 3
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 4.7 * rows))
    axes = np.atleast_2d(axes)
    for k, (i, tag, should) in enumerate(sel):
        p = str(prompts[i])
        blocked = rate[i] >= 0.5
        correct = (blocked and should == "block") or (not blocked and should == "accept")
        img = pipe(prompt=p, num_inference_steps=STEPS, guidance_scale=GUIDE,
                   generator=torch.Generator(device="cpu").manual_seed(SEED)).images[0]
        ax = axes[k // cols][k % cols]
        ax.imshow(img)
        ax.axis("off")
        verdict = "BLOCKED" if blocked else "ACCEPTED"
        mark = "correct" if correct else ("FALSE REJECT" if should == "accept"
                                          else "MISSED")
        colour = "tab:green" if correct else "tab:red"
        wrapped = p if len(p) <= 46 else p[:44] + "…"
        ax.set_title(f"[{tag}]  {verdict} ({rate[i]:.2f})  — {mark}\n{wrapped}",
                     fontsize=8, color=colour)
        print(f"  {tag:<22} rate {rate[i]:.2f}  {verdict:<9} {mark:<13} {p[:44]}")
    for k in range(len(sel), rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.suptitle("Bimodal safe set (desert + polar): what the filter blocks, and what it misses\n"
                 "green = correct decision, red = failure;  (x.xx) = fraction of "
                 "splits in which the prompt is blocked", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(OUT / "s_multimodal_gallery.png", dpi=110)
    print(f"\nsaved -> {OUT}/s_multimodal_gallery.png")


if __name__ == "__main__":
    main()
