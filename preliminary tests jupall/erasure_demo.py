"""Generate before/after images for the semantically selective projection.

Passes the erased (77, 768) sequence straight to SD 1.5's U-Net, which is the
tensor it actually cross-attends to -- unlike the pooled-vector delta that
generate_comparison.py broadcasts across all tokens.

    python erasure_demo.py            # ~4 min on MPS
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

from pairs import PAIRS
from embeddings import load_text_encoder
from concept_erasure import ConceptEraser

OUT = Path(__file__).parent / "outputs"
DEVICE = "mps"
STEPS, GUIDE, SEED = 30, 7.5, 12345
RANK, STRENGTH = 2, 1.0

DEMO = [
    ("P2", "a portrait of a woman with a blurred forest behind her"),
    ("P2", "a close-up of a hiker's face with mountains out of focus"),
    ("P3", "a fisherman casting a line into a misty river at dawn"),
    ("P3", "a woman in a red coat walking along a rocky shoreline"),
    ("P4", "a vast mountain valley with a tiny hiker on the ridgeline"),
    ("safe", "a misty forest with tall pine trees at dawn"),
]


def main():
    from diffusers import StableDiffusionPipeline

    print("[1/3] fitting the human-subject subspace from matched pairs ...")
    tok, enc = load_text_encoder(device=DEVICE)
    er = ConceptEraser.fit(PAIRS, tok, enc, device=DEVICE, rank=RANK,
                           strength=STRENGTH, token_level=True)

    print("[2/3] loading Stable Diffusion 1.5 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    def encode(prompt):
        """Encode the way SD's own encode_prompt does: input_ids ONLY, no mask."""
        t = pipe.tokenizer([prompt], padding="max_length", max_length=77,
                           truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            h = pipe.text_encoder(t.input_ids).last_hidden_state
        return h, int(t.input_ids[0].argmax())

    neg, _ = None, None

    def gen(seq):
        g = torch.Generator(device="cpu").manual_seed(SEED)
        with torch.no_grad():
            return pipe(prompt_embeds=seq, negative_prompt_embeds=neg,
                        num_inference_steps=STEPS, guidance_scale=GUIDE,
                        generator=g).images[0]

    neg = encode("")[0]
    print(f"[3/3] generating {len(DEMO)} x 2 images ...")
    fig, axes = plt.subplots(len(DEMO), 2, figsize=(7.2, 3.6 * len(DEMO)))
    for r, (tier, prompt) in enumerate(DEMO):
        seq, eos = encode(prompt)
        seq_er = er.erase_sequence(seq, eos_idx=[eos])
        shift = (seq_er - seq).norm().item() / seq.norm().item()
        print(f"  [{r+1}/{len(DEMO)}] {tier}: {prompt[:52]}  (rel. shift {shift:.3f})")

        for c, (img, ttl) in enumerate([(gen(seq), "original"),
                                        (gen(seq_er), "person-subspace erased")]):
            ax = axes[r, c]
            ax.imshow(img); ax.axis("off")
            if r == 0:
                ax.set_title(ttl, fontsize=11)
        axes[r, 0].text(-0.04, 0.5, f"[{tier}]  {prompt[:38]}", rotation=90,
                        transform=axes[r, 0].transAxes, va="center", ha="right", fontsize=7)

    fig.suptitle("Selective projection: strip the human subject, keep the landscape",
                 fontsize=12, y=.997)
    fig.tight_layout(rect=[0.02, 0, 1, .99])
    fig.savefig(OUT / "w_erasure_demo.png", dpi=110)
    print(f"\nsaved -> {OUT}/w_erasure_demo.png")


if __name__ == "__main__":
    main()
