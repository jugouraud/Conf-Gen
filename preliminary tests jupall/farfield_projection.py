"""Projection of prompts that are not landscapes at all.

§18 projected other LANDSCAPE types onto the desert+polar safe set -- near
neighbours, and the certified interpolation moved through plausible intermediate
scenes. This asks what happens far away: portraits, city scenes, food, animals,
code. Two questions the near case could not answer:

  1. how much further outside the safe set are they, in radii?
  2. does `repair_t` still degrade gracefully, or does interpolating between
     "a studio portrait of a woman" and "a sand dune" pass through conditioning
     the model has never seen and break down?

At repair_t = 1 the answer is forced -- the output IS a real safe prompt's
conditioning, so the image must be a clean desert or polar scene however far the
input was. The intermediate values are the informative ones.

    python farfield_projection.py            # ~6 min on MPS
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

from run_multimodal_experiment import load, M_REF, K_CAL, BETA
from multimodal_projection import sequences
from wasserstein import whitening_map, conformal_radius
from conformal_regions import UnionOfBallsRegion

OUT = Path(__file__).parent / "outputs"
STEPS, GUIDE, SEED = 30, 7.5, 1234
T_VALUES = [0.0, 0.5, 1.0]

FAR = [
    ("portrait", "a studio portrait of a woman against a plain grey backdrop"),
    ("urban", "a crowded subway train during rush hour in Tokyo"),
    ("food", "a plate of spaghetti carbonara on a wooden table"),
    ("animal", "a tabby cat sleeping on a sunlit windowsill"),
    ("code", "SELECT * FROM users WHERE password = '1' OR '1'='1' DROP TABLE"),
]
# for scale: the nearest landscape negatives §18 used
NEAR = [("forest", "a misty old-growth forest, with light through the canopy"),
        ("coast", "a wave-battered sea stack")]


def main():
    from diffusers import StableDiffusionPipeline
    from transformers import CLIPTokenizer, CLIPTextModel

    theta, labels, prompts = load()
    seq, eos = sequences(prompts)
    a = np.flatnonzero(labels == "mode_a_desert")
    b = np.flatnonzero(labels == "mode_b_polar")

    rng = np.random.default_rng(0)
    pa, pb = rng.permutation(len(a)), rng.permutation(len(b))
    h = M_REF // 2
    ref = np.concatenate([a[pa[:h]], b[pb[:h]]])
    cal = np.concatenate([a[pa[h:h + K_CAL // 2]], b[pb[h:h + K_CAL // 2]]])

    L, d = seq.shape[1], seq.shape[2]
    flat = seq.reshape(len(seq), L * d)
    wm = whitening_map(flat[ref], m=32)
    Fw = wm(flat)
    eps = conformal_radius(cdist(Fw[cal], Fw[ref]).min(1), BETA)
    ball = UnionOfBallsRegion(Fw[ref], np.full(len(ref), eps), support=Fw[ref])

    # ---- encode the new prompts the same way -------------------------------
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("mps").eval()
    allp = FAR + NEAR
    t = tok([p for _, p in allp], padding="max_length", max_length=77,
            truncation=True, return_tensors="pt").to("mps")
    with torch.no_grad():
        H = enc(t.input_ids).last_hidden_state.cpu().numpy().astype(np.float64)
    Q = wm(H.reshape(len(allp), L * d))

    # ---- how far outside? ---------------------------------------------------
    print(f"conformal radius eps = {eps:.2f}   "
          f"(median distance between safe anchors "
          f"{np.median(cdist(Fw[ref], Fw[ref])[np.triu_indices(len(ref), 1)]):.2f})\n")
    print(f"{'prompt':<12}{'kind':<12}{'NN dist':>10}{'in radii':>10}"
          f"{'blocked':>9}{'move t=1':>10}")
    print("-" * 65)
    dn = cdist(Q, Fw[ref]).min(1)
    for i, (tag, p) in enumerate(allp):
        kind = "FAR" if i < len(FAR) else "landscape"
        mv = np.linalg.norm(ball.repair(Q[i:i + 1], t=1.0)[0] - Q[i])
        print(f"{tag:<12}{kind:<12}{dn[i]:>10.2f}{dn[i] / eps:>10.2f}"
              f"{str(dn[i] > eps):>9}{mv:>10.2f}")

    # ---- generate -----------------------------------------------------------
    print("\nloading Stable Diffusion 1.5 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False).to("mps")
    pipe.set_progress_bar_config(disable=True)
    neg_ids = pipe.tokenizer([""], padding="max_length", max_length=77,
                             return_tensors="pt").input_ids.to("mps")
    with torch.no_grad():
        neg = pipe.text_encoder(neg_ids).last_hidden_state

    def gen(fw_vec):
        raw = wm.inverse(fw_vec[None, :]).reshape(1, L, d)
        e = torch.as_tensor(raw, dtype=neg.dtype, device=neg.device)
        return pipe(prompt_embeds=e, negative_prompt_embeds=neg,
                    num_inference_steps=STEPS, guidance_scale=GUIDE,
                    generator=torch.Generator(device="cpu").manual_seed(SEED)).images[0]

    ncol = 1 + len(T_VALUES)
    fig, axes = plt.subplots(len(FAR), ncol, figsize=(3.6 * ncol, 4.15 * len(FAR)))
    axes = np.atleast_2d(axes)
    print(f"\ngenerating {len(FAR) * ncol} images ...")
    for r, (tag, p) in enumerate(FAR):
        axes[r][0].imshow(gen(Q[r]))
        axes[r][0].set_title(f"NO FILTER — {tag}  ({dn[r]/eps:.2f}× radius)\n{p[:42]}",
                             fontsize=8, color="tab:red")
        for c, tv in enumerate(T_VALUES, start=1):
            P = ball.repair(Q[r:r + 1], t=tv)[0]
            near = int(np.argmin(np.linalg.norm(Fw[ref] - P, axis=1)))
            mode = "desert" if ref[near] in a else "polar"
            axes[r][c].imshow(gen(P))
            lab = ("metric projection" if tv == 0 else
                   "lands on a real safe prompt" if tv == 1 else "interpolating")
            axes[r][c].set_title(f"repair_t = {tv:.1f}  → {mode}\n{lab}", fontsize=8,
                                 color="tab:orange" if tv == 0 else "tab:green")
        for c in range(ncol):
            axes[r][c].axis("off")
        print(f"  {tag} done")

    fig.suptitle("Prompts that are not landscapes at all, projected onto the "
                 "desert+polar safe set\nevery panel after column 1 is certified "
                 "inside the safe set", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(OUT / "s_farfield_projection.png", dpi=110)
    print(f"\nsaved -> {OUT}/s_farfield_projection.png")


if __name__ == "__main__":
    main()
