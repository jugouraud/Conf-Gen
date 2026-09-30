"""Projection vs interpolation, under an ADAPTIVE region, at small and large K.

Companion images for §20. The sweep shows that the metric projection is
insensitive to K (it lands ~25.6-26.8 from real data whatever K is) while
repair_t = 1 is very sensitive: at K = 2 it lands on the weight-nearest anchor
only 10% of the time, against 100% for the non-adaptive score.

This asks whether that matters in the output. For each blocked prompt:

    column 1   no filter
    columns 2-3  K = 2   (aggressively adaptive)  projection, then repair_t = 1
    columns 4-5  K = 300 (non-adaptive)           projection, then repair_t = 1

If the K = 2 repair lands somewhere semantically inappropriate, it shows here.

    python adaptive_gallery.py            # ~4 min
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

from run_multimodal_experiment import load, M_REF, K_CAL, BETA, bottleneck_n
from multimodal_projection import sequences
from run_adaptive_projection import task_distances, task_clouds
from wasserstein import whitening_map, conformal_radius
from conformal_regions import UnionOfBallsRegion

OUT = Path(__file__).parent / "outputs"
STEPS, GUIDE, SEED = 30, 7.5, 1234
K_SHOW = [2, 300]
DEMO = ["other_forest", "other_coast", "other_alpine"]


def main():
    from diffusers import StableDiffusionPipeline

    theta, labels, prompts = load()
    seq, eos = sequences(prompts)
    a_idx = np.flatnonzero(labels == "mode_a_desert")
    b_idx = np.flatnonzero(labels == "mode_b_polar")
    rng = np.random.default_rng(0)
    pa, pb = rng.permutation(len(a_idx)), rng.permutation(len(b_idx))
    h = M_REF // 2
    ref = np.concatenate([a_idx[pa[:h]], b_idx[pb[:h]]])
    cal = np.concatenate([a_idx[pa[h:h + K_CAL // 2]], b_idx[pb[h:h + K_CAL // 2]]])

    L, d = seq.shape[1], seq.shape[2]
    flat = seq.reshape(len(seq), L * d).astype(np.float32)
    wm = whitening_map(flat[ref].astype(np.float64), m=32)
    Fw = wm(flat.astype(np.float64)).astype(np.float32)
    Dw = cdist(Fw, Fw[ref])
    Daa = Dw[ref]
    tok_wm = whitening_map(np.vstack([seq[i, 1:max(int(eos[i]), 2)] for i in ref]), m=32)
    Dt = task_distances(task_clouds(seq, eos), ref, tok_wm)
    ord_t = np.argsort(Dt, axis=1)

    def radius(K):
        s = np.empty(len(cal))
        for j, i in enumerate(cal):
            sel = ord_t[i, :K]
            n = len(sel)
            G = np.zeros((n + 1, n + 1))
            G[:n, :n] = Daa[np.ix_(sel, sel)]
            G[n, :n] = Dw[i, sel]
            G[:n, n] = Dw[i, sel]
            s[j] = bottleneck_n(G, 0)
        return conformal_radius(s, BETA)

    eps = {K: radius(K) for K in K_SHOW}
    print("calibrated radius: " + ", ".join(f"K={K}: {e:.2f}" for K, e in eps.items()))

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
        raw = wm.inverse(np.asarray(fw_vec, dtype=np.float64)[None, :]).reshape(1, L, d)
        e = torch.as_tensor(raw, dtype=neg.dtype, device=neg.device)
        return pipe(prompt_embeds=e, negative_prompt_embeds=neg,
                    num_inference_steps=STEPS, guidance_scale=GUIDE,
                    generator=torch.Generator(device="cpu").manual_seed(SEED)).images[0]

    picks = [int(np.flatnonzero(labels == t)[0]) for t in DEMO]
    ncol = 1 + 2 * len(K_SHOW)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(3.4 * ncol, 4.0 * len(picks)))
    axes = np.atleast_2d(axes)
    print(f"generating {len(picks) * ncol} images ...")
    for r, i in enumerate(picks):
        p = str(prompts[i])
        axes[r][0].imshow(gen(Fw[i]))
        axes[r][0].set_title(f"NO FILTER\n{p[:36]}", fontsize=8, color="tab:red")
        c = 1
        for K in K_SHOW:
            sel = ord_t[i, :K]
            A = Fw[ref[sel]].astype(np.float64)
            reg = UnionOfBallsRegion(A, np.full(K, eps[K]), support=A)
            z = Fw[i:i + 1].astype(np.float64)
            for t in (0.0, 1.0):
                P = reg.repair(z, t=t)[0]
                land = ref[sel][int(np.argmin(np.linalg.norm(A - P[None, :], axis=1)))]
                mode = "desert" if land in a_idx else "polar"
                tag = "non-adaptive" if K == len(ref) else f"K={K}"
                axes[r][c].imshow(gen(P))
                axes[r][c].set_title(
                    f"{tag}  ·  {'projection' if t == 0 else 'repair_t=1'}\n"
                    f"→ {mode}: {str(prompts[land])[:30]}", fontsize=7.5,
                    color="tab:orange" if t == 0 else "tab:green")
                axes[r][c].axis("off")
                c += 1
        for k in range(ncol):
            axes[r][k].axis("off")
        print(f"  {DEMO[r]} done")

    fig.suptitle("Adaptive region: projection vs interpolation, at K = 2 and K = 300\n"
                 "the projection is insensitive to K; repair_t = 1 lands on whichever "
                 "anchor the TASK metric selected", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(OUT / "s_adaptive_gallery.png", dpi=110)
    print(f"\nsaved -> {OUT}/s_adaptive_gallery.png")


if __name__ == "__main__":
    main()
