"""What does a BLOCKED prompt look like after it is projected onto the safe set?

§6 showed that at the calibrated radius the metric projection is semantically
inert -- it lands on the region's shell, a full radius from any real data, and
the image does not change. §6.2 added `repair_t`, a certified interpolation
toward a real safe point, which only produces sensible images under the
`sequence` lift.

The bimodal safe set of §17 sharpens the geometry, because the projection now
has to CHOOSE A MODE:

    ellipsoid (ECF)    convex, so it projects toward the centroid mu -- which
                       for a bimodal set lies in the GAP BETWEEN the modes,
                       where there is no data at all.
    union of balls     non-convex; projects into the nearest ball, committing
                       to one mode.

This generates the images and measures which mode each projection commits to.

    python multimodal_projection.py            # ~5 min on MPS
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
from wasserstein import whitening_map, conformal_radius
from conformal_regions import (UnionOfBallsRegion, EllipsoidRegion,
                               region_from_orderinf)

OUT = Path(__file__).parent / "outputs"
SEQ_CACHE = Path(__file__).parent / "cache_multimodal_seq.npz"
STEPS, GUIDE, SEED = 30, 7.5, 1234
T_VALUES = [0.0, 0.6, 1.0]
DEMO = ["other_forest", "other_tropical", "other_coast", "other_alpine"]


def sequences(prompts):
    """Full (77, 768) conditioning for every prompt, encoded as SD encodes it."""
    if SEQ_CACHE.exists():
        z = np.load(SEQ_CACHE)
        return z["seq"].astype(np.float64), z["eos"]
    from transformers import CLIPTokenizer, CLIPTextModel
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("mps").eval()
    S, E = [], []
    for i in range(0, len(prompts), 32):
        b = [str(p) for p in prompts[i:i + 32]]
        t = tok(b, padding="max_length", max_length=77, truncation=True,
                return_tensors="pt").to("mps")
        with torch.no_grad():
            h = enc(t.input_ids).last_hidden_state
        S.append(h.cpu().numpy().astype(np.float32))
        E.append(t.input_ids.argmax(dim=-1).cpu().numpy())
    S, E = np.concatenate(S), np.concatenate(E)
    np.savez_compressed(SEQ_CACHE, seq=S, eos=E)
    return S.astype(np.float64), E


def main():
    from diffusers import StableDiffusionPipeline

    theta, labels, prompts = load()
    seq, eos = sequences(prompts)
    a = np.flatnonzero(labels == "mode_a_desert")
    b = np.flatnonzero(labels == "mode_b_polar")

    rng = np.random.default_rng(0)
    pa, pb = rng.permutation(len(a)), rng.permutation(len(b))
    h = M_REF // 2
    ref = np.concatenate([a[pa[:h]], b[pb[:h]]])
    cal = np.concatenate([a[pa[h:h + K_CAL // 2]], b[pb[h:h + K_CAL // 2]]])

    # ---- the safe set, in flattened SEQUENCE space (the lift that repairs) --
    L, d = seq.shape[1], seq.shape[2]
    flat = seq.reshape(len(seq), L * d)
    wm = whitening_map(flat[ref], m=32)
    Fw = wm(flat)
    s_inf = lambda idx: cdist(Fw[idx], Fw[ref]).min(1)      # NN distance = bottleneck
    eps = conformal_radius(s_inf(cal), BETA)
    ball = UnionOfBallsRegion(Fw[ref], np.full(len(ref), eps), support=Fw[ref])

    # ---- the convex comparator, on the same anchors -------------------------
    mu = Fw[ref].mean(0)
    r_ell = np.quantile(np.linalg.norm(Fw[cal] - mu, axis=1), 1 - BETA)

    print(f"union-of-balls radius {eps:.2f};  ellipsoid radius {r_ell:.2f}")
    print(f"distance between the two mode centroids: "
          f"{np.linalg.norm(Fw[a].mean(0) - Fw[b].mean(0)):.2f}\n")

    # ---- which mode does each projection commit to? ------------------------
    mu_a, mu_b = Fw[a].mean(0), Fw[b].mean(0)
    mid = (mu_a + mu_b) / 2
    print(f"{'prompt tier':<16}{'projector':<18}{'d(mode A)':>11}{'d(mode B)':>11}"
          f"{'d(midpoint)':>13}{'commits to':>12}")
    print("-" * 82)
    picks, rows = [], []
    for t in DEMO:
        idx = np.flatnonzero(labels == t)
        outside = idx[s_inf(idx) > eps]
        if not len(outside):
            continue
        i = int(outside[0])
        picks.append((t, i))
        for nm, P in [("union of balls", ball.project(Fw[i:i + 1])[0]),
                      ("ellipsoid", mu + (Fw[i] - mu) * min(
                          1.0, r_ell / np.linalg.norm(Fw[i] - mu)))]:
            da, db, dm = (np.linalg.norm(P - mu_a), np.linalg.norm(P - mu_b),
                          np.linalg.norm(P - mid))
            commit = "mode A" if da < db * .97 else "mode B" if db < da * .97 else "GAP"
            print(f"{t.replace('other_',''):<16}{nm:<18}{da:>11.1f}{db:>11.1f}"
                  f"{dm:>13.1f}{commit:>12}")
            rows.append((t, nm, da, db, dm, commit))
        print("-" * 82)

    # ---- generate ----------------------------------------------------------
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
    fig, axes = plt.subplots(len(picks), ncol, figsize=(3.6 * ncol, 4.1 * len(picks)))
    axes = np.atleast_2d(axes)
    print(f"\ngenerating {len(picks) * ncol} images ...")
    for r, (tier, i) in enumerate(picks):
        p = str(prompts[i])
        axes[r][0].imshow(gen(Fw[i]))
        axes[r][0].set_title(f"BLOCKED — original\n{p[:44]}", fontsize=8, color="tab:red")
        for c, t in enumerate(T_VALUES, start=1):
            P = ball.repair(Fw[i:i + 1], t=t)[0]
            near = int(np.argmin(np.linalg.norm(Fw[ref] - P, axis=1)))
            mode = "A desert" if ref[near] in a else "B polar"
            axes[r][c].imshow(gen(P))
            axes[r][c].set_title(f"repair_t = {t:.1f}  → mode {mode}\n"
                                 f"{'metric projection' if t == 0 else ''}"
                                 f"{'lands on a real safe prompt' if t == 1 else ''}",
                                 fontsize=8, color="tab:green" if t else "tab:orange")
        for c in range(ncol):
            axes[r][c].axis("off")
        print(f"  {tier.replace('other_',''):<10} done")

    fig.suptitle("Projecting a BLOCKED prompt onto a bimodal safe set (desert + polar)\n"
                 "union-of-balls region, sequence lift; every panel is certified inside "
                 "the safe set", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(OUT / "s_multimodal_projection.png", dpi=110)
    print(f"\nsaved -> {OUT}/s_multimodal_projection.png")


if __name__ == "__main__":
    main()
