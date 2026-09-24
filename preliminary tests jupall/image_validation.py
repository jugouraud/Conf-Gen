"""Were the filter's decisions right? Judge them by the IMAGES, not the words.

§17 reports that the GAP tier -- prompts mixing desert and polar vocabulary --
is accepted by every estimator (0.005-0.055 blocked) and calls this a failure.
The gallery suggests otherwise: "a crystalline playa under a sky of thin cirrus"
renders as an ordinary salt flat, "a sun-scoured frozen lake" as an ordinary
frozen lake. The prompts are lexically mixed but the MODEL resolves them onto
one of the two safe modes.

If that is systematic, the GAP "failure" is an artefact of judging a filter by
prompt vocabulary. The images are the ground truth: the filter's job is to stop
out-of-domain IMAGES, so a prompt whose image is in-domain should be accepted.

METHOD. Generate images for each tier, embed them with CLIP's IMAGE encoder,
and build a second conformal safe set in image space from the desert+polar
images. Then ask, for each tier, what fraction of its IMAGES falls outside that
set. Comparing image-space rejection with the prompt-space rejection of §17
says whether each decision was right.

    python image_validation.py            # ~10 min on MPS
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

from run_multimodal_experiment import load, features, M_REF, K_CAL, BETA
from wasserstein import conformal_radius
from run_residual_experiment import Basis

OUT = Path(__file__).parent / "outputs"
CACHE = Path(__file__).parent / "cache_images.npz"
N_PER = 16
STEPS, GUIDE = 20, 7.5
TIERS = ["mode_a_desert", "mode_b_polar", "gap", "other_alpine",
         "other_coast", "other_forest"]


def build():
    from diffusers import StableDiffusionPipeline
    from transformers import CLIPModel, CLIPImageProcessor

    theta, labels, prompts = load()
    rng = np.random.default_rng(0)
    chosen = {t: rng.choice(np.flatnonzero(labels == t), N_PER, replace=False)
              for t in TIERS}

    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False).to("mps")
    pipe.set_progress_bar_config(disable=True)

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("mps").eval()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    feats, tags, idxs = [], [], []
    total = sum(len(v) for v in chosen.values())
    done = 0
    for t, idx in chosen.items():
        for i in idx:
            img = pipe(prompt=str(prompts[i]), num_inference_steps=STEPS,
                       guidance_scale=GUIDE,
                       generator=torch.Generator(device="cpu").manual_seed(1234)).images[0]
            px = proc(images=img, return_tensors="pt").to("mps")
            with torch.no_grad():
                # NB transformers 5.x returns the vision tower's hidden states
                # from get_image_features, not the pooled embedding. Take the
                # pooled output explicitly.
                out = clip.vision_model(**px)
                f = clip.visual_projection(out.pooler_output)
            feats.append(f[0].cpu().numpy())
            tags.append(t)
            idxs.append(int(i))
            done += 1
            if done % 16 == 0:
                print(f"      {done}/{total}")
    F = np.array(feats, dtype=np.float64)
    np.savez_compressed(CACHE, feats=F, tags=np.array(tags), idxs=np.array(idxs))
    return F, np.array(tags), np.array(idxs)


def load_images():
    if not CACHE.exists():
        return build()
    z = np.load(CACHE, allow_pickle=True)
    F = z["feats"].astype(np.float64)
    if F.ndim == 4:
        # cache written before the API fix above: (n, 1, tokens, width) vision
        # hidden states. Token 0 is the CLS summary, which is the pooled image
        # representation prior to the projection -- adequate as a feature space.
        F = F[:, 0, 0, :]
    return F, z["tags"], z["idxs"]


def prompt_block_rates():
    """§17's prompt-space decision, per prompt (s_perp, whitened, beta=0.05)."""
    theta, labels, prompts = load()
    a = np.flatnonzero(labels == "mode_a_desert")
    b = np.flatnonzero(labels == "mode_b_polar")
    rate, seen = np.zeros(len(theta)), np.zeros(len(theta))
    for sp in range(40):
        r = np.random.default_rng(sp)
        pa, pb = r.permutation(len(a)), r.permutation(len(b))
        h = M_REF // 2
        ref = np.concatenate([a[pa[:h]], b[pb[:h]]])
        cal = np.concatenate([a[pa[h:h + K_CAL // 2]], b[pb[h:h + K_CAL // 2]]])
        W, A_, Bt_ = features(theta, ref, "full")
        s = np.sqrt(Bt_)
        e = conformal_radius(s[cal], BETA)
        held = np.setdiff1d(np.arange(len(theta)), np.concatenate([ref, cal]))
        rate[held] += (s[held] > e)
        seen[held] += 1
    return rate / np.maximum(seen, 1), labels, prompts


def main():
    print("[1/3] generating and embedding images ...")
    F, tags, idxs = load_images()
    print(f"      {len(F)} images, CLIP image features {F.shape}")

    # --- conformal safe set in IMAGE space, from the two safe modes ----------
    safe = np.flatnonzero((tags == "mode_a_desert") | (tags == "mode_b_polar"))
    rng = np.random.default_rng(0)
    n_img = {}
    reps = 200
    acc = {t: [] for t in TIERS}
    for _ in range(reps):
        p = rng.permutation(len(safe))
        ref, cal, tst = safe[p[:14]], safe[p[14:26]], safe[p[26:]]
        B_ = Basis(F[ref], 8)
        A_, Bt_, _, _ = B_.parts(F)
        s = np.sqrt(Bt_)
        e = conformal_radius(s[cal], 0.10)
        for t in TIERS:
            m = np.flatnonzero(tags == t)
            m = np.setdiff1d(m, np.concatenate([ref, cal])) if t in (
                "mode_a_desert", "mode_b_polar") else m
            if len(m):
                acc[t].append(np.mean(s[m] > e))

    p_rate, labels, prompts = prompt_block_rates()

    print(f"\n{'='*88}\nPROMPT-SPACE DECISION vs IMAGE-SPACE GROUND TRUTH\n{'='*88}")
    print(f"{'tier':<18}{'prompt blocked':>16}{'IMAGE out-of-domain':>22}{'verdict':>22}")
    print("-" * 88)
    for t in TIERS:
        pb = np.mean([p_rate[i] for i in idxs[tags == t]])
        ib = np.mean(acc[t])
        if t.startswith("mode_"):
            v = "correct (both low)" if pb < .3 and ib < .3 else "check"
        elif ib < 0.3:
            v = "ACCEPT WAS RIGHT" if pb < .5 else "over-blocked"
        else:
            v = "missed" if pb < .5 else "correct (both high)"
        print(f"{t:<18}{pb:>16.3f}{ib:>22.3f}{v:>22}")
    print("-" * 88)
    print("image out-of-domain = fraction of the tier's IMAGES outside a conformal")
    print("safe set built in CLIP image space from the desert+polar images (beta=0.10).")

    # ------------------------------------------------------------- figure --
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(TIERS))
    w = .38
    pb = [np.mean([p_rate[i] for i in idxs[tags == t]]) for t in TIERS]
    ib = [np.mean(acc[t]) for t in TIERS]
    ax.bar(x - w / 2, pb, w, label="prompt blocked (§17 filter)", color="tab:blue")
    ax.bar(x + w / 2, ib, w, label="IMAGE out-of-domain (ground truth)",
           color="tab:red")
    ax.set_xticks(x, [t.replace("other_", "").replace("mode_a_", "").replace("mode_b_", "")
                      for t in TIERS], fontsize=9)
    ax.set_ylabel("fraction"); ax.set_ylim(0, 1.05)
    ax.set_title("Was the filter right? Prompt-space decision vs image-space truth")
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "s_image_validation.png", dpi=150)
    print(f"\nfigure -> {OUT}/s_image_validation.png")


if __name__ == "__main__":
    main()
