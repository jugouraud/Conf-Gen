"""Extract features from where the conditioning is actually consumed.

README §15.5 direction 1(b). Every score so far has been computed on the CLIP
text embedding. But the text embedding is not what determines the image -- it is
an input to 16 cross-attention blocks inside the U-Net, and it is there that the
text and image representations interact and that composition materialises.

This runs the U-Net for ONE denoising step on a FIXED latent, so the only thing
that varies across prompts is the conditioning, and captures the output of a
cross-attention block. Mean-pooling over spatial positions gives one vector per
prompt, in exactly the form every estimator in this project already accepts.

Two hook points are captured:

    mid    mid_block.attentions.0...attn2      8x8 spatial, 1280 channels
           the deepest, most semantic level
    down2  down_blocks.2.attentions.1...attn2  16x16 spatial, 1280 channels
           one level up, more spatially resolved

Cost: one U-Net forward per prompt (~1/50 of a generation), batched. The whole
corpus takes ~2 minutes, and at deployment it would be ~20 ms per prompt --
still 400x cheaper than generating the image.

    python unet_features.py --force
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import numpy as np
import torch

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_unet.npz")
DEVICE = "mps"
TIMESTEP = 500          # mid-trajectory: structure is formed, detail is not
BATCH = 8
HOOKS = {"mid": "mid_block.attentions.0.transformer_blocks.0.attn2",
         "down2": "down_blocks.2.attentions.1.transformer_blocks.0.attn2"}


def _build():
    import warnings
    warnings.filterwarnings("ignore")
    from diffusers import StableDiffusionPipeline
    import cache_large as cl

    d = cl.load()
    prompts = list(d["prompts"])
    labels = d["labels"]
    print(f"[1/3] {len(prompts)} prompts")

    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    unet, tok, te = pipe.unet, pipe.tokenizer, pipe.text_encoder

    grabbed = {}

    def make_hook(name):
        def hook(_m, _inp, out):
            grabbed[name] = out.detach()
        return hook

    mods = dict(unet.named_modules())
    handles = [mods[path].register_forward_hook(make_hook(k))
               for k, path in HOOKS.items()]

    # one fixed latent for every prompt: the only variation is the conditioning
    g = torch.Generator(device="cpu").manual_seed(0)
    z0 = torch.randn(1, 4, 64, 64, generator=g).to(DEVICE)
    t = torch.tensor([TIMESTEP], device=DEVICE)

    feats = {k: [] for k in HOOKS}   # plus "mid2x2", added below
    print(f"[2/3] U-Net forward at t={TIMESTEP}, fixed latent, batch {BATCH} ...")
    t0 = time.time()
    for i in range(0, len(prompts), BATCH):
        b = prompts[i:i + BATCH]
        ids = tok(b, padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            c = te(ids).last_hidden_state                      # no attention mask
            unet(z0.expand(len(b), -1, -1, -1), t.expand(len(b)),
                 encoder_hidden_states=c)
        for k in HOOKS:
            g_ = grabbed[k]                                   # (B, n_spatial, d)
            feats[k].append(g_.mean(dim=1).cpu().numpy().astype(np.float32))
            if k == "mid":
                # 2x2 spatial pooling: keeps coarse LAYOUT, which mean-pooling
                # destroys. "flowing upward" is a spatial property, so a readout
                # blind to layout cannot represent it.
                B_, n, dd = g_.shape
                side = int(round(n ** 0.5))
                q = g_.reshape(B_, side, side, dd)
                h = side // 2
                cells = [q[:, :h, :h], q[:, :h, h:], q[:, h:, :h], q[:, h:, h:]]
                grid = torch.cat([c.mean(dim=(1, 2)) for c in cells], dim=-1)
                feats.setdefault("mid2x2", []).append(
                    grid.cpu().numpy().astype(np.float32))
        if (i // BATCH) % 30 == 0 and i:
            el = time.time() - t0
            print(f"      {i}/{len(prompts)}  ({el:.0f}s, "
                  f"eta {el / i * (len(prompts) - i):.0f}s)")
    for h in handles:
        h.remove()

    out = {k: np.concatenate(v) for k, v in feats.items()}
    print(f"[3/3] shapes: " + ", ".join(f"{k} {v.shape}" for k, v in out.items()))
    np.savez_compressed(CACHE, labels=labels, prompts=np.array(prompts, dtype=object),
                        n_geom=d["n_geom"], n_pool=d["n_pool"],
                        **{f"f_{k}": v for k, v in out.items()})
    print(f"saved -> {CACHE}  ({os.path.getsize(CACHE) / 1e6:.0f} MB)")
    return load(force=False)


def load(force=False):
    if force or not os.path.exists(CACHE):
        return _build()
    z = np.load(CACHE, allow_pickle=True)
    keys = [k[2:] for k in z.files if k.startswith("f_")]
    return dict(labels=z["labels"], prompts=z["prompts"],
                n_geom=int(z["n_geom"]), n_pool=int(z["n_pool"]),
                **{k: z[f"f_{k}"].astype(np.float64) for k in keys})


if __name__ == "__main__":
    d = load(force="--force" in sys.argv)
    for k in [x for x in d if x not in ("labels", "prompts", "n_geom", "n_pool")]:
        print(f"{k:<8} {d[k].shape}  norm {np.linalg.norm(d[k], axis=1).mean():.2f}")
