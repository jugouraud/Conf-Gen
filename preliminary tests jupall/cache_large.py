"""Cache for the scaling study: 1200 safe prompts + the adversarial tiers.

Layout of the prompt array (contiguous blocks, in this order):

    [0    : 200)   GEOMETRY   fits the whiteners; never scored, never a reference
    [200  : 1200)  POOL       reference / calibration / test are drawn from here
    [1200 : 1349)  ADVERSARIAL  degenerate, borderline, subtle, P1..P4

The geometry block exists so that the METRIC is a fixed function of data that is
disjoint from every split's calibration and test sets. Without it, whitening the
token clouds per split would require recomputing 1.15M optimal-transport
problems for every split, which is not affordable; with it the metric is fixed
once and exchangeability of the calibration/test scores still holds.

Two ground metrics on token clouds are cached, because whether whitening helps
the TASK metric (as it demonstrably helps the weight-space metric) is one of the
questions the study answers:

    D_raw   W1 between raw content-token clouds
    D_wht   W1 between clouds whitened by the token-level pPCA covariance

Both have shape (1149, 1000): rows are POOL + ADVERSARIAL, columns are POOL.

    python cache_large.py --force        # ~4 min
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

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_large.npz")
DEVICE = "mps"
N_GEOM, N_POOL = 200, 1000
ADV_TIERS = ["safe_handwritten", "degenerate", "borderline", "subtle",
             "P1_studio", "P2_nature_bg", "P3_environmental", "P4_figure_in_landscape"]


def _build():
    import ot as pot
    from scipy.spatial.distance import cdist
    from transformers import CLIPTokenizer, CLIPTextModel
    from prompts_large import SAFE_LARGE
    from prompts import (SAFE_PROMPTS, DEGENERATE_PROMPTS, BORDERLINE_PROMPTS,
                         SUBTLE_PROMPTS, PORTRAIT_TIERS)
    from wasserstein import whitening_map

    # safe_handwritten is the ORIGINAL hand-written safe corpus. It is genuinely
    # in-domain, so a filter calibrated on the generated corpus should ACCEPT it.
    # Rejection rate here is the direct measure of how much the generated corpus
    # narrows the safe set relative to prompts a person would actually write.
    adv = {"safe_handwritten": SAFE_PROMPTS,
           "degenerate": DEGENERATE_PROMPTS, "borderline": BORDERLINE_PROMPTS,
           "subtle": SUBTLE_PROMPTS, **PORTRAIT_TIERS}
    prompts = list(SAFE_LARGE[:N_GEOM + N_POOL])
    labels = ["geometry"] * N_GEOM + ["pool"] * N_POOL
    for t in ADV_TIERS:
        prompts += adv[t]
        labels += [t] * len(adv[t])
    N = len(prompts)
    print(f"[1/4] {N} prompts: {N_GEOM} geometry, {N_POOL} pool, {N - N_GEOM - N_POOL} adversarial")

    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()

    print("[2/4] encoding ...")
    theta, clouds = [], []
    for i in range(0, N, 32):
        b = prompts[i:i + 32]
        t = tok(b, padding="max_length", max_length=77, truncation=True,
                return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            h = enc(t.input_ids).last_hidden_state          # no attention mask
        e = t.input_ids.argmax(dim=-1)
        theta.append(h[torch.arange(len(b), device=DEVICE), e].cpu().numpy())
        for j in range(len(b)):
            k = int(e[j])
            clouds.append(h[j, 1:k].cpu().numpy().astype(np.float64)
                          if k > 1 else h[j, k:k + 1].cpu().numpy().astype(np.float64))
    theta = np.concatenate(theta).astype(np.float64)

    # --- token-level whitener, fitted on the GEOMETRY block only ---
    print("[3/4] fitting the token whitener on the geometry block ...")
    geom_tokens = np.vstack(clouds[:N_GEOM])
    wm_tok = whitening_map(geom_tokens, m=32)
    clouds_w = [wm_tok(c) for c in clouds]

    # --- cloud-to-cloud W1, both ground metrics ---
    pool_slice = slice(N_GEOM, N_GEOM + N_POOL)
    rows = list(range(N_GEOM, N))                # POOL + ADVERSARIAL
    cols = list(range(N_GEOM, N_GEOM + N_POOL))  # POOL

    def w1_matrix(cl, tag):
        D = np.empty((len(rows), len(cols)))
        t0 = time.time()
        colc = [cl[j] for j in cols]
        for a, i in enumerate(rows):
            ci = cl[i]
            ai = np.ones(len(ci)) / len(ci)
            for b, cj in enumerate(colc):
                C = np.ascontiguousarray(cdist(ci, cj))
                D[a, b] = pot.emd2(ai, np.ones(len(cj)) / len(cj), C, numThreads=1)
            if (a + 1) % 200 == 0:
                el = time.time() - t0
                print(f"      {tag}: {a + 1}/{len(rows)}  ({el:.0f}s, "
                      f"eta {el / (a + 1) * (len(rows) - a - 1):.0f}s)")
        return D

    print("[4/4] cloud-to-cloud W1 matrices (2 x 1.15M transport problems) ...")
    D_raw = w1_matrix(clouds, "raw")
    D_wht = w1_matrix(clouds_w, "whitened")

    flat = np.concatenate(clouds).astype(np.float32)
    off = np.cumsum([0] + [len(c) for c in clouds])
    np.savez_compressed(CACHE, theta=theta.astype(np.float32),
                        labels=np.array(labels), prompts=np.array(prompts, dtype=object),
                        tok_flat=flat, tok_off=off, D_raw=D_raw, D_wht=D_wht,
                        n_geom=N_GEOM, n_pool=N_POOL)
    print(f"saved -> {CACHE}")
    return load(force=False)


def load(force=False):
    """Returns a dict with theta, labels, prompts, clouds, D_raw, D_wht, blocks."""
    if force or not os.path.exists(CACHE):
        return _build()
    z = np.load(CACHE, allow_pickle=True)
    off = z["tok_off"]
    flat = z["tok_flat"]
    clouds = [flat[off[i]:off[i + 1]].astype(np.float64) for i in range(len(off) - 1)]
    ng, npool = int(z["n_geom"]), int(z["n_pool"])
    return dict(theta=z["theta"].astype(np.float64), labels=z["labels"],
                prompts=z["prompts"], clouds=clouds,
                D_raw=z["D_raw"], D_wht=z["D_wht"],
                geom=np.arange(ng), pool=np.arange(ng, ng + npool),
                n_geom=ng, n_pool=npool)


if __name__ == "__main__":
    d = load(force="--force" in sys.argv)
    print(f"\ntheta {d['theta'].shape}   clouds {len(d['clouds'])}")
    print(f"D_raw {d['D_raw'].shape}   D_wht {d['D_wht'].shape}")
    for t in ["geometry", "pool"] + ADV_TIERS:
        print(f"  {t:<24} {(d['labels'] == t).sum()}")
    for nm, D in [("raw", d["D_raw"]), ("whitened", d["D_wht"])]:
        s = D[:d["n_pool"], :]
        off = s[~np.eye(len(s), s.shape[1], dtype=bool)[:len(s)]] if s.shape[0] == s.shape[1] else s
        print(f"  {nm:<9} within-pool W1: med {np.median(off):.3f}  "
              f"p5 {np.percentile(off, 5):.3f}  p95 {np.percentile(off, 95):.3f}  "
              f"spread p95/p5 {np.percentile(off, 95) / np.percentile(off, 5):.2f}")
