"""E3u, step 1: the outputs of the historical unsafe prompts of ../results/e3/prompts_unsafe.json, as e3_pool.py does for the other
prompts, on the probe noises only: these prompts are scored (they define the unsafe sets) but never judged or deployed.

For every prompt: the pooled Gemma embedding, the source slice on the probes, the final latents on the P_PROBE probe noises and their
features (chi, the judge's P(winter), ViT-L/14 P(winter), snow and green fractions, scene similarity). One file per prompt in
../results/e3/pool_unsafe/, written atomically (a rerun resumes); the last run assembles ../results/e3/pool_unsafe.npz, whose rows
follow prompts_unsafe.json.     Usage: python e3u_pool.py
"""
import os, json, time
import numpy as np
import torch
import e3_common as C
from e3_pool import features, valid


def main():
    rows = json.load(open(os.path.join(C.RES, "prompts_unsafe.json")))["prompts"]
    out_dir = os.path.join(C.RES, "pool_unsafe"); os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    pipe = C.load_pipe()
    chi, judge = C.Clip("openai/clip-vit-large-patch14"), C.Clip("openai/clip-vit-base-patch32")
    xp = C.noises(C.PROBE_SEED, C.P_PROBE)
    print(f"[{time.time()-t0:.0f}s] models loaded", flush=True)
    done = 0
    for r in rows:
        f = os.path.join(out_dir, f"{r['id']:04d}.npz")
        if valid(f):
            continue
        subj = judge.text([r["subject_text"]])[0]
        conds, pooled, neg = C.encode(pipe, [r["prompt"]])
        rec = dict(pooled=pooled[0])
        rec["slice"] = C.source_slice(pipe, neg, conds[0], xp).astype(np.float16)
        lat = C.sample(pipe, neg, conds, [1.0], xp)
        rec["lat_probe"] = lat.numpy().astype(np.float16)
        rec.update(features(chi, judge, subj, C.decode(pipe, lat), "probe"))
        tmp = f[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **rec); os.replace(tmp, f)
        done += 1
        if done % 10 == 0:
            print(f"[{time.time()-t0:.0f}s] {done} prompts ({r['id']}, pwB probe {rec['pwB_probe'].mean():.2f})", flush=True)
        if done % 25 == 0:
            torch.mps.empty_cache()
    recs = [dict(np.load(os.path.join(out_dir, f"{r['id']:04d}.npz"))) for r in rows]
    out = {k: np.stack([rc[k] for rc in recs]) for k in recs[0]}
    out["ids"] = np.array([r["id"] for r in rows])
    np.savez_compressed(os.path.join(C.RES, "pool_unsafe.npz"), **out)
    print(f"[{time.time()-t0:.0f}s] assembled pool_unsafe.npz:", {k: v.shape for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
