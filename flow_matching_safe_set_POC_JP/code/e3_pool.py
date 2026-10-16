"""E3, step 1: the outputs of every prompt of ../results/e3/prompts.json under its own field, with their features.

For every prompt: the pooled Gemma embedding (prompt space), the source slice of its guided field on the probes (pull-back
distance), the final latents of the Euler sampler on the P_PROBE frozen probe noises (field distances) and, for the evaluation
pool, on the P_EVAL evaluation noises (judging, independent of every score). For every image: chi (CLIP ViT-L/14 embedding), the
judge's P(winter) (CLIP ViT-B/32), the ViT-L/14 zero-shot P(winter) (the set certified in chi-space), the snow and green pixel
fractions, and the judge's similarity to the prompt's scene alone (content).
One file per prompt in ../results/e3/pool/, written atomically; a rerun skips the prompts already done and resumes,
and the last run assembles ../results/e3/pool.npz.
Usage: python e3_pool.py [n_prompts]
"""
import os, sys, json, time
import numpy as np
import torch
import e3_common as C


def features(chi, judge, subj, imgs, tag):
    fl, fb = chi.image(imgs), judge.image(imgs)
    return {f"chi_{tag}": fl.astype(np.float32), f"pwB_{tag}": judge.p_winter(fb), f"pwL_{tag}": chi.p_winter(fl),
            f"snow_{tag}": C.snow_fraction(imgs), f"green_{tag}": C.green_fraction(imgs), f"subj_{tag}": fb @ subj}


def valid(f):
    """True if the per-prompt file exists and loads completely (a damaged file is removed and recomputed)."""
    if not os.path.exists(f):
        return False
    try:
        d = np.load(f); [d[k] for k in d.files]
        return "lat_probe" in d.files
    except Exception:
        os.remove(f); return False


def assemble(rows, out_dir):
    recs = [dict(np.load(os.path.join(out_dir, f"{r['id']:04d}.npz"))) for r in rows]
    out = {}
    for key in recs[0]:
        out[key] = np.stack([rc[key] for rc in recs])
    for key in [k for rc in recs for k in rc if k.endswith("_eval")]:
        if key in out:
            continue
        ref = next(rc[key] for rc in recs if key in rc)
        out[key] = np.stack([rc.get(key, np.full_like(ref, np.nan)) for rc in recs])
    out["ids"] = np.array([r["id"] for r in rows])
    np.savez_compressed(os.path.join(C.RES, "pool.npz"), **out)
    return out


def main():
    rows = json.load(open(os.path.join(C.RES, "prompts.json")))["prompts"]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(rows)
    out_dir = os.path.join(C.RES, "pool"); os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    pipe = C.load_pipe()
    chi, judge = C.Clip("openai/clip-vit-large-patch14"), C.Clip("openai/clip-vit-base-patch32")
    xp, xe = C.noises(C.PROBE_SEED, C.P_PROBE), C.noises(C.EVAL_SEED, C.P_EVAL)
    print(f"[{time.time()-t0:.0f}s] models loaded", flush=True)
    done = 0
    for r in rows[:limit]:
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
        if r["pool"] == "eval":
            lat = C.sample(pipe, neg, conds, [1.0], xe)
            rec["lat_eval"] = lat.numpy().astype(np.float16)
            rec.update(features(chi, judge, subj, C.decode(pipe, lat), "eval"))
        tmp = f[:-4] + ".tmp.npz"                                    # atomic write: an interruption never leaves a partial file
        np.savez_compressed(tmp, **rec); os.replace(tmp, f)
        done += 1
        if done % 10 == 0:
            print(f"[{time.time()-t0:.0f}s] {done} prompts ({r['id']}: {r['kind']}, pwB probe {rec['pwB_probe'].mean():.2f})", flush=True)
        if done % 25 == 0:
            torch.mps.empty_cache()
    if limit == len(rows):
        out = assemble(rows, out_dir)
        print(f"[{time.time()-t0:.0f}s] assembled pool.npz:", {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
