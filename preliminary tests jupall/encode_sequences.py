"""Encode every prompt tier to full (77, 768) CLIP sequences, the way SD does.

Cached to sequences.npz. Uses input_ids ONLY (no attention mask), matching
StableDiffusionPipeline.encode_prompt -- see wasserstein.Whitener notes.
"""
import os, sys
import numpy as np
import torch

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_sequences.npz")
DEVICE = "mps"


def build():
    from transformers import CLIPTokenizer, CLIPTextModel
    import data_cache as dc
    from prompts import (SAFE_PROMPTS, DEGENERATE_PROMPTS, BORDERLINE_PROMPTS,
                         SUBTLE_PROMPTS, PORTRAIT_TIERS)
    tiers = {"safe": SAFE_PROMPTS, "degenerate": DEGENERATE_PROMPTS,
             "borderline": BORDERLINE_PROMPTS, "subtle": SUBTLE_PROMPTS, **PORTRAIT_TIERS}
    prompts, labels = [], []
    for name in dc.TIERS:
        prompts += tiers[name]; labels += [name] * len(tiers[name])

    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()

    seqs, eos = [], []
    for i in range(0, len(prompts), 16):
        b = prompts[i:i + 16]
        t = tok(b, padding="max_length", max_length=77, truncation=True,
                return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            h = enc(t.input_ids).last_hidden_state          # NO attention mask
        seqs.append(h.cpu().numpy().astype(np.float32))
        eos.append(t.input_ids.argmax(dim=-1).cpu().numpy())
    S = np.concatenate(seqs); E = np.concatenate(eos)
    np.savez_compressed(CACHE, seq=S, eos=E, labels=np.array(labels),
                        prompts=np.array(prompts, dtype=object))
    return S, E, np.array(labels), np.array(prompts, dtype=object)


def load(force=False):
    if force or not os.path.exists(CACHE):
        return build()
    z = np.load(CACHE, allow_pickle=True)
    return z["seq"], z["eos"], z["labels"], z["prompts"]


if __name__ == "__main__":
    S, E, lab, pr = load(force="--force" in sys.argv)
    print("sequences", S.shape, S.dtype, " eos range", E.min(), E.max())
    for t in sorted(set(lab.tolist())):
        print(f"  {t:<24} {(lab == t).sum()}")
