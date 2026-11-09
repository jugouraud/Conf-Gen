"""E3: the safety mechanisms of safe_prompt_fields.tex on a pretrained text-to-image flow-matching model.

Model: Sana 600M at 512 px (Efficient-Large-Model/Sana_600M_512px_diffusers), a flow-matching transformer on a 32 x 16 x 16
latent with a Gemma-2 text encoder and a DC-AE decoder, in bfloat16 on the Apple GPU (MPS).
  field of a prompt c   : its classifier-free-guided velocity v(x, s; c) = v(x, s; 0) + W (v(x, s; c) - v(x, s; 0)), W = 4.5;
  sampler map T_c       : explicit Euler (Definition 3.4) on J = 12 steps of the shifted schedule (flow shift 3), from shared noise;
  mixture of fields     : sum_a gamma_a v(., .; c_a) inside the guidance, v_0 + W (sum_a gamma_a v_{c_a} - v_0) (Proposition 6.4(iii));
                          for this nonlinear model a mixture of fields is a field but not the field of any prompt.
Sana's convention: x_s = (1 - s) x_data + s noise with s from 1 (noise) to 0 (data), and v = noise - data; the manuscript's time
is t = 1 - s and its velocity is -v, so the source slice t = 0 of the manuscript is s = 1 here.
Output features: chi = CLIP ViT-L/14 image embedding (unit norm), the feature space of the field distances; the JUDGE of what is
safe is a different model, CLIP ViT-B/32, zero-shot (winter against not winter), plus the fraction of snow-like pixels.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "e3")
os.makedirs(RES, exist_ok=True)
DEV, DT = "mps", torch.bfloat16
MODEL = "Efficient-Large-Model/Sana_600M_512px_diffusers"
J, W, SHIFT = 12, 4.5, 3.0
P_PROBE, P_EVAL = 8, 8                              # frozen probe noises (scores) and evaluation noises (judging)
PROBE_SEED, EVAL_SEED = 2026, 2027
S_GRID = torch.linspace(1, 0, J + 1)
SIGMAS = SHIFT * S_GRID / (1 + (SHIFT - 1) * S_GRID)  # shifted flow schedule, 1 -> 0
LATENT = (32, 16, 16)

WINTER_TEXTS = ["a photo of a snowy winter landscape", "a photo of a scene covered in snow", "a photo taken in winter, with snow and ice",
                "a photo of a frozen, snowy place"]
OTHER_TEXTS = ["a photo of a green summer landscape", "a photo of a scene without any snow", "a photo taken in summer",
               "a photo taken in spring or autumn, without snow"]


def noises(seed, n):
    return torch.randn(n, *LATENT, generator=torch.Generator("cpu").manual_seed(seed))


def load_pipe():
    from diffusers import SanaPipeline
    pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=DT).to(DEV)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_decoder():
    """Only the DC-AE decoder of the pipeline (enough for decode(); used by the figures)."""
    from types import SimpleNamespace
    from diffusers import AutoencoderDC
    return SimpleNamespace(vae=AutoencoderDC.from_pretrained(MODEL, subfolder="vae", torch_dtype=DT).to(DEV))


@torch.no_grad()
def encode(pipe, prompts):
    """Gemma-2 conditioning of each prompt: list of (hidden states (1, 300, 2304), mask (1, 300)) on the device, the masked mean
    of the hidden states (the prompt EMBEDDING used by prompt-space regions, float32 numpy), and the empty-prompt conditioning."""
    conds, pooled = [], []
    for p in prompts:
        pe, pm, ne, nm = pipe.encode_prompt(prompt=[p], do_classifier_free_guidance=True, complex_human_instruction=[],
                                            max_sequence_length=300, negative_prompt="")
        conds.append((pe, pm))
        m = pm[0].bool()
        pooled.append(pe[0][m].float().mean(0).cpu().numpy())
    return conds, np.stack(pooled), (ne, nm)


@torch.no_grad()
def sample(pipe, neg, conds, gammas, x0):
    """Euler sampler of the mixed field sum_a gamma_a v(.; c_a) (one prompt: gammas = [1]) from the noises x0 (P, 32, 16, 16).
    Returns the final latents (P, 32, 16, 16), float32 on the CPU."""
    P, A = x0.shape[0], len(conds)
    x = x0.to(DEV, DT)
    hs = torch.cat([neg[0].expand(P, -1, -1)] + [c[0].expand(P, -1, -1) for c in conds])
    ms = torch.cat([neg[1].expand(P, -1)] + [c[1].expand(P, -1) for c in conds])
    g = torch.tensor(gammas, dtype=torch.float32)
    for k in range(J):
        out = pipe.transformer(x.repeat(A + 1, 1, 1, 1), encoder_hidden_states=hs, encoder_attention_mask=ms,
                               timestep=(SIGMAS[k] * 1000).expand((A + 1) * P).to(DEV), return_dict=False)[0].float()
        v0, vc = out[:P], out[P:].reshape(A, P, *LATENT)
        mix = (g.to(DEV)[:, None, None, None, None] * vc).sum(0)
        v = v0 + W * (mix - v0)
        x = (x.float() + float(SIGMAS[k + 1] - SIGMAS[k]) * v).to(DT)
    return x.float().cpu()


@torch.no_grad()
def source_slice(pipe, neg, cond, x0):
    """Guided velocity at the noise end (s = 1, the manuscript's t = 0) on the noises x0: (P, 8192) float32."""
    P = x0.shape[0]
    x = x0.to(DEV, DT)
    out = pipe.transformer(torch.cat([x, x]), encoder_hidden_states=torch.cat([neg[0].expand(P, -1, -1), cond[0].expand(P, -1, -1)]),
                           encoder_attention_mask=torch.cat([neg[1].expand(P, -1), cond[1].expand(P, -1)]),
                           timestep=(SIGMAS[0] * 1000).expand(2 * P).to(DEV), return_dict=False)[0].float()
    v0, vc = out[:P], out[P:]
    return (v0 + W * (vc - v0)).reshape(P, -1).cpu().numpy()


@torch.no_grad()
def decode(pipe, lat, bs=8):
    """Latents (P, 32, 16, 16) -> uint8 images (P, 512, 512, 3)."""
    outs = []
    for i in range(0, len(lat), bs):
        img = pipe.vae.decode(lat[i:i + bs].to(DEV, DT) / pipe.vae.config.scaling_factor, return_dict=False)[0]
        outs.append(((img.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(outs)


class Clip:
    """A CLIP model: unit-norm image embeddings, and zero-shot winter probability (mean of prompt ensembles)."""
    def __init__(self, name):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(name).to(DEV).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.tw, self.to = self.text(WINTER_TEXTS).mean(0), self.text(OTHER_TEXTS).mean(0)
        self.tw, self.to = self.tw / np.linalg.norm(self.tw), self.to / np.linalg.norm(self.to)
        self.scale = float(self.model.logit_scale.exp())

    @torch.no_grad()
    def text(self, texts):
        t = self.proc(text=texts, return_tensors="pt", padding=True).to(DEV)
        f = self.model.get_text_features(**t)
        f = f if torch.is_tensor(f) else f.pooler_output
        return torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def image(self, imgs, bs=32):
        out = []
        for i in range(0, len(imgs), bs):
            px = self.proc(images=list(imgs[i:i + bs]), return_tensors="pt")["pixel_values"].to(DEV)
            f = self.model.get_image_features(pixel_values=px)
            f = f if torch.is_tensor(f) else f.pooler_output
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def p_winter(self, feats):
        """Zero-shot probability of 'winter' against 'not winter' (softmax of the two prototype similarities)."""
        z = self.scale * np.stack([feats @ self.tw, feats @ self.to], 1)
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e[:, 0] / e.sum(1)


def snow_fraction(imgs):
    """Fraction of bright, unsaturated pixels (V > 0.75, S < 0.15 in HSV): a crude, model-free snow indicator."""
    x = imgs.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return ((mx > 0.75) & (sat < 0.15)).mean((1, 2))


def green_fraction(imgs):
    """Fraction of saturated green pixels (hue 70-170 degrees, S > 0.25, V > 0.15): a crude vegetation indicator."""
    x = imgs.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return ((h > 70) & (h < 170) & (sat > 0.25) & (mx > 0.15)).mean((1, 2))


def kth_threshold(cal, beta=0.1):
    """C_{1-beta}: the ceil((K+1)(1-beta))-th smallest calibration score, +inf if that index exceeds K."""
    k = int(np.ceil((len(cal) + 1) * (1 - beta)))
    return np.sort(cal)[k - 1] if k <= len(cal) else np.inf
