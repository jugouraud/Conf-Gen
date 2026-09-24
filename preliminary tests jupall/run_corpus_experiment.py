"""Can a SYNTHETIC corpus be made representative? An audited attempt.

README §15.5 direction 2. §11 showed the 1,200-prompt template grid was
unrepresentative: it rejected 58% of hand-written prompts while reporting 0.95
coverage. §12 turned representativeness into a testable property. This uses the
audit as a DESIGN SIGNAL and asks whether a better generator can pass it.

PROTOCOL (this is the point of the script -- using a holdout to iterate on and
then reporting on the same holdout would be selection on the test set):

    the 120 hand-written prompts are split ONCE, at the top, into
        dev-audit    60   used to iterate on corpus design
        final-audit  60   touched exactly once, at the end

Four corpus variants of increasing effort are audited against dev-audit; the
best is then tested once against final-audit.

    V0  template grid            the §11 corpus, known to fail (the control)
    V1  compositional            variable clause count, optional elements,
                                 four registers, lengths 3-25 words
    V2  length-matched           V1, rejection-sampled to match the hand-written
                                 token-length distribution
    V3  vocabulary-grounded      built by recombining the hand-written DEV half's
                                 own vocabulary (the most favourable case a
                                 generator can be given)

A positive control -- hand-written dev vs hand-written final, the same process
on both sides -- checks that the protocol can pass at all.

    python run_corpus_experiment.py            # ~3 min
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from prompts import SAFE_PROMPTS
import prompts_large as PL
from wasserstein import whitening_map, conformal_radius
from corpus_audit import audit_scores

OUT = Path(__file__).parent / "outputs"
BETA = 0.05
N_CORPUS = 900
DEVICE = "mps"


# ------------------------------------------------------------ generators ---

def gen_compositional(n, seed=0, target_lengths=None):
    """Variable clause count, optional elements, four registers.

    A template grid's intrinsic dimension is bounded by its slot count. This
    varies the STRUCTURE as well as the slots: how many clauses, which kinds,
    whether modifiers appear at all, and the register.
    """
    rng = random.Random(seed)
    L, D = PL.LANDFORM, PL.DESCRIPTOR
    T, W, S, V, E = PL.TIME, PL.WEATHER, PL.SEASON, PL.VIEW, PL.ELEMENT

    def core():
        if rng.random() < .55:
            return f"a {rng.choice(D)} {rng.choice(L)}"
        return f"a {rng.choice(L)}"

    def clause():
        return rng.choice([lambda: rng.choice(T), lambda: rng.choice(W),
                           lambda: rng.choice(S), lambda: rng.choice(V),
                           lambda: rng.choice(E)])()

    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < 400 * n:
        guard += 1
        k = rng.choices([0, 1, 2, 3], weights=[.12, .40, .33, .15])[0]
        parts = [core()] + [clause() for _ in range(k)]
        reg = rng.random()
        if reg < .18:
            body = " ".join(parts)
            p = f"{body}, photographed on film"
        elif reg < .34:
            p = ", ".join(parts)
        elif reg < .50:
            p = f"a photograph of {' '.join(parts)[2:]}"
        else:
            p = " ".join(parts)
        p = re.sub(r"\s+", " ", p).strip().rstrip(",")
        if target_lengths is not None and len(p.split()) not in target_lengths:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def gen_vocab_grounded(n, source, seed=0):
    """Recombine the SOURCE corpus's own vocabulary.

    The most favourable case a generator can be given: every content word comes
    from the very distribution it is being audited against. If even this fails,
    the failure is structural rather than lexical.
    """
    rng = random.Random(seed)
    heads, tails = [], []
    for s in source:
        w = s.split()
        if len(w) < 5:
            continue
        cut = rng.randint(2, len(w) - 2)
        heads.append(" ".join(w[:cut]))
        tails.append(" ".join(w[cut:]))
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < 400 * n:
        guard += 1
        p = f"{rng.choice(heads)} {rng.choice(tails)}"
        p = re.sub(r"\s+", " ", p).strip()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ------------------------------------------------------------- machinery ---

def encode(prompts):
    from transformers import CLIPTokenizer, CLIPTextModel
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
    out = []
    for i in range(0, len(prompts), 64):
        b = prompts[i:i + 64]
        t = tok(b, padding="max_length", max_length=77, truncation=True,
                return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            h = enc(t.input_ids).last_hidden_state
        e = t.input_ids.argmax(dim=-1)
        out.append(h[torch.arange(len(b), device=DEVICE), e].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def d_eff(W):
    D = np.linalg.norm(W[:, None] - W[None], axis=2)
    np.fill_diagonal(D, np.inf)
    s = np.sort(D, 1)
    mu = s[:, 1] / s[:, 0]
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return len(mu) / np.log(mu).sum()


def audit_corpus(theta_corpus, theta_hold, beta=BETA, seed=0):
    """Fit the metric and anchors on the corpus, calibrate, audit the holdout."""
    rng = np.random.default_rng(seed)
    n = len(theta_corpus)
    p = rng.permutation(n)
    fit, cal = p[:n // 2], p[n // 2:]
    wm = whitening_map(theta_corpus[fit], m=32)
    V = wm.V
    def s(X):
        w = wm(X)
        c = w @ V
        return np.sqrt(np.maximum((w ** 2).sum(1) - (c ** 2).sum(1), 0.0))
    return audit_scores(s(theta_corpus[cal]), s(theta_hold), beta=beta, n_perm=5000)


def main():
    rng = random.Random(0)
    hw = list(SAFE_PROMPTS)
    rng.shuffle(hw)
    dev, final = hw[:60], hw[60:]
    lens = {len(p.split()) for p in dev}
    print(f"hand-written split ONCE: {len(dev)} dev-audit, {len(final)} final-audit")
    print(f"dev token-length support: {min(lens)}-{max(lens)} words\n")

    variants = {
        "V0 template grid": PL.SAFE_LARGE[:N_CORPUS],
        "V1 compositional": gen_compositional(N_CORPUS, seed=1),
        "V2 length-matched": gen_compositional(N_CORPUS, seed=2, target_lengths=lens),
        "V3 vocab-grounded": gen_vocab_grounded(N_CORPUS, dev, seed=3),
    }
    for k, v in variants.items():
        print(f"{k:<22} {len(v):>4} prompts   e.g. {v[0][:56]!r}")

    print("\nencoding ...")
    emb = {k: encode(v) for k, v in variants.items()}
    emb["hand-written dev"] = encode(dev)
    emb["hand-written final"] = encode(final)

    # geometry of each corpus, in a common metric fitted on the dev half
    wm0 = whitening_map(emb["hand-written dev"], m=16)
    print(f"\n{'='*94}\nCORPUS GEOMETRY (measured in a common metric)\n{'='*94}")
    # d_eff is compared at MATCHED n: the two-NN estimator is biased at small
    # sample size, so 900 generated points cannot be compared with 60 written
    # ones directly. Every corpus is subsampled to 60, averaged over 20 draws.
    print(f"{'corpus':<22}{'n':>6}{'d_eff@60':>10}{'mean pair dist':>16}{'mean words':>12}")
    print("-" * 94)
    r_ss = np.random.default_rng(0)
    for k in list(variants) + ["hand-written dev"]:
        W = wm0(emb[k])
        src = variants[k] if k in variants else dev
        Dm = np.linalg.norm(W[:, None] - W[None], axis=2)
        de = np.mean([d_eff(W[r_ss.choice(len(W), min(60, len(W)), replace=False)])
                      for _ in range(20)])
        print(f"{k:<22}{len(W):>6}{de:>10.1f}"
              f"{Dm[np.triu_indices(len(W), 1)].mean():>16.2f}"
              f"{np.mean([len(p.split()) for p in src]):>12.1f}")

    print(f"\n{'='*94}\nAUDIT against the DEV holdout   (iterate here)\n{'='*94}")
    print(f"{'corpus':<22}{'FRR':>8}{'p':>11}{'radius x':>10}{'verdict':>9}   shift")
    print("-" * 94)
    reports = {}
    for k in variants:
        r = audit_corpus(emb[k], emb["hand-written dev"])
        reports[k] = r
        print(f"{k:<22}{r.frr:>8.3f}{r.p_value:>11.4g}{r.radius_factor:>10.2f}"
              f"{r.verdict.upper():>9}   {r.shift}")

    # positive control: the SAME process on both sides. (Duplicating the dev
    # embeddings to pad the corpus would degenerate the whitener and produce a
    # spurious failure -- the control has to be a genuine second sample.)
    ctl = audit_corpus(emb["hand-written dev"], emb["hand-written final"])
    print(f"{'[control] hw dev':<22}{ctl.frr:>8.3f}{ctl.p_value:>11.4g}"
          f"{ctl.radius_factor:>10.2f}{ctl.verdict.upper():>9}   {ctl.shift}")

    best = min(reports, key=lambda k: reports[k].frr)
    print(f"\n{'='*94}\nFINAL holdout, touched once: {best}\n{'='*94}")
    print(audit_corpus(emb[best], emb["hand-written final"]))

    # ------------------------------------------------------------- figure --
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ks = list(variants)
    x = np.arange(len(ks))
    ax[0].bar(x, [reports[k].frr for k in ks],
              color=["tab:red" if reports[k].failed else "tab:orange"
                     if reports[k].verdict == "warn" else "tab:green" for k in ks])
    ax[0].axhline(BETA, ls="--", c="k", label=f"target β={BETA}")
    ax[0].axhline(ctl.frr, ls=":", c="tab:blue", label="same-process control")
    ax[0].set_xticks(x, [k.split()[0] + "\n" + " ".join(k.split()[1:]) for k in ks], fontsize=8)
    ax[0].set_ylabel("false rejection of hand-written prompts")
    ax[0].set_title("audit against the dev holdout"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, axis="y")
    W0 = wm0(emb["hand-written dev"])
    r_f = np.random.default_rng(0)
    de60 = lambda W: np.mean([d_eff(W[r_f.choice(len(W), min(60, len(W)), replace=False)])
                              for _ in range(20)])
    ax[1].bar(x, [de60(wm0(emb[k])) for k in ks], color="tab:blue")
    ax[1].axhline(de60(W0), ls="--", c="k", label="hand-written")
    ax[1].set_xticks(x, [k.split()[0] for k in ks], fontsize=9)
    ax[1].set_ylabel("intrinsic dimension $d_{eff}$ (at n=60)")
    ax[1].set_title("corpus geometry"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    fig.suptitle("Can a synthetic corpus be made representative?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(OUT / "s_corpus_design.png", dpi=150)
    print(f"\nfigure -> {OUT}/s_corpus_design.png")


if __name__ == "__main__":
    main()
