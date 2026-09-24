"""Deploy a conformal safe set as a PROJECTION STEP inside the generation pipeline.

The filter never refuses a prompt. It replaces the conditioning tensor by its
nearest conforming point, so generation always proceeds and always proceeds from
inside the calibrated safe set.

    guard = ConformalGuard.fit(safe_sequences, eos_idx, score="orderinf")
    attach(pipe, guard)                 # from now on pipe(prompt=...) is filtered

THE LIFT PROBLEM
----------------
The safe set is a region in a vector space; the U-Net consumes a (77, 768)
sequence. Three ways to connect them, all of them pure conformal prediction on
a vector space -- they differ only in what counts as a "sample".

  lift="pooled"     sample = the pooled vector of a prompt (R^768).
                    Project the pooled vector, then add the SAME delta to every
                    token row. The lift is exact: shifting all rows by delta
                    shifts both the [EOS] row and the mean by exactly delta, so
                    the projected sequence pools precisely to the projected
                    point. Cheapest, and the only one that leaves relative token
                    structure untouched -- it is a rigid translation.

  lift="per_token"  sample = one token hidden state (R^768). The region
                    constrains every position the U-Net attends to, which is the
                    tightest control over generation. Calibrated at the PROMPT
                    level (score of a prompt = max over its content tokens) so
                    the marginal guarantee is per prompt, not per token: token
                    scores within a prompt are strongly dependent and are not
                    exchangeable with tokens of another prompt.

  lift="sequence"   sample = the whole flattened (77*768) sequence, one point.
                    Conceptually the cleanest -- one prompt, one vector, one
                    projection -- but the ambient dimension is 59136 with only
                    ~60 anchors, so the geometry is extremely sparse. Provided
                    for completeness; `pooled` and `per_token` are the practical
                    choices.

All three give a certified post-condition: after guard(), the conditioning is
inside the calibrated conformal safe set.

WHICH REGION
------------
Any ConformalRegion works. `fit` builds the standard ones by name:

    "ellipsoid"         ECF / order-1 Mahalanobis   -> EllipsoidRegion
    "order1"            s_(d,1)                     -> SumOfNormsRegion
    "orderinf"          s_(d,inf)                   -> union of balls (default)
    "lowcard_order1"    s~_(d,1)                    -> SumOfNormsRegion on centroids
    "lowcard_orderinf"  s~^(n)_(d,inf)              -> union of balls, per-ball radii

Task-adaptive variants (g_(d,1), g^(n,K)_(d,inf)) select their anchors from the
candidate's task descriptor, so the region is per-prompt: pass `task_distances`
to fit() and `adaptive_K`, and the guard rebuilds the region for each prompt
from the K task-nearest anchors.

METRIC
------
Scores and projections are computed in the whitened coordinates of the reference
split's probabilistic-PCA covariance (ecf.tex's Mahalanobis metric). Whitening
is an affine bijection, so projecting there and mapping back is exactly the
Mahalanobis projection in the original space.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from dataclasses import dataclass, field

from conformal_regions import (
    ConformalRegion, EllipsoidRegion, SumOfNormsRegion, UnionOfBallsRegion,
    region_from_order1, region_from_orderinf, region_from_lowcard_orderinf,
)
from wasserstein import (conformal_radius, score_order1, score_orderinf,
                         whitening_map, LowCardOrder1, LowCardOrderInf)

class UnrepresentativeCorpus(RuntimeError):
    """The calibration corpus failed validation against an external holdout."""


LIFTS = ("pooled", "per_token", "sequence")
SCORES = ("ellipsoid", "order1", "orderinf", "lowcard_order1", "lowcard_orderinf")


# ------------------------------------------------------------- extraction --

def pooled_from(seq, eos_idx, pool="eos"):
    """(N, L, d) -> (N, d). `eos` takes the [EOS] row, `mean` averages content rows."""
    seq = np.asarray(seq, dtype=float)
    eos_idx = np.asarray(eos_idx, dtype=int)
    if pool == "eos":
        return seq[np.arange(len(seq)), eos_idx]
    if pool == "mean":
        return np.stack([seq[i, 1:e].mean(axis=0) if e > 1 else seq[i, e]
                         for i, e in enumerate(eos_idx)])
    raise ValueError(f"pool must be 'eos' or 'mean', got {pool}")


def content_rows(seq, eos_idx):
    """(N, L, d) -> list of (L_i, d) content blocks (positions 1..eos-1)."""
    seq = np.asarray(seq, dtype=float)
    out = []
    for i, e in enumerate(np.asarray(eos_idx, dtype=int)):
        out.append(seq[i, 1:e] if e > 1 else seq[i, e:e + 1])
    return out


# ------------------------------------------------------------------ guard --

@dataclass
class ConformalGuard:
    """A calibrated safe set plus the machinery to project a conditioning tensor."""
    region: ConformalRegion
    whiten: object                    # callable (n, d) -> (n, d)
    unwhiten: object                  # callable (n, d) -> (n, d)
    lift: str = "pooled"
    pool: str = "eos"
    epsilon: float = 0.05
    score_name: str = "orderinf"
    seq_shape: tuple = ()
    # task-adaptive support
    anchors_w: np.ndarray | None = None       # (M, d) whitened anchors
    adaptive_K: int | None = None
    radius: float = np.inf
    #: corpus_audit.CorpusAudit report, when fit() was given an audit holdout
    audit: object = None
    #: repair strength in [0, 1]. 0 = the plain metric projection (minimum
    #: displacement, lands on the region's boundary in empty certified space).
    #: 1 = land exactly on the nearest real safe point (maximum support
    #: fidelity). Every value is certified inside the region -- see
    #: ConformalRegion.repair.
    repair_t: float = 0.0

    # ------------------------------------------------------------- fitting --

    @classmethod
    def fit(cls, sequences, eos_idx, epsilon=0.05, score="orderinf", lift="pooled",
            pool="eos", m_pca=32, n_ref=None, lowcard_K=20, n_cuts=0,
            adaptive_K=None, seed=0, audit_sequences=None, audit_eos=None,
            audit_raises=False):
        """Calibrate on safe prompt sequences.

        sequences : (N, L, d) CLIP hidden states of the SAFE prompts, encoded the
                    way the pipeline encodes them (input_ids only, no attention
                    mask -- see concept notes in wasserstein.whitening_map).
        eos_idx   : (N,) [EOS] position of each prompt.

        The N prompts are split into a reference half (defines the geometry) and
        a calibration half (defines the radius); they must be disjoint for the
        conformal guarantee to hold.

        audit_sequences / audit_eos : an optional holdout drawn from a DIFFERENT
        SOURCE than `sequences`, believed to be genuinely in-domain. Supplying it
        runs corpus_audit and stores the report on `.audit`; a failing audit
        warns, or raises if audit_raises=True. Coverage on held-out calibration
        data CANNOT detect an unrepresentative corpus -- it reported 0.95 while
        the filter rejected two thirds of real prompts (README section 11) -- so
        this check is the only thing standing between a valid guarantee and a
        useless one. Run it before comparing estimators.
        """
        if lift not in LIFTS:
            raise ValueError(f"lift must be one of {LIFTS}")
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}")

        sequences = np.asarray(sequences, dtype=float)
        N, L, d = sequences.shape
        eos_idx = np.asarray(eos_idx, dtype=int)
        n_ref = n_ref if n_ref is not None else N // 2

        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)
        ref_i, cal_i = perm[:n_ref], perm[n_ref:]

        # --- build the sample matrices for this lift ---
        if lift == "pooled":
            X = pooled_from(sequences, eos_idx, pool)
            ref, cal = X[ref_i], X[cal_i]
            cal_groups = [np.array([j]) for j in range(len(cal))]
        elif lift == "sequence":
            X = sequences.reshape(N, L * d)
            ref, cal = X[ref_i], X[cal_i]
            cal_groups = [np.array([j]) for j in range(len(cal))]
        else:                                     # per_token
            blocks = content_rows(sequences, eos_idx)
            ref = np.vstack([blocks[i] for i in ref_i])
            cal_list, cal_groups, k = [], [], 0
            for i in cal_i:
                b = blocks[i]
                cal_list.append(b)
                cal_groups.append(np.arange(k, k + len(b)))
                k += len(b)
            cal = np.vstack(cal_list)

        # --- metric: whiten on the reference sample only ---
        wm = whitening_map(ref, m=min(m_pca, ref.shape[0] - 1, ref.shape[1]))
        ref_w, cal_w = wm(ref), wm(cal)

        unwhiten = wm.inverse            # closed form, see wasserstein.Whitener

        # --- score every calibration sample, aggregate to PROMPT level ---
        raw = _score_samples(score, cal_w, ref_w, n_cuts=n_cuts, lowcard_K=lowcard_K)
        prompt_scores = np.array([raw[g].max() for g in cal_groups])
        radius = conformal_radius(prompt_scores, epsilon)

        region = _build_region(score, ref_w, radius, n_cuts=n_cuts, lowcard_K=lowcard_K)

        guard = cls(region=region, whiten=wm, unwhiten=unwhiten, lift=lift, pool=pool,
                    epsilon=epsilon, score_name=score, seq_shape=(L, d),
                    anchors_w=ref_w, adaptive_K=adaptive_K, radius=radius)

        if audit_sequences is not None:
            from corpus_audit import audit_guard
            guard.audit = audit_guard(guard, sequences[cal_i], eos_idx[cal_i],
                                      np.asarray(audit_sequences, dtype=float),
                                      np.asarray(audit_eos, dtype=int))
            if guard.audit.failed:
                msg = ("corpus audit FAILED -- the calibration corpus is not "
                       f"representative of the audit source:\n{guard.audit}")
                if audit_raises:
                    raise UnrepresentativeCorpus(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return guard

    # ------------------------------------------------------------ guarding --

    def _region_for(self, task_dists=None):
        """The region for this candidate: fixed, or rebuilt from K task-nearest anchors."""
        if self.adaptive_K is None or task_dists is None:
            return self.region
        sel = np.argsort(np.asarray(task_dists))[:self.adaptive_K]
        return region_from_orderinf(self.anchors_w[sel], self.radius, n_cuts=0)

    def project_vectors(self, X, task_dists=None):
        """Project raw-space rows onto the safe set (whiten -> project -> unwhiten)."""
        reg = self._region_for(task_dists)
        Xw = self.whiten(np.atleast_2d(np.asarray(X, dtype=float)))
        return self.unwhiten(reg.repair(Xw, t=self.repair_t))

    def guard(self, prompt_embeds, eos_idx, task_dists=None):
        """Project a (B, L, d) conditioning tensor into the safe set.

        Returns (new_embeds, info). `info` reports, per batch element, the score
        before and after and the Euclidean displacement applied.
        """
        is_torch = torch.is_tensor(prompt_embeds)
        dev = prompt_embeds.device if is_torch else None
        dt = prompt_embeds.dtype if is_torch else None
        seq = (prompt_embeds.detach().cpu().numpy().astype(float) if is_torch
               else np.asarray(prompt_embeds, dtype=float))
        eos_idx = np.atleast_1d(np.asarray(eos_idx, dtype=int))
        reg = self._region_for(task_dists)
        out = seq.copy()
        info = {"lift": self.lift, "score": self.score_name, "radius": self.radius}

        if self.lift == "pooled":
            p = pooled_from(seq, eos_idx, self.pool)
            pw = self.whiten(p)
            info["score_before"] = reg.score(pw)
            proj = self.unwhiten(reg.repair(pw, t=self.repair_t))
            delta = proj - p                                   # (B, d)
            out = seq + delta[:, None, :]                      # rigid translation
            info["score_after"] = reg.score(self.whiten(pooled_from(out, eos_idx, self.pool)))

        elif self.lift == "sequence":
            L, d = self.seq_shape
            flat = seq.reshape(len(seq), L * d)
            fw = self.whiten(flat)
            info["score_before"] = reg.score(fw)
            out = self.unwhiten(reg.repair(fw, t=self.repair_t)).reshape(seq.shape)
            info["score_after"] = reg.score(self.whiten(out.reshape(len(seq), L * d)))

        else:                                                  # per_token
            before, after = [], []
            for b, e in enumerate(eos_idx):
                lo, hi = 1, max(int(e), 2)
                rows = seq[b, lo:hi]
                rw = self.whiten(rows)
                before.append(reg.score(rw).max())
                out[b, lo:hi] = self.unwhiten(reg.repair(rw, t=self.repair_t))
                after.append(reg.score(self.whiten(out[b, lo:hi])).max())
            info["score_before"] = np.array(before)
            info["score_after"] = np.array(after)

        info["displacement"] = np.linalg.norm(
            (out - seq).reshape(len(seq), -1), axis=1)
        info["was_outside"] = np.asarray(info["score_before"]) > reg.radius

        if is_torch:
            out = torch.as_tensor(out, dtype=dt, device=dev)
        return out, info

    __call__ = guard


def _score_samples(score, Zw, ref_w, n_cuts=0, lowcard_K=20):
    if score == "ellipsoid":
        mu = ref_w.mean(axis=0)
        return 1.0 + ((Zw - mu) ** 2).sum(axis=1)          # whitened => Sigma = I
    if score == "order1":
        return score_order1(Zw, ref_w)
    if score == "orderinf":
        return score_orderinf(Zw, ref_w, n_cuts=n_cuts)
    if score == "lowcard_order1":
        return LowCardOrder1.fit(ref_w, K=min(lowcard_K, len(ref_w)))(Zw)
    if score == "lowcard_orderinf":
        return LowCardOrderInf.fit(ref_w, K=min(lowcard_K, len(ref_w)), n_cuts=n_cuts)(Zw)
    raise ValueError(score)


def _build_region(score, ref_w, radius, n_cuts=0, lowcard_K=20):
    if score == "ellipsoid":
        d = ref_w.shape[1]
        mu = ref_w.mean(axis=0)
        return EllipsoidRegion(mean=mu, V=np.zeros((d, 0)), eig=np.zeros(0),
                               gamma=1.0, alpha=radius)       # whitened: Sigma = I
    if score == "order1":
        return region_from_order1(ref_w, radius)
    if score == "orderinf":
        return region_from_orderinf(ref_w, radius, n_cuts=n_cuts)
    if score == "lowcard_order1":
        lc = LowCardOrder1.fit(ref_w, K=min(lowcard_K, len(ref_w)))
        return region_from_order1(lc.centroids, radius, weights=lc.w_k,
                                  scale=lc.M + 1.0, offset=lc.const)
    if score == "lowcard_orderinf":
        lc = LowCardOrderInf.fit(ref_w, K=min(lowcard_K, len(ref_w)), n_cuts=n_cuts)
        return region_from_lowcard_orderinf(lc, radius)
    raise ValueError(score)


# --------------------------------------------------------- pipeline hookup --

def attach(pipe, guard, verbose=False):
    """Wrap `pipe.encode_prompt` so every generation is filtered. Returns a detach().

    After attaching, ordinary calls work unchanged:

        attach(pipe, guard)
        pipe("a portrait of a woman in a forest").images[0]   # conditioning projected

    Only the positive conditioning is projected; the negative prompt is left
    alone (it is not a sample from the safe distribution).
    """
    if getattr(pipe, "_conformal_original_encode", None) is not None:
        raise RuntimeError("a guard is already attached; call detach() first")
    original = pipe.encode_prompt
    tokenizer = pipe.tokenizer

    def wrapped(prompt, device, num_images_per_prompt, do_classifier_free_guidance,
                negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None,
                lora_scale=None, clip_skip=None, **kw):
        pe, ne = original(prompt, device, num_images_per_prompt,
                          do_classifier_free_guidance, negative_prompt,
                          prompt_embeds, negative_prompt_embeds, lora_scale,
                          clip_skip, **kw)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt or [])
        if not prompts:                       # caller supplied embeddings directly
            eos = np.full(len(pe), pe.shape[1] - 1)
        else:
            ids = tokenizer(prompts, padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").input_ids
            eos = ids.argmax(dim=-1).numpy()
            if len(eos) != len(pe):           # repeated by num_images_per_prompt
                eos = np.repeat(eos, len(pe) // max(len(eos), 1))
        out, info = guard.guard(pe, eos)
        if verbose:
            for i in range(len(np.atleast_1d(info["score_before"]))):
                print(f"[conformal] score {np.atleast_1d(info['score_before'])[i]:.3f}"
                      f" -> {np.atleast_1d(info['score_after'])[i]:.3f}"
                      f" (radius {guard.region.radius:.3f}, moved"
                      f" {info['displacement'][i]:.3f})")
        return out, ne

    pipe.encode_prompt = wrapped
    pipe._conformal_original_encode = original

    def detach():
        pipe.encode_prompt = pipe._conformal_original_encode
        pipe._conformal_original_encode = None
    return detach
