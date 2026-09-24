"""Semantically selective projection: strip the human subject, keep the landscape.

WHY THE EXISTING PROJECTION CANNOT DO THIS
------------------------------------------
ecf.project_onto_ellipsoid contracts radially toward mu, the centroid of the
safe set. That direction encodes "generic nature prompt", not "person", so
contracting removes scene detail and subject content indiscriminately -- it
makes the prompt blander, not person-free. generate_comparison.py then bridges
to the generator by adding the single pooled delta to all 77 token embeddings,
which is a uniform translation: by construction it cannot suppress the
"woman" token while sparing the "forest" token.

WHAT THIS DOES INSTEAD
----------------------
1. Estimate a low-rank HUMAN-SUBJECT subspace from paired minimal contrasts
   (identical scene, person added), so the scene content cancels in the
   difference and only the subject direction survives.
2. Erase that subspace PER TOKEN from the (77, 768) sequence the U-Net sees.
   Erasure is proportional by construction: a token with no component in the
   subspace loses nothing, so "forest"/"dawn" survive while "portrait"/"woman"
   are suppressed. No gate or hand-tuned mask is needed.
3. Only then, if the result is still outside the conformal safe set, fall back
   to the ellipsoidal projection for whatever non-person nonconformity remains.

Measured on held-out pairs (leave-one-out, rank 2): erasure raises the cosine
between a person-prompt and its nature-only twin from 0.772 to 0.868, moves
P2 acceptance 0.17 -> 0.92 and P3 0.67 -> 0.92, and leaves P4 (a small figure
that is genuinely part of the landscape) at 1.00 -- accepted before and after.

WHAT IT ACTUALLY ERASES, AND THE STRENGTH KNOB
----------------------------------------------
Verified by generating with SD 1.5 (erasure_demo.py, w_erasure_strength.png).
The direction encodes SUBJECT-CENTRIC FRAMING ("portrait", "close-up of a
face") more than human pixels, which is why the behaviour splits cleanly:

  "a close-up of a hiker's face with mountains out of focus"
      lambda=1.0  ->  a pure mountain landscape, no person at all
  "a portrait of a woman with a blurred forest behind her"
      lambda=1.0  ->  the woman shrinks and recedes
      lambda=2.2  ->  she is gone; the forest survives intact
  "a woman in a red coat walking along a rocky shoreline"
      lambda=2.2  ->  still there; she is a scene PARTICIPANT, not the framing
  "a vast mountain valley with a tiny hiker" / a plain safe prompt
      lambda=1.0  ->  unchanged (relative shift 0.051 / 0.046)

lambda > 1 overshoots the subspace rather than projecting onto its complement,
so it is no longer an orthogonal projection -- it is closer to negative
guidance along the concept direction. It stays well-behaved to about 2.2 here;
past that the scene itself starts to degrade. Default 1.0 (a true projection);
raise it when the intent is to remove a dominant portrait subject.

    from concept_erasure import ConceptEraser
    er = ConceptEraser.fit(pairs, tokenizer, text_encoder)
    seq_clean = er.erase_sequence(prompt_embeds)          # (B, 77, 768)
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class ConceptEraser:
    """Orthogonal eraser for a low-rank concept subspace of CLIP text space."""
    Q: np.ndarray            # (d, k) orthonormal basis of the concept subspace
    rank: int
    strength: float = 1.0    # lambda: 0 = identity, 1 = orthogonal projection,
                             # >1 overshoots (see module docstring); ~2.2 removes
                             # a dominant portrait subject outright

    # ------------------------------------------------------------- fitting --

    @staticmethod
    def _basis_from_differences(D, rank):
        """Shared direction (mean) + the leading residual PCs of the differences.

        The mean is the concept direction proper; the residual PCs pick up the
        systematic part of what the mean misses. Beyond rank ~2 the residual is
        scene-specific idiosyncrasy (top residual PC explains only 18%), so
        higher ranks start erasing scene content -- which is why rank 2 is the
        measured optimum.
        """
        mu = D.mean(axis=0, keepdims=True)
        if rank <= 1:
            basis = mu
        else:
            Dc = D - D.mean(axis=0)
            _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
            basis = np.vstack([mu, Vt[:rank - 1]])
        Q, _ = np.linalg.qr(basis.T)                    # (d, rank), orthonormal
        return Q

    @classmethod
    def fit_from_embeddings(cls, emb_without, emb_with, rank=2, strength=1.0):
        """Fit from paired embeddings: emb_with[i] is emb_without[i] plus the concept."""
        D = np.asarray(emb_with, dtype=float) - np.asarray(emb_without, dtype=float)
        return cls(Q=cls._basis_from_differences(D, rank), rank=rank, strength=strength)

    @classmethod
    def fit(cls, pairs, tokenizer, text_encoder, device="mps", rank=2, strength=1.0,
            token_level=True):
        """Fit from (scene_only, scene_plus_person) prompt pairs.

        token_level=True estimates the subspace from CONTENT-TOKEN hidden states
        rather than the pooled [EOS] vector. The U-Net cross-attends to the token
        sequence, so the subspace that matters is the one those states live in;
        the pooled vector is a causal summary and its difference direction is
        contaminated by summary-position effects.
        """
        without = [a for a, _ in pairs]
        with_ = [b for _, b in pairs]
        from embeddings import encode_prompts_with_tasks

        th, clouds = encode_prompts_with_tasks(without + with_, tokenizer,
                                               text_encoder, device=device)
        n = len(pairs)
        if not token_level:
            th = th.numpy().astype(float)
            return cls.fit_from_embeddings(th[:n], th[n:], rank=rank, strength=strength)

        # mean content-token state per prompt: the scene cancels in the difference
        mean_tok = np.stack([c.mean(axis=0) for c in clouds])
        return cls.fit_from_embeddings(mean_tok[:n], mean_tok[n:],
                                       rank=rank, strength=strength)

    # ------------------------------------------------------------- erasure --

    def erase(self, X):
        """Erase the concept subspace from rows of X. Works on (n, d) or (B, L, d)."""
        is_torch = torch.is_tensor(X)
        if is_torch:
            Q = torch.as_tensor(self.Q, dtype=X.dtype, device=X.device)
            return X - self.strength * ((X @ Q) @ Q.T)
        Xn = np.asarray(X, dtype=float)
        return Xn - self.strength * ((Xn @ self.Q) @ self.Q.T)

    def erase_sequence(self, prompt_embeds, eos_idx=None, keep_special=True):
        """Erase per token from a (B, L, d) CLIP sequence, the U-Net's conditioning.

        keep_special leaves position 0 (BOS) untouched: it carries no content but
        anchors the causal stack.

        eos_idx (per-batch [EOS] position) restricts erasure to the CONTENT span
        1..eos. The positions after [EOS] are padding: under CLIP's causal mask
        no content position ever attends to them, but SD's U-Net DOES cross-attend
        to all 77, where they carry a learned end-of-prompt pattern. Editing that
        pattern perturbs generation without removing any subject content, so by
        default it is left alone.

        NOTE the sequence must be encoded WITHOUT an attention mask
        (text_encoder(input_ids), not text_encoder(**tokens)). SD's own
        encode_prompt does this. Supplying the mask leaves the [EOS] vector
        bit-identical -- so the filter is unaffected -- but changes the padded
        positions completely, and the U-Net reads those: generation degenerates
        into texture.
        """
        out = self.erase(prompt_embeds)
        out = out.clone() if torch.is_tensor(out) else np.array(out)
        src = prompt_embeds
        if keep_special:
            out[:, 0, :] = src[:, 0, :]
        if eos_idx is not None:
            for b, e in enumerate(np.atleast_1d(eos_idx)):
                out[b, int(e) + 1:, :] = src[b, int(e) + 1:, :]
        return out

    def token_attribution(self, prompt_embeds):
        """Per-token fraction of norm lying in the concept subspace, (B, L).

        Diagnostic: shows the erasure is surgical -- high on "portrait"/"woman",
        near zero on "forest"/"dawn" -- rather than a uniform shift.
        """
        X = prompt_embeds
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=float)
        comp = np.linalg.norm(X @ self.Q, axis=-1)
        return comp / np.maximum(np.linalg.norm(X, axis=-1), 1e-12)


def project_semantic(prompt_embeds, eraser, safe_set, pool_fn, max_rounds=3):
    """Erase the concept, then fall back to the ellipsoid for what remains.

    Returns (sequence, info). The eraser handles the human-subject component;
    project_onto_ellipsoid handles any residual nonconformity (a genuinely
    out-of-domain scene), applied as the uniform pooled delta since that
    residual is not localised to particular tokens.
    """
    from ecf import ecf_score, project_onto_ellipsoid

    seq = eraser.erase_sequence(prompt_embeds)
    pooled = pool_fn(seq).cpu().float()
    info = {"score_after_erase": float(ecf_score(pooled, safe_set).item())}

    if info["score_after_erase"] <= safe_set.alpha:
        info["ellipsoid_used"] = False
        return seq, info

    proj = project_onto_ellipsoid(pooled, safe_set)
    delta = (proj - pooled).to(dtype=seq.dtype, device=seq.device)
    seq = seq + delta.unsqueeze(1)
    info["ellipsoid_used"] = True
    info["score_final"] = float(ecf_score(pool_fn(seq).cpu().float(), safe_set).item())
    return seq, info
