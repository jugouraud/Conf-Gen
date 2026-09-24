"""The deployable prompt filter: the configurations that actually won the sweeps.

WHAT THE EXPERIMENTS SETTLED (run_wasserstein_experiment.py, sweep_reweighting.py;
100-200 random splits, M=60 reference / K=40 calibration / 20 test, eps=0.05)

  metric      The MAHALANOBIS metric of the reference split's probabilistic-PCA
              covariance, not raw Euclidean. ecf.tex motivates it (it is the
              metric under which s_(d,1) links to the order-1 inverse ECF), and
              it roughly doubles subtle-prompt detection: 0.216 -> 0.384.

  adaptation  HARD K-NEAREST SELECTION in task space. Smooth mass reallocation
              -- the manuscript's default inv_power, and softmax -- does not
              help and monotonically hurts as it sharpens (inv^8: -0.023 vs
              uniform). CLIP token-cloud distances sit in a narrow band
              (~31 +/- 4), so an inverse-power law cannot concentrate; only a
              hard cut can. knn-3 is +0.029 over uniform (> 2 s.e., paired).

  n_cuts      ZERO. Cuts are actively harmful here: degenerate detection falls
              from 1.000 to 0.05-0.15 at every K tried. B^(n) drops the n
              heaviest MST edges assuming they bridge modes -- but when the
              candidate IS the outlier, its own attachment edge is the heaviest,
              so cutting deletes exactly the signal. Cuts need a genuinely
              multimodal reference set; a single-domain safe set is not one.

  cardinality Free. s~_(d,1) is flat in K (0.367 from K=5 to K=60): 12x memory
              reduction at no cost. s~_(d,inf) has an interior optimum at
              K~20-30 that BEATS the exact score (0.396 vs 0.365 subtle),
              reproducing the manuscript's own observation that the clustered
              regions are less conservative than the exact ones.

Two configurations are worth deploying; ORDERINF is the default.

    ORDERINF  g^(0,K)_(d,inf), K=5 task-nearest anchors    subtle .400  border .563
    ORDER1    g_(d,1) with knn-3 mass                      subtle .396  border .563
                                            (ECF baseline: subtle .367  border .536)

Both hold coverage (0.944-0.948 against the exact split-CP expectation 0.9512).

    from prompt_filter import PromptFilter
    f = PromptFilter.fit(safe_prompts, epsilon=0.05)
    f.accepts(["a golden sunset over calm ocean waves"])       # -> array([True])
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from wasserstein import (conformal_radius, score_orderinf, whitening_map,
                         Reweighter, _cd)

VARIANTS = ("orderinf", "order1")


@dataclass
class PromptFilter:
    """Conformal prompt filter on CLIP conditioning embeddings.

    Fitted on a reference split (the anchors + the whitening covariance) and
    calibrated on a disjoint split (the radius), so split-conformal validity
    holds: P[safe prompt accepted] >= 1 - epsilon, marginally.
    """
    variant: str
    whiten: object                    # callable: raw embeddings -> whitened coords
    ref_w: np.ndarray                 # (M, d) whitened reference anchors
    reweighter: Reweighter            # task-space distances to the reference clouds
    radius: float
    epsilon: float
    K: int = 5                        # task-nearest anchors (orderinf) / knn mass (order1)
    n_cuts: int = 0
    m_pca: int = 32

    # ---------------------------------------------------------------- fit --

    @classmethod
    def fit(cls, ref_theta, ref_clouds, cal_theta, cal_clouds,
            epsilon=0.05, variant="orderinf", K=None, m_pca=32, n_cuts=0):
        """Fit on the reference split, calibrate the radius on a disjoint split.

        ref_theta/cal_theta : (n, d) pooled [EOS] embeddings
        ref_clouds/cal_clouds : lists of (L_i, d) content-token clouds
        """
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if K is None:
            K = 5 if variant == "orderinf" else 3
        if n_cuts != 0:
            import warnings
            warnings.warn(
                "n_cuts > 0 collapsed degenerate detection to 0.05-0.15 in every "
                "configuration tested: the candidate's own MST edge is the heaviest, "
                "so cutting removes the outlier signal. Only use cuts on a genuinely "
                "multimodal safe set.", RuntimeWarning)

        ref_theta = np.asarray(ref_theta, dtype=float)
        wm = whitening_map(ref_theta, m=m_pca)
        rw = Reweighter(ref_clouds, scheme="knn", knn=K, rescaling_order=1.0)

        f = cls(variant=variant, whiten=wm, ref_w=wm(ref_theta), reweighter=rw,
                radius=np.inf, epsilon=epsilon, K=K, n_cuts=n_cuts, m_pca=m_pca)
        f.radius = conformal_radius(f.score(cal_theta, cal_clouds), epsilon)
        return f

    # -------------------------------------------------------------- score --

    def score(self, theta, clouds):
        """Nonconformity score for each (prompt embedding, token cloud) pair."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        Zw = self.whiten(theta)
        M = len(self.ref_w)

        if self.variant == "orderinf":
            # g^(n,K)_(d,inf): exact n-cuts bottleneck on the K task-nearest anchors
            out = np.empty(len(Zw))
            for i, cloud in enumerate(clouds):
                d_t = self.reweighter.dists_for(cloud)
                sel = np.argsort(d_t)[:min(self.K, M)]
                out[i] = score_orderinf(Zw[i:i + 1], self.ref_w[sel],
                                        n_cuts=self.n_cuts)[0]
            return out

        # g_(d,1): knn-3 mass reallocation over all M anchors
        W = np.stack([self.reweighter.weights_for(c) for c in clouds])
        return (_cd(Zw, self.ref_w) * W).sum(axis=1) / (M + 1)

    def accepts(self, theta, clouds):
        """True where the prompt falls inside the conformal safe set."""
        return self.score(theta, clouds) <= self.radius

    # ------------------------------------------------------- convenience --

    @staticmethod
    def encode(prompts, device="mps"):
        """Encode raw prompt strings into the (theta, clouds) pair this filter needs."""
        from embeddings import load_text_encoder, encode_prompts_with_tasks
        tok, enc = load_text_encoder(device=device)
        theta, clouds = encode_prompts_with_tasks(prompts, tok, enc, device=device)
        return theta.numpy().astype(float), clouds
