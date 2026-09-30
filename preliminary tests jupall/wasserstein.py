"""
Wasserstein support-estimation score functions on the prompt-embedding space.

Direct port of the score functions in meta_full_paper/ (content.tex,
lower_cardinality.tex, idea_proof_lower_cardinality_inf.tex, adaptation.tex) to
the diffusion prompt-filter setting.

MAPPING TO THE MANUSCRIPT
-------------------------
The manuscript scores a candidate WEIGHT theta against M historical weights
{theta_i} produced by a deterministic pipeline f from tasks {tau_i}. Here:

    theta_i  = pooled [EOS] CLIP embedding of safe prompt i        (R^768)
    tau_i    = the cloud of CONTENT-token hidden states of prompt i (R^{L_i x 768})
    f        = CLIP's causal-attention pooling map, tau -> theta

f is deterministic and continuous, so Assumption as:proximity (pipeline
continuity) holds in the form the adaptive scores need: prompts with nearby
token clouds have nearby pooled embeddings. BOS/EOS are excluded from tau so
that the task descriptor is not a trivial copy of the weight it predicts.

SCORE FUNCTIONS
---------------
    s_{d,1}            score_order1              content.tex def:sym_score
    s^{(n)}_{d,inf}    score_orderinf            content.tex def:sym_score_inf
    g_{d,1}            reweighted_score_order1   adaptation.tex eq:adaptive_W1
    g^{(n,K)}_{d,inf}  reweighted_score_orderinf adaptation.tex eq:adaptive_score_winf
    s~_{d,1}           LowCardOrder1             lower_cardinality.tex lem:sym_score_tract
    s~^{(n)}_{d,inf}   LowCardOrderInf           idea_proof_lower_cardinality_inf.tex

All are symmetric in the reference set (permutation-invariant), so they are
admissible for full conformal prediction as well as the split setting used here.

THREADING. On macOS, calling scikit-learn's KMeans (LowCardOrder1/OrderInf) and
then POT's emd2 (Reweighter) in one process segfaults -- two OpenMP runtimes get
loaded. Pin the thread pools BEFORE importing numpy, as every script here does:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")

METRIC. The manuscript's low-cardinality propositions require d to be a NORM
(the triangle-inequality/Jensen steps). Euclidean is the default. Cosine
distance is offered for convenience but is not norm-induced, so the
lower-cardinality guarantees do not transfer to it.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree

try:
    import ot as pot
except ImportError:                                     # pragma: no cover
    pot = None


# ---------------------------------------------------------------- utilities --

def _cd(A, B, metric="euclidean"):
    """Pairwise distances, (nA, nB)."""
    return cdist(np.atleast_2d(A), np.atleast_2d(B), metric=metric)


def conformal_radius(scores, epsilon: float) -> float:
    """Split-conformal radius: the ceil((K+1)(1-eps))-th smallest calibration score.

    content.tex:325 -- sort the K calibration scores, append +inf as the
    (K+1)-th element, return r_{ceil((K+1)(1-eps))}. When that rank exceeds K
    the correct radius IS +inf (region = whole space: vacuous but VALID).
    Requires K >= ceil(1/eps) - 1 for a finite radius.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    rank = int(np.ceil((1.0 - epsilon) * (n + 1)))
    return float("inf") if rank > n else float(s[rank - 1])


# --------------------------------------------------- non-adaptive: order 1 ---

def score_order1(theta, ref, metric="euclidean"):
    """s_{d,1}(theta; D) = 1/(M(M+1)) sum_i d(theta, theta_i).

    The exact order-1 Wasserstein distance between the reference empirical
    measure and itself augmented by a Dirac at theta (content.tex
    lem:trivial_W1): the new sample's mass is split into M equal parts, each
    shipped to one reference atom.
    """
    ref = np.atleast_2d(ref)
    M = len(ref)
    return _cd(theta, ref, metric).sum(axis=1) / (M * (M + 1))


# ------------------------------------------------- non-adaptive: order inf ---

def _mst_edges_sorted(D):
    """Ascending MST edge weights of a dense symmetric distance matrix."""
    mst = minimum_spanning_tree(csr_array(D)).toarray()
    w = mst[mst > 0]
    return np.sort(w)


def score_orderinf(theta, ref, n_cuts=0, metric="euclidean"):
    """s^{(n)}_{d,inf}(theta; D) = B^{(n)}(MST(D u {theta})).

    The n-cuts bottleneck (content.tex def:pruned_forest): drop the n heaviest
    MST edges and report the heaviest survivor, i.e. the (n+1)-th heaviest edge.
    n=0 recovers the plain bottleneck = W_{d,inf}.

    Uses the cycle property to avoid an O(M^2) MST per candidate: every edge of
    MST(D u {theta}) not incident to theta is also in MST(D), so it suffices to
    run the MST over MST(D)'s M-1 edges plus theta's M edges.
    """
    theta = np.atleast_2d(theta)
    ref = np.atleast_2d(ref)
    M = len(ref)
    n_cuts = int(n_cuts)

    D_ref = _cd(ref, ref, metric)
    base = minimum_spanning_tree(csr_array(D_ref)).toarray()
    base = np.maximum(base, base.T)                      # symmetrise the M-1 kept edges

    d_new = _cd(theta, ref, metric)                      # (n, M)
    out = np.empty(len(theta))
    G = np.zeros((M + 1, M + 1))
    G[:M, :M] = base
    for i, row in enumerate(d_new):
        G[:M, M] = row
        G[M, :M] = row
        w = _mst_edges_sorted(G)
        out[i] = w[-(n_cuts + 1)] if len(w) > n_cuts else 0.0
    return out


def bottleneck_of(ref, n_cuts=0, metric="euclidean"):
    """B^{(n)}(T_M) for the reference set alone (a constant, used by LowCardOrderInf)."""
    w = _mst_edges_sorted(_cd(ref, ref, metric))
    return float(w[-(int(n_cuts) + 1)]) if len(w) > n_cuts else 0.0


# ------------------------------------------------------ adaptive: order 1 ----

def reweighted_score_order1(theta, ref, w, metric="euclidean"):
    """g_{d,1}(tau, theta; D) = 1/(M+1) sum_i w_i(tau, tau_i) d(theta, theta_i).

    adaptation.tex eq:adaptive_W1. Reduces to s_{d,1} exactly when w_i = 1/M.
    Mass is reallocated toward reference prompts whose TASK (token cloud) is
    close to the candidate's, which tightens the region around the part of the
    safe manifold the candidate actually lives near.
    """
    ref = np.atleast_2d(ref)
    M = len(ref)
    return (_cd(theta, ref, metric) @ np.asarray(w, dtype=float)) / (M + 1)


# ---------------------------------------------------- adaptive: order inf ----

def reweighted_score_orderinf(theta, ref, w, K, n_cuts=0, metric="euclidean"):
    """g^{(n,K)}_{d,inf}(tau, theta; D) = B^{(n)}(MST({theta} u {theta_sigma(i)}_{i<=K})).

    adaptation.tex eq:adaptive_score_winf. Order-inf cost depends only on the
    longest displacement, so mass reallocation cannot move it; the paper's
    prescription is SAMPLE SELECTION instead -- keep the K task-nearest
    references (largest mass) and take the exact n-cuts bottleneck on that
    subset. Requires n_cuts <= K-1.
    """
    ref = np.atleast_2d(ref)
    K = int(min(K, len(ref)))
    idx = np.argsort(np.asarray(w, dtype=float))[-K:]    # K largest masses
    return score_orderinf(theta, ref[idx], n_cuts=n_cuts, metric=metric)


# ------------------------------------------- task-space mass reallocation ----

class Reweighter:
    """Task-conditioned mass reallocation over the reference prompts (adaptation.tex).

    Holds the M reference TASK clouds (content-token hidden states) and turns a
    new prompt's cloud into a probability vector over the M reference
    embeddings, using the order-`task_order` Wasserstein distance between token
    clouds as the task-space metric d_T.

    Schemes (all map task distances d_i -> simplex masses):
        inv_power   w_i ~ (1/d_i)^r                  satisfies eq:reweight_identity
        softmax     w_i ~ exp(-r d_i / bandwidth)    smooth; identity only in the limit r->inf
        knn         keep the k nearest, w_i ~ (1/d_i)^r among them, 0 elsewhere
        knn_uniform keep the k nearest, uniform among them
        top1        all mass on the single nearest task (winner-take-all)
        uniform     w_i = 1/M (recovers the NON-adaptive scores)

    eq:reweight_identity requires tau = tau_j => w_j = 1. inv_power, knn and
    top1 satisfy it; softmax does not (it keeps mass on every reference at any
    finite bandwidth), so it trades that property for smoothness.
    """

    def __init__(self, task_clouds, metric="euclidean", task_order=1,
                 scheme="inv_power", rescaling_order=1.0, bandwidth=None,
                 knn=None, max_points=200):
        if pot is None:
            raise ImportError("task-space reweighting needs POT: pip install pot")
        self.metric = metric
        self.task_order = float(task_order)
        self.scheme = scheme
        self.rescaling_order = float(rescaling_order)
        self.bandwidth = bandwidth
        self.knn = knn
        self.max_points = int(max_points)
        self.hist = [np.ascontiguousarray(np.atleast_2d(Z)[:self.max_points], dtype=float)
                     for Z in task_clouds]
        self.M = len(self.hist)

    def _task_wasserstein(self, ZA, ZB):
        """W_{task_order}(hat ZA, hat ZB) between two token clouds, uniform marginals."""
        C = _cd(ZA, ZB, self.metric) ** self.task_order
        a = np.ones(len(ZA)) / len(ZA)
        b = np.ones(len(ZB)) / len(ZB)
        cost = float(pot.emd2(a, b, np.ascontiguousarray(C), numThreads=1))
        return cost ** (1.0 / self.task_order)

    def dists_for(self, Z):
        """Task-space distances from cloud Z to each of the M reference clouds."""
        Zc = np.ascontiguousarray(np.atleast_2d(Z)[:self.max_points], dtype=float)
        return np.array([self._task_wasserstein(Zc, h) for h in self.hist])

    def masses_from_dists(self, dists):
        d = np.asarray(dists, dtype=float) + 1e-10
        if self.scheme == "uniform":
            s = np.ones_like(d)
        elif self.scheme == "inv_power":
            s = (1.0 / d) ** self.rescaling_order
        elif self.scheme == "softmax":
            tau = self.bandwidth if self.bandwidth else float(np.median(d))
            s = np.exp(-self.rescaling_order * d / max(tau, 1e-12))
        elif self.scheme in ("knn", "knn_uniform"):
            k = int(min(self.knn or self.M, self.M))
            idx = np.argsort(d)[:k]
            s = np.zeros_like(d)
            s[idx] = 1.0 if self.scheme == "knn_uniform" else (1.0 / d[idx]) ** self.rescaling_order
        elif self.scheme == "top1":
            s = np.zeros_like(d)
            s[int(np.argmin(d))] = 1.0
        else:
            raise ValueError(f"unknown reweighting scheme: {self.scheme}")
        total = s.sum()
        if not np.isfinite(total) or total <= 0:         # a coincident task overflows inv_power
            s = np.zeros_like(d)
            s[int(np.argmin(d))] = 1.0
            total = 1.0
        return s / total

    def weights_for(self, Z):
        return self.masses_from_dists(self.dists_for(Z))


def uniform_mass(M):
    return np.ones(M) / M


def effective_support(w):
    """Participation ratio 1/sum(w^2): how many references actually carry mass."""
    w = np.asarray(w, dtype=float)
    return 1.0 / np.sum(w ** 2)


# ------------------------------------------------- low-cardinality: order 1 --

@dataclass
class LowCardOrder1:
    """s~_{d,1}: order-1 score against K k-means centroids (lower_cardinality.tex).

        s~_{d,1}(theta) = 1/(M+1) [ sum_k sum_{i in C_k} d(theta_i, cbar_k)
                                    + sum_k w_k d(theta, cbar_k) ]

    The first term is the constant M * eta_{1,M}(K); w_k = |C_k| / M. Under
    Assumption as:centroid (arithmetic-mean centroids, which k-means gives for
    the Euclidean norm) pr:suboptimal_cluster guarantees this upper-bounds the
    true W_{d,1} to the clustered measure, so it is a valid support estimator
    with the eta_{1,M}(K) slack of pr:cluster_bound.
    """
    centroids: np.ndarray
    w_k: np.ndarray
    const: float                 # sum_k sum_{i in C_k} d(theta_i, cbar_k) = M * eta_1
    M: int
    metric: str = "euclidean"
    eta1: float = 0.0

    @classmethod
    def fit(cls, ref, K, metric="euclidean", seed=0):
        from sklearn.cluster import KMeans
        ref = np.atleast_2d(ref)
        M = len(ref)
        K = int(min(K, M))
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(ref)
        C, lab = km.cluster_centers_, km.labels_
        w_k = np.array([(lab == k).sum() for k in range(K)], dtype=float) / M
        const = sum(float(_cd(ref[lab == k], C[k:k + 1], metric).sum())
                    for k in range(K) if (lab == k).any())
        return cls(centroids=C, w_k=w_k, const=const, M=M, metric=metric,
                   eta1=const / M)

    def __call__(self, theta):
        return (self.const + _cd(theta, self.centroids, self.metric) @ self.w_k) / (self.M + 1)


# ----------------------------------------------- low-cardinality: order inf --

@dataclass
class LowCardOrderInf:
    """s~^{(n)}_{d,inf}: order-inf score from K centroids + K+1 scalars.

        s~^{(n)}(theta) = eta_inf + max( B^{(n)}(T_M),
                                          min_k [ d(cbar_k, theta) + r_k ] )

    with r_k the cluster radius and eta_inf = max_k r_k
    (idea_proof_lower_cardinality_inf.tex pr:suboptimal_cluster_inf). Inference
    memory is the K centroids plus {r_k} and B^{(n)}(T_M) -- independent of M.

    NOTE the max(B, .) floor makes the score CONSTANT for every theta within
    B - r_k of some centroid. Calibration scores then tie at eta + B, and if
    more than (1-eps) of them sit on that floor the conformal radius lands on
    the floor itself. That is still valid, just uninformative -- the region
    degenerates to the union of balls {theta : min_k [d(cbar_k,theta)+r_k] <= B}.
    """
    centroids: np.ndarray
    r_k: np.ndarray
    eta_inf: float
    B_n: float
    metric: str = "euclidean"

    @classmethod
    def fit(cls, ref, K, n_cuts=0, metric="euclidean", seed=0):
        from sklearn.cluster import KMeans
        ref = np.atleast_2d(ref)
        K = int(min(K, len(ref)))
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(ref)
        C, lab = km.cluster_centers_, km.labels_
        r_k = np.array([
            float(_cd(ref[lab == k], C[k:k + 1], metric).max()) if (lab == k).any() else 0.0
            for k in range(K)
        ])
        return cls(centroids=C, r_k=r_k, eta_inf=float(r_k.max()),
                   B_n=bottleneck_of(ref, n_cuts=n_cuts, metric=metric), metric=metric)

    def __call__(self, theta):
        nn = (_cd(theta, self.centroids, self.metric) + self.r_k[None, :]).min(axis=1)
        return self.eta_inf + np.maximum(self.B_n, nn)


# --------------------------------------------------------------- whitening --

class Whitener:
    """Affine bijection f(z) = A (z - mu) with ||f(z)||^2 = d_Mahalanobis(z, mu)^2.

    A = gamma^-1/2 I + V diag(lam_i^-1/2 - gamma^-1/2) V', so A acts as
    lam_i^-1/2 along each retained eigenvector and gamma^-1/2 elsewhere. Both A
    and its inverse are available in closed form and applied low-rank, so no
    d x d matrix is ever built:

        A^-1 = gamma^1/2 I + V diag(lam_i^1/2 - gamma^1/2) V'

    Being affine and invertible, projecting in whitened coordinates and mapping
    back is exactly the Mahalanobis projection in the original space.
    """

    def __init__(self, mean, V, eig, gamma):
        self.mean = np.asarray(mean, dtype=float)
        self.V = np.asarray(V, dtype=float)
        self.eig = np.asarray(eig, dtype=float)
        self.gamma = float(gamma)
        self._fwd_iso = 1.0 / np.sqrt(self.gamma)
        self._fwd_sub = 1.0 / np.sqrt(self.eig) - self._fwd_iso
        self._inv_iso = np.sqrt(self.gamma)
        self._inv_sub = np.sqrt(self.eig) - self._inv_iso

    def __call__(self, Z):
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        diff = Z - self.mean
        return diff * self._fwd_iso + (diff @ self.V * self._fwd_sub) @ self.V.T

    def inverse(self, W):
        W = np.atleast_2d(np.asarray(W, dtype=float))
        z = W * self._inv_iso + (W @ self.V * self._inv_sub) @ self.V.T
        return z + self.mean


def whitening_map(ref, m=32, gamma=None):
    """Whitener for the reference sample: ||f(z)-f(z')|| = d_Mahalanobis(z, z').

    ecf.tex proves that s_(d,1) under the MAHALANOBIS metric is what links the
    order-1 Wasserstein set to the order-1 inverse-ECF set (via Jensen). Raw
    Euclidean distance in CLIP space is dominated by the leading "this is a
    nature prompt" directions, which is exactly the variance the ECF divides
    out; whitening first puts every Wasserstein score on that same footing and
    makes the two families comparable.

    Uses the same probabilistic-PCA covariance as ecf.py (top-m eigenvalues,
    isotropic tail gamma), fitted on `ref` only.
    """
    import torch
    from ecf import fit_regularized_covariance

    R = torch.as_tensor(np.asarray(ref), dtype=torch.float64)
    mean, V, eig_reg, eig_inv, g = fit_regularized_covariance(R, m=m, gamma=gamma)
    r = V.shape[1]
    return Whitener(mean.numpy(), V.numpy(), eig_reg.numpy()[:r], g)
