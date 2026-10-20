"""Memory-bounded Euclidean matrices with float64 Gram accumulation."""

import torch


def euclidean_matrix(queries, references, *, reference_squared_norms=None):
    """Float64 avoids cancellation/TF32 errors for long field signatures."""
    if (queries.ndim != 2 or references.ndim != 2 or
            queries.shape[1] != references.shape[1] or len(references) == 0):
        raise ValueError("Expected two matching feature matrices and nonempty references")
    q, r = queries.double(), references.double()
    norms = (torch.linalg.vector_norm(r, dim=1).square() if reference_squared_norms is None
             else reference_squared_norms)
    squared = torch.linalg.vector_norm(q, dim=1).square()[:, None] + norms[None, :] - 2 * (q @ r.T)
    distances = squared.clamp_min_(0).sqrt_()
    if not torch.isfinite(distances).all():
        raise ValueError("Non-finite Euclidean distance matrix")
    return distances
