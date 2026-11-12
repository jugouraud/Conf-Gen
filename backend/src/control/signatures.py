"""Fixed-probe effects and existing RMS distance/anchor aggregation semantics."""

import numpy as np
import torch

from backend.src.control.contracts import Context, Decision, Detector, ProbeBank
from backend.src.control.model import TrajectoryModel
from backend.src.detection.calibration import aggregate_anchor_distances
from backend.src.fields.distances import calculate_probe_deviation_matrix
from backend.src.repair.steering import SteeringSchedule


def evaluate_signature(
    model: TrajectoryModel, probes: ProbeBank, c: Context, a: str
) -> np.ndarray:
    model.assert_frozen()
    if model.revision != probes.model_revision:
        raise ValueError("Probe/model revision mismatch")
    values = []
    for x, t in zip(probes.states, probes.times, strict=True):
        drift = np.asarray(model.drift(x.copy(), float(t), c, a))
        baseline = np.asarray(model.baseline(x.copy(), float(t)))
        if drift.shape != x.shape or baseline.shape != x.shape:
            raise ValueError("Model field shape mismatch")
        values.append(drift - baseline)
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite signature")
    model.assert_frozen()
    return result


def signature_distances(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    """rho = Frobenius norm / sqrt(J), using the repository RMS kernel."""
    query = np.asarray(query, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    if (
        query.ndim < 2
        or references.ndim != query.ndim + 1
        or references.shape[1:] != query.shape
    ):
        raise ValueError("Mismatched signature dimensions")
    # Translate both operands by the query before the shared Gram kernel. This
    # preserves rho and avoids cancellation at identical/nearly identical fields.
    q = torch.zeros((1, len(query), query[0].size), dtype=torch.float64)
    r = torch.tensor((references - query).reshape(len(references), len(query), -1))
    return calculate_probe_deviation_matrix(q, r)[0].numpy()


def score_signature(signature: np.ndarray, detector: Detector) -> Decision:
    method = {
        "nearest": "euclidean",
        "mean": "anchor_static",
        "exponential": "anchor_adaptive",
    }
    spec = {"method": method[detector.aggregation], "bandwidth": detector.bandwidth}
    scores = [
        float(
            aggregate_anchor_distances(
                signature_distances(signature, np.stack([a.signature for a in bank])),
                spec,
            )
        )
        for bank in (detector.safe, detector.unsafe)
    ]
    return Decision(
        scores[0],
        scores[1],
        scores[0] > detector.r_safe,
        scores[1] <= detector.r_unsafe,
    )


def mix_signatures(
    nominal: np.ndarray,
    anchor: np.ndarray,
    times: np.ndarray,
    schedule: SteeringSchedule,
) -> np.ndarray:
    if nominal.shape != anchor.shape or len(times) != len(nominal):
        raise ValueError("Mismatched signature mixing inputs")
    strength = np.array([schedule(float(t)) for t in times]).reshape(
        (-1,) + (1,) * (nominal.ndim - 1)
    )
    return (1 - strength) * nominal + strength * anchor
