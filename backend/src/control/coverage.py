"""Separate marginal clearance-score calibration, never a repair objective."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def upper_score_quantile(scores: Sequence[float], alpha: float) -> float:
    """Upper finite-sample rank with ties and +inf (failed outputs) retained."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or np.isnan(values).any() or not 0 < alpha < 1:
        raise ValueError("Expected scores without NaN and 0<alpha<1")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else math.inf


@dataclass(frozen=True)
class CoverageCalibration:
    policy_id: str
    target: str
    episodes: tuple[str, ...]
    alpha: float
    cutoff: float

    def coverage(
        self, scores: Mapping[str, float], *, policy_id: str, target: str
    ) -> float:
        if policy_id != self.policy_id or target != self.target:
            raise ValueError("Policy or clearance target changed after calibration")
        if set(scores) & set(self.episodes):
            raise ValueError("Coverage evaluation reused calibration episodes")
        values = np.asarray(list(scores.values()), dtype=float)
        if not len(values) or np.isnan(values).any():
            raise ValueError("Coverage needs valid held-out scores")
        return float(np.mean(values <= self.cutoff))


def run_output_calibration(
    scores: Mapping[str, float],
    *,
    policy_id: str,
    target: str,
    alpha: float,
    forbidden_episodes: Sequence[str],
) -> CoverageCalibration:
    """One prespecified episode-level score per fresh independent episode.

    Reference and actual executed rollout scores must be supplied separately.
    Abstention without a reference has R=+inf; it is not silently discarded.
    """
    if not policy_id or not scores or target not in {"reference", "executed"}:
        raise ValueError(
            "Frozen policy identity and explicit clearance target required"
        )
    if set(scores) & set(forbidden_episodes):
        raise ValueError("Output calibration is not fresh")
    cutoff = upper_score_quantile(list(scores.values()), alpha)
    return CoverageCalibration(policy_id, target, tuple(scores), alpha, cutoff)
