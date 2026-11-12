"""Freeze contextual reference banks and class-conditional detector quantiles."""

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from backend.src.control.contracts import Anchor, Detector, Example, ProbeBank
from backend.src.control.model import TrajectoryModel
from backend.src.control.partition import (
    require_independent_examples,
    validate_partitions,
)
from backend.src.control.signatures import evaluate_signature, score_signature
from backend.src.detection.calibration import calculate_conformal_cutoff


def run_detector_calibration(
    model: TrajectoryModel,
    probes: ProbeBank,
    groups: Mapping[str, Sequence[Example]],
    *,
    alpha_safe: float,
    alpha_unsafe: float,
    aggregation: str = "nearest",
    bandwidth: float = 1.0,
) -> tuple[Detector, dict]:
    """Labels must come from an independent validator with frozen catalogue rules.

    No unsafe cap: exact requested upper class quantiles, including +infinity.
    All scoring parameters are fixed before detector-calibration episodes.
    """
    membership = validate_partitions(groups)
    if not 0 < alpha_safe < 1 or not 0 < alpha_unsafe < 1:
        raise ValueError("Both class alphas must lie in (0,1)")
    if not set(probes.episodes).issubset(membership["reference"]):
        raise ValueError("Probe fitting leaked outside reference episodes")
    require_independent_examples(groups["detector_calibration"])
    banks: dict[bool, list[Anchor]] = {True: [], False: []}
    for row in groups["reference"]:
        c = model.encode(row.context)
        banks[row.safe].append(
            Anchor(
                c.episode,
                row.guidance,
                c.compatibility,
                evaluate_signature(model, probes, c, row.guidance),
            )
        )
    detector = Detector(
        probes,
        tuple(banks[True]),
        tuple(banks[False]),
        np.inf,
        np.inf,
        aggregation,
        bandwidth,
    )
    scores: dict[bool, list[float]] = {True: [], False: []}
    for row in groups["detector_calibration"]:
        signature = evaluate_signature(
            model, probes, model.encode(row.context), row.guidance
        )
        decision = score_signature(signature, detector)
        scores[row.safe].append(decision.d_safe if row.safe else decision.d_unsafe)
    # At n_l=0 the requested rank is 1>n_l, so the class radius is +inf.
    # Keep the existing finite/nonempty quantile helper's contract unchanged.
    safe = (
        calculate_conformal_cutoff(scores[True], alpha_safe) if scores[True] else np.inf
    )
    unsafe = (
        calculate_conformal_cutoff(scores[False], alpha_unsafe)
        if scores[False]
        else np.inf
    )
    return replace(detector, r_safe=safe, r_unsafe=unsafe), {
        "safe_scores": scores[True],
        "unsafe_scores": scores[False],
        "alpha_safe": alpha_safe,
        "alpha_unsafe": alpha_unsafe,
        "r_safe": safe,
        "r_unsafe": unsafe,
        "partitions": membership,
        "aggregation": aggregation,
        "bandwidth": bandwidth,
    }
