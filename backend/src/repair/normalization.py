"""Safe-validation-only temporal deviation normalization shared by repair runtimes."""

import numpy as np

from backend.src.repair.contracts import RepairExperiment
from backend.src.repair.steering import calculate_per_time_field_distances


def fit_safe_deviation_normalization(experiment: RepairExperiment) -> dict:
    """Fit per-time latent L2 mean/std using ordered safe-validation rows only."""
    limit = experiment.settings["experiment"]["normalization_limit"]
    if type(limit) is not int or limit < 1:
        raise ValueError("normalization_limit must be a positive integer")
    rows = experiment.groups["safe_validation"][:limit]
    if not rows:
        raise ValueError("Normalization requires safe-validation prompts")
    records = []
    for index, row in enumerate(rows):
        target_index = int(
            np.argmin(experiment.distances["safe__safe_validation"][index])
        )
        anchor = experiment.groups["safe_reference"][target_index]
        query = experiment.features.load_signature("safe_validation", row)
        reference = experiment.features.load_signature("safe_reference", anchor)
        records.append(
            calculate_per_time_field_distances(
                query, reference, experiment.bank
            ).numpy()
        )
    values = np.concatenate(records, axis=1)
    if not np.isfinite(values).all():
        raise ValueError("Normalization requires finite field distances")
    location = values.mean(axis=1)
    scale = np.maximum(values.std(axis=1), np.maximum(1e-8, np.abs(location) * 1e-6))
    return {
        "location": location.tolist(),
        "scale": scale.tolist(),
        "fit_split": "safe_validation",
        "fit_ids": [row["id"] for row in rows],
        "definition": "per-time safe-calibration to nearest safe-reference latent norm mean/std",
    }
