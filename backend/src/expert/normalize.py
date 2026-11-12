"""Training-only moments; raw observations and expert targets stay unchanged."""

from pathlib import Path

import numpy as np

from backend.src.expert.config import ExpertConfig
from backend.src.expert.storage import publish_record


def observation_vector(row: dict, config: ExpertConfig) -> np.ndarray:
    observation = row["observation"]
    state = observation["initial_state"]
    features = (
        state
        if config.vehicle.kind == "velocity_disc"
        else [*state[:2], np.sin(state[2]), np.cos(state[2]), *state[3:]]
    )
    return np.asarray(
        [*features, *observation["goal"], *observation["initial_velocity"]]
    )


def training_statistics(targets: list[dict], config: ExpertConfig) -> dict:
    training = [row for row in targets if row["split"] == "train"]
    if not training:
        return {
            "status": "no_training_targets",
            "fit_split": "train",
            "target_ids": [],
            "groups": [],
            "statistics": {},
        }
    fields = {
        "observation_vector": np.stack(
            [observation_vector(row, config) for row in training]
        ),
        "y_ref_star": np.concatenate(
            [np.asarray(row["y_ref_star"]) for row in training]
        ),
        "controls": np.concatenate([np.asarray(row["controls"]) for row in training]),
    }
    statistics = {}
    for name, values in fields.items():
        mean, standard_deviation = values.mean(axis=0), values.std(axis=0)
        statistics[name] = {
            "mean": mean.tolist(),
            "scale": np.maximum(
                standard_deviation, config.dataset.normalization_epsilon
            ).tolist(),
            "constant_dimensions": (
                standard_deviation < config.dataset.normalization_epsilon
            ).tolist(),
            "count": len(values),
        }
    return {
        "status": "fitted",
        "fit_split": "train",
        "target_ids": [row["target_id"] for row in training],
        "groups": sorted({row["group_id"] for row in training}),
        "statistics": statistics,
        "observation_order": "state (heading as sin,cos for bicycle), goal x,y, initial velocity x,y",
        "untransformed": [
            "obstacle trajectories",
            "timestamps",
            "map bounds",
            "IDs",
            "guidance",
        ],
        "ddof": 0,
    }


def run_training_normalization(
    targets: list[dict], config: ExpertConfig, output: Path
) -> dict:
    record = training_statistics(targets, config)
    publish_record(output / "normalization.json", record)
    return record
