"""Freeze deployed policy identity and publish calibration inputs atomically."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np

from backend.experiments.conformal_mpc.prepare import PreparedExperiment
from backend.src.common.storage import calculate_file_sha256, write_json
from backend.src.control.contracts import Detector


def json_values(value: object) -> object:
    """Strict JSON, with explicit infinity strings rather than nonstandard NaN."""
    if isinstance(value, np.ndarray):
        return json_values(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            raise ValueError("NaN artifact")
        return (
            "+inf" if value == np.inf else "-inf" if value == -np.inf else float(value)
        )
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run_policy_freezing(
    experiment: PreparedExperiment, detector: Detector, calibration: dict, output: Path
) -> str:
    """Changes to code, model, controller, schedules or inputs require new calibration."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; frozen evidence is immutable")
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "backend/src").rglob("*.py")) + sorted(
        Path(__file__).parent.glob("*.py")
    )
    manifest = json_values(
        {
            "schema": "conformal-mpc-v1",
            "status": "frozen_before_output_calibration",
            "config": experiment.config,
            "config_id": experiment.config_id,
            "model_revision": experiment.model.revision,
            "synthetic": True,
            "sources": {
                path.relative_to(root).as_posix(): calculate_file_sha256(path)
                for path in paths
            },
            "scenes": {
                name: [asdict(scene) for scene in scenes]
                for name, scenes in experiment.scenes.items()
            },
            "labels": {
                name: [
                    {
                        "episode": row.context.episode,
                        "safe": row.safe,
                        "guidance": row.guidance,
                    }
                    for row in rows
                ]
                for name, rows in experiment.groups.items()
            },
            "detector": asdict(detector),
            "detector_calibration": calibration,
            "schedules": [schedule.to_record() for schedule in experiment.schedules],
            "tracking": asdict(experiment.tracking),
            "sampling": asdict(experiment.sampling),
            "catalogue_rule": "all independently geometrically labelled reference episodes, stable order",
            "anchor_compatibility": "same disc-v1 latent/physical contract; field reevaluated in current context",
            "fallback": "validate and solve shifted old reference; otherwise abstain; simulator applies zero input",
            "numerics": "FP64 CPU; uniform probe RMS; Euler or Euler-Maruyama; piecewise-linear physical motion",
        }
    )
    policy_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    write_json(output / "frozen_policy.json", dict(manifest, policy_id=policy_id))
    return policy_id
