"""Freeze scene groups, configuration and source identity before maneuver jobs."""

import platform
from pathlib import Path

import numpy as np
import scipy

from backend.src.common.storage import calculate_file_sha256
from backend.src.expert.audit import run_dataset_validation
from backend.src.expert.config import ExpertConfig
from backend.src.expert.scenes import Scene, assign_groups
from backend.src.expert.storage import digest, publish_record, read_record
from backend.src.expert.validate import validate_initial_scene


def run_expert_preparation(
    config: ExpertConfig,
    raw_scenes: list[dict],
    output: Path,
    *,
    resume: bool,
    scene_source: dict | None = None,
) -> dict:
    if not isinstance(raw_scenes, list) or not 4 <= len(raw_scenes) <= 2000:
        raise ValueError("Dataset needs 4..2000 scene records")
    maximum_attempts = (
        len(raw_scenes)
        * len(config.guidance)
        * len(config.planner.initial_guess_offsets)
        * (1 + config.planner.retries)
    )
    if maximum_attempts > 10000:
        raise ValueError("Dataset exceeds 10,000 bounded attempts")
    records, scene_ids, group_ids = [], set(), []
    for index, raw in enumerate(raw_scenes):
        try:
            scene = Scene.from_record(raw)
        except (ValueError, TypeError, KeyError):
            records.append(
                {
                    "index": index,
                    "raw_observation": raw,
                    "status": "rejected",
                    "scene_id": None,
                    "group_id": None,
                    "split": None,
                    "reasons": ["scene_schema"],
                }
            )
            continue
        if scene.scene_id in scene_ids:
            raise ValueError(
                "Duplicate scene ID; related variants need distinct scene IDs and one group ID"
            )
        scene_ids.add(scene.scene_id)
        group_ids.append(scene.group_id)
        reasons = validate_initial_scene(scene, config)
        records.append(
            {
                "index": index,
                "raw_observation": raw,
                "scene_id": scene.scene_id,
                "group_id": scene.group_id,
                "status": "rejected" if reasons else "ready",
                "reasons": reasons,
            }
        )
    assignment = assign_groups(group_ids, config.dataset)
    for row in records:
        row["split"] = assignment.get(row["group_id"])
    root = Path(__file__).resolve().parents[3]
    source_files = sorted(Path(__file__).parent.glob("*.py")) + sorted(
        (root / "backend/experiments/expert_trajectories").glob("*.py")
    )
    source_files += [
        root / name
        for name in (
            "backend/src/mpc/geometry.py",
            "backend/src/mpc/dynamics.py",
            "backend/src/common/storage.py",
            "pyproject.toml",
            "uv.lock",
        )
    ]
    science = config.to_record()
    science["execution"].pop("workers")
    identity = {
        "schema": "expert-dataset-v1",
        "scientific_config": science,
        "scenes": records,
        "group_splits": assignment,
        "scene_source": scene_source or {"kind": "provided_records"},
        "sources": {
            path.relative_to(root).as_posix(): calculate_file_sha256(path)
            for path in source_files
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "semantics": "physical seconds/metres/radians; FP64; expert labels are not deployed-output conformal scores",
    }
    dataset_id = digest(identity)
    path = output / "frozen.json"
    manifest = dict(identity, dataset_id=dataset_id, config=config.to_record())
    if path.exists():
        if not resume:
            raise ValueError("Dataset exists; use --resume or a fresh output")
        previous = read_record(path)
        if (
            previous["dataset_id"] != dataset_id
            or digest(
                {k: v for k, v in previous.items() if k not in {"dataset_id", "config"}}
            )
            != dataset_id
        ):
            raise ValueError("Frozen dataset source/config/scenes changed")
        status_path = output / "status.json"
        if status_path.exists() and read_record(status_path)["state"] in {
            "complete",
            "no_training_targets",
        }:
            run_dataset_validation(output)
        return previous
    if resume:
        raise ValueError("Cannot resume without a frozen manifest")
    if any(item.name != ".writer.lock" for item in output.iterdir()):
        raise ValueError("Use an empty output directory")
    publish_record(path, manifest)
    return manifest
