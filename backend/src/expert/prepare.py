"""Freeze scene groups, configuration and source identity before maneuver jobs."""

import platform
from pathlib import Path

import numpy as np
import scipy

from backend.src.common.storage import calculate_file_sha256
from backend.src.expert.config import ExpertConfig
from backend.src.expert.scenes import Scene, assign_groups
from backend.src.expert.storage import (
    MAX_EXPERT_TRAJECTORIES,
    OPERATIONAL_SOURCE_FILES,
    digest,
    publish_record,
    read_record,
)
from backend.src.expert.validate import validate_initial_scene


def _resumable_operational_source_change(previous: dict, identity: dict) -> bool:
    """Allow export/I-O limit fixes without relabeling completed trajectories."""
    previous_identity = {
        key: value for key, value in previous.items() if key not in {"config", "dataset_id", "sources"}
    }
    current_identity = {key: value for key, value in identity.items() if key != "sources"}
    changed = {
        name
        for name, checksum in identity["sources"].items()
        if previous["sources"].get(name) != checksum
    }
    return (
        digest(previous_identity) == digest(current_identity)
        and changed <= OPERATIONAL_SOURCE_FILES
    )


def run_expert_preparation(
    config: ExpertConfig,
    raw_scenes: list[dict],
    output: Path,
    *,
    resume: bool,
    scene_source: dict | None = None,
) -> dict:
    if not isinstance(raw_scenes, list) or not 4 <= len(raw_scenes) <= MAX_EXPERT_TRAJECTORIES:
        raise ValueError(f"Dataset needs 4..{MAX_EXPERT_TRAJECTORIES} scene records")
    candidate_count = (
        config.planner.procedural_candidates
        if config.planner.generation_method == "procedural"
        else len(config.planner.initial_guess_offsets)
    )
    maximum_attempts = (
        len(raw_scenes)
        * len(config.guidance)
        * candidate_count
        * (1 + config.planner.retries)
    )
    if maximum_attempts > MAX_EXPERT_TRAJECTORIES:
        raise ValueError(f"Dataset exceeds {MAX_EXPERT_TRAJECTORIES:,} bounded attempts")
    records, scene_ids, group_ids, tracks = [], set(), [], set()
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
        tracks.add(scene.track)
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
    if config.planner.generation_method == "procedural" and (
        len(tracks) != 1 or None in tracks
    ):
        raise ValueError("Procedural datasets require one identical fixed track")
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
            if not _resumable_operational_source_change(previous, identity):
                raise ValueError("Frozen dataset source/config/scenes changed")
        return previous
    if resume:
        raise ValueError("Cannot resume without a frozen manifest")
    if any(item.name != ".writer.lock" for item in output.iterdir()):
        raise ValueError("Use an empty output directory")
    publish_record(path, manifest)
    return manifest
