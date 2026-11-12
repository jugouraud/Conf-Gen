"""Select distinct validated maneuvers and publish raw target records and indices."""

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from backend.src.common.storage import calculate_file_sha256
from backend.src.expert.config import SPLITS, ExpertConfig, Guidance
from backend.src.expert.problem import expert_cost
from backend.src.expert.scenes import Scene
from backend.src.expert.storage import publish_jsonl, publish_record
from backend.src.expert.validate import run_expert_validation


def run_expert_export(manifest: dict, attempts: list[dict], output: Path) -> list[dict]:
    config = ExpertConfig.from_record(manifest["config"])
    candidates = defaultdict(list)
    outcomes = []
    for attempt in attempts:
        job, result = attempt["job"], attempt["result"]
        outcome = {
            "attempt_id": job["attempt_id"],
            "status": result["status"],
            "selected": False,
            "eligible": False,
            "validation": result.get("validation"),
        }
        outcomes.append(outcome)
        if not result["eligible"]:
            continue
        scene, guidance = (
            Scene.from_record(job["observation"]),
            Guidance(**job["guidance"]),
        )
        validation = run_expert_validation(
            scene,
            guidance,
            config,
            result["states"],
            result["controls"],
            result["timestamps"],
            result["y_ref_star"],
        )
        outcome["validation"] = validation
        if not validation["valid"] or (
            config.planner.require_convergence and not result["solver_success"]
        ):
            outcome["status"] = "export_validation_failed"
            continue
        outcome["eligible"] = True
        cost = expert_cost(
            scene, np.asarray(result["controls"]), np.asarray(result["states"]), config
        )
        candidates[(job["scene_index"], job["guidance_index"])].append(
            (cost, job["attempt_id"], attempt, validation, outcome)
        )
    targets, scene_paths = [], defaultdict(list)
    for key in sorted(candidates):
        _, _, attempt, validation, outcome = min(
            candidates[key], key=lambda entry: entry[:2]
        )
        job, result = attempt["job"], attempt["result"]
        path = np.asarray(result["y_ref_star"])[:, :2]
        if any(
            np.sqrt(np.mean(np.sum((path - old) ** 2, axis=1)))
            < config.dataset.minimum_diversity
            for old in scene_paths[job["scene_index"]]
        ):
            outcome["selection_reason"] = "duplicate_maneuver"
            continue
        scene_paths[job["scene_index"]].append(path)
        outcome.update(selected=True, selection_reason="minimum_cost_validated_guess")
        target = {
            "schema": "expert-target-v1",
            "dataset_id": manifest["dataset_id"],
            "target_id": job["attempt_id"],
            "scene_id": job["observation"]["scene_id"],
            "group_id": job["group_id"],
            "split": job["split"],
            "observation": job["observation"],
            "guidance": job["guidance"],
            "y_ref_star": result["y_ref_star"],
            "states": result["states"],
            "controls": result["controls"],
            "timestamps": result["timestamps"],
            "vehicle": asdict(config.vehicle),
            "planner": asdict(config.planner),
            "solver": {
                name: result.get(name)
                for name in (
                    "status",
                    "solver_success",
                    "solver_code",
                    "iterations",
                    "cost",
                    "runtime_seconds",
                    "worker_seconds",
                )
            },
            "validation": validation,
            "seed": job["seed"],
            "input_id": attempt["input_id"],
            "usage": "expert_training_target"
            if job["split"] == "train"
            else "reserved_expert_label_not_conformal_score",
        }
        publish_record(output / "targets" / (job["attempt_id"] + ".json"), target)
        targets.append(target)
    publish_jsonl(output / "dataset.jsonl", targets)
    for split in SPLITS:
        publish_jsonl(
            output / "splits" / (split + ".jsonl"),
            [row for row in targets if row["split"] == split],
        )
    files = [
        "dataset.jsonl",
        *(f"splits/{split}.jsonl" for split in SPLITS),
        *(f"targets/{target['target_id']}.json" for target in targets),
        *(f"attempts/{attempt['job']['attempt_id']}.json" for attempt in attempts),
    ]
    publish_record(
        output / "index.json",
        {
            "dataset_id": manifest["dataset_id"],
            "target_ids": [target["target_id"] for target in targets],
            "outcomes": outcomes,
            "rejected_scenes": [
                row for row in manifest["scenes"] if row["status"] == "rejected"
            ],
            "files": {name: calculate_file_sha256(output / name) for name in files},
        },
    )
    return targets
