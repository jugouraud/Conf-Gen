"""Verified dataset metrics and final complete marker."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from backend.src.common.storage import calculate_file_sha256
from backend.src.expert.config import SPLITS
from backend.src.expert.storage import publish_record, read_record


def run_dataset_summary(
    manifest: dict,
    attempts: list[dict],
    targets: list[dict],
    normalization: dict,
    validation: dict,
    output: Path,
) -> dict:
    scene_targets = defaultdict(list)
    for target in targets:
        scene_targets[target["scene_id"]].append(target)
    distances = []
    for group in scene_targets.values():
        for i, first in enumerate(group):
            for second in group[i + 1 :]:
                delta = (
                    np.asarray(first["y_ref_star"])[:, :2]
                    - np.asarray(second["y_ref_star"])[:, :2]
                )
                distances.append(float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))))
    clearances = [
        row["validation"]["minimum_clearance"]
        for row in targets
        if row["validation"]["minimum_clearance"] is not None
    ]
    solver_times = [
        row["result"]["runtime_seconds"]
        for row in attempts
        if "runtime_seconds" in row["result"]
    ]
    worker_times = [row["result"]["worker_seconds"] for row in attempts]
    ready = sum(row["status"] == "ready" for row in manifest["scenes"])
    pairs = ready * len(manifest["config"]["guidance"])
    successful = sum(row["result"]["eligible"] for row in attempts)
    metrics = {
        "scenes": len(manifest["scenes"]),
        "ready_scenes": ready,
        "rejected_scenes": len(manifest["scenes"]) - ready,
        "attempts": len(attempts),
        "eligible_attempts": successful,
        "attempt_acceptance_rate": successful / len(attempts) if attempts else 0,
        "requested_scene_guidance_pairs": pairs,
        "validated_targets": len(targets),
        "pair_acceptance_rate": len(targets) / pairs if pairs else 0,
        "failure_counts": dict(
            Counter(
                row["result"]["status"]
                for row in attempts
                if not row["result"]["eligible"]
            )
        ),
        "split_targets": {
            split: sum(row["split"] == split for row in targets) for split in SPLITS
        },
        "split_groups": {
            split: sum(value == split for value in manifest["group_splits"].values())
            for split in SPLITS
        },
        "minimum_clearance": min(clearances, default=None),
        "mean_clearance": float(np.mean(clearances)) if clearances else None,
        "maximum_goal_error": max(
            (row["validation"]["goal_error"] for row in targets), default=None
        ),
        "mean_goal_error": float(
            np.mean([row["validation"]["goal_error"] for row in targets])
        )
        if targets
        else None,
        "maximum_constraint_residual": max(
            (row["validation"]["maximum_constraint_residual"] for row in targets),
            default=None,
        ),
        "scenes_with_multiple_maneuvers": sum(
            len(rows) > 1 for rows in scene_targets.values()
        ),
        "minimum_pairwise_reference_rms": min(distances, default=None),
        "mean_pairwise_reference_rms": float(np.mean(distances)) if distances else None,
        "guidance_counts": dict(Counter(row["guidance"]["name"] for row in targets)),
        "solver_seconds_sum": sum(solver_times),
        "worker_seconds_sum": sum(worker_times),
        "mean_solver_seconds": float(np.mean(solver_times)) if solver_times else None,
        "p95_worker_seconds": float(np.quantile(worker_times, 0.95))
        if worker_times
        else None,
    }
    report = {
        "schema": "expert-summary-v1",
        "dataset_id": manifest["dataset_id"],
        "status": "complete"
        if normalization["status"] == "fitted"
        else "no_training_targets",
        "validation": validation,
        "metrics": metrics,
        "generation_sessions": read_record(output / "generation_sessions.json"),
        "artifacts": {
            name: calculate_file_sha256(output / name)
            for name in (
                "frozen.json",
                "index.json",
                "normalization.json",
                "generation_sessions.json",
            )
        },
        "limitations": [
            "Procedural references are heuristic, independently validated candidates, not optimal plans.",
            "Local solver convergence is not global optimality; failure is not scene infeasibility.",
            "Targets are expert solutions, not deployed-generator conformal calibration samples.",
            "Circular footprints and model-specific intersample assumptions require physical validation.",
            "Numerical values depend on recorded runtime; wall-clock times and timeout outcomes depend on hardware/load.",
        ],
    }
    publish_record(output / "summary.json", report)
    publish_record(
        output / "status.json",
        {
            "state": report["status"],
            "dataset_id": manifest["dataset_id"],
            "summary_sha256": calculate_file_sha256(output / "summary.json"),
        },
    )
    return report
