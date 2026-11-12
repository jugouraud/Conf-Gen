"""Compare independently validated procedural and optimizer datasets on matched scenes."""

from pathlib import Path

from backend.src.expert.audit import run_dataset_validation
from backend.src.expert.storage import read_record


def run_generation_comparison(procedural: Path, optimization: Path) -> dict:
    """Input: complete dataset roots; output: metrics with identical scene inputs."""
    reports = {}
    scene_sets = []
    for method, directory in (("procedural", procedural), ("optimization", optimization)):
        run_dataset_validation(directory)
        manifest = read_record(directory / "frozen.json")
        summary = read_record(directory / "summary.json")
        configured = manifest["config"]["planner"].get("generation_method", "optimization")
        if configured != method:
            raise ValueError(f"Expected {method} generation in {directory}")
        scene_sets.append([
            (row["raw_observation"], row["group_id"], row["split"])
            for row in manifest["scenes"]
        ])
        metrics = summary["metrics"]
        reports[method] = {
            "validated_targets": metrics["validated_targets"],
            "attempt_acceptance_rate": metrics["attempt_acceptance_rate"],
            "pair_acceptance_rate": metrics["pair_acceptance_rate"],
            "worker_seconds_sum": metrics["worker_seconds_sum"],
            "mean_clearance": metrics["mean_clearance"],
            "minimum_clearance": metrics["minimum_clearance"],
            "mean_pairwise_reference_rms": metrics["mean_pairwise_reference_rms"],
        }
    if scene_sets[0] != scene_sets[1]:
        raise ValueError("Benchmark scenarios or split assignments differ")
    return {"matched_scenes": len(scene_sets[0]), "results": reports}
