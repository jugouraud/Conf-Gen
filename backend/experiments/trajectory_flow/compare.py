"""Evaluate frozen trajectory-flow checkpoints on identical held-out scenes."""

import argparse
import json
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from backend.src.common.storage import write_json
from backend.src.expert.storage import read_record
from backend.src.trajectory_flow.data import run_training_preparation
from backend.src.trajectory_flow.evaluate import run_flow_evaluation
from backend.src.trajectory_flow.train import load_flow_checkpoint


def run_checkpoint_comparison(
    dataset: Path,
    checkpoints: list[tuple[Path, str]],
    output: Path,
    *,
    seed: int = 20261112,
    steps: int = 32,
    solver: str = "heun",
) -> dict:
    """Compare paired sampler seeds on the same untouched training-group holdout.

    The checkpoint SHA values must come from trusted training receipts, not from
    the directories being evaluated. No model is fitted or recalibrated here.
    """
    if not 2 <= len(checkpoints) <= 6 or len({p.resolve() for p, _ in checkpoints}) != len(checkpoints):
        raise ValueError("Compare 2..6 distinct checkpoints")
    if output.exists():
        raise ValueError("Comparison output already exists")
    if solver not in {"euler", "heun"} or type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("Invalid sampler settings")
    prepared = run_training_preparation(dataset, seed=seed, validation_fraction=0.2)
    results = []
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for directory, trusted_sha in checkpoints:
            model = load_flow_checkpoint(directory, expected_sha256=trusted_sha)
            if (model.architecture.horizon, model.architecture.dt) != (prepared.horizon, prepared.dt):
                raise ValueError("Checkpoint physical horizon/dt differs from dataset")
            if any(row.track != model.architecture.track for row in prepared.validation):
                raise ValueError("Checkpoint track differs from evaluation dataset")
            if any(row.guidance not in model.guidance for row in prepared.validation):
                raise ValueError("Checkpoint guidance differs from evaluation dataset")
            metadata = read_record(directory / "metadata.json")
            if metadata["dataset_id"] == prepared.dataset_id and (
                tuple(metadata["validation_groups"]) != prepared.validation_groups
                or set(metadata["fit_groups"]) & set(prepared.validation_groups)
            ):
                raise ValueError("Checkpoint's validation partition differs or overlaps fitting")
            replay = metadata.get("replay_provenance")
            if replay is not None and replay["dataset_id"] == prepared.dataset_id and (
                set(replay["fit_groups"]) & set(prepared.validation_groups)
            ):
                raise ValueError("Replay fitting groups overlap evaluated groups")
            evaluation = run_flow_evaluation(
                model, prepared, seed=seed + 1000, steps=steps, solver=solver,
                footprint_radius=prepared.footprint_radius,
            )
            results.append({
                "checkpoint": str(directory), "sha256": trusted_sha,
                "model_revision": model.revision, "validation": evaluation,
            })
    report = {
        "dataset_id": prepared.dataset_id,
        "manifest_sha256": prepared.manifest_sha256,
        "validation_groups": len(prepared.validation_groups),
        "validation_examples": len(prepared.validation),
        "seed": seed, "solver": solver, "steps": steps,
        "checkpoints": results,
        "not_a_safety_certificate": True,
    }
    write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", nargs=2, action="append", required=True,
                        metavar=("DIRECTORY", "TRUSTED_SHA256"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--solver", choices=("heun", "euler"), default="heun")
    args = parser.parse_args(argv)
    print(json.dumps(run_checkpoint_comparison(
        args.dataset, [(Path(directory), sha) for directory, sha in args.checkpoint],
        args.output, seed=args.seed, steps=args.steps, solver=args.solver,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
