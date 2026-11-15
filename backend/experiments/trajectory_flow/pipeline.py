"""Train a new trajectory-space flow from the existing expert-trajectory dataset."""

import argparse
import json
import tomllib
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from backend.src.common.storage import write_json
from backend.src.trajectory_flow.data import run_training_preparation
from backend.src.trajectory_flow.evaluate import run_flow_evaluation
from backend.src.trajectory_flow.model import FlowArchitecture
from backend.src.trajectory_flow.train import TrainingSettings, run_flow_training


def read_configuration(path: Path) -> tuple[FlowArchitecture, TrainingSettings, dict]:
    with path.open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("Training config exceeds 128 KiB")
    config = tomllib.loads(raw.decode("utf-8"))
    if set(config) != {"model", "training", "evaluation"}:
        raise ValueError("Expected model, training and evaluation sections")
    for name, cls in (("model", FlowArchitecture), ("training", TrainingSettings)):
        if set(config[name]) != {f.name for f in fields(cls)}:
            raise ValueError(f"Missing/unknown {name} settings")
    eval_config = config["evaluation"]
    if set(eval_config) != {"solver", "steps"} or eval_config["solver"] not in {"euler", "heun"}:
        raise ValueError("Invalid evaluation solver configuration")
    if type(eval_config["steps"]) is not int or not 1 <= eval_config["steps"] <= 10000:
        raise ValueError("Invalid evaluation steps")
    return FlowArchitecture(**config["model"]), TrainingSettings(**config["training"]), eval_config


def run_trajectory_flow_pipeline(
    dataset: Path, config: Path, output: Path, *,
    initial_checkpoint: Path | None = None, initial_sha256: str | None = None,
    replay_dataset: Path | None = None, replay_fraction: float = 0.0,
) -> dict:
    """Compose public stage entry points; optionally replay prior fitting groups."""
    if (type(replay_fraction) not in (int, float) or not np.isfinite(replay_fraction)
            or not 0 <= replay_fraction < 0.5
            or (replay_dataset is None) != (replay_fraction == 0)):
        raise ValueError("Replay requires a separate dataset and fraction in (0, 0.5)")
    architecture, settings, evaluation = read_configuration(config)
    prepared = run_training_preparation(
        dataset, seed=settings.seed, validation_fraction=settings.validation_fraction,
        require_matched_triplets=settings.require_matched_triplets,
    )
    if replay_dataset is not None:
        replay = run_training_preparation(
            replay_dataset, seed=settings.seed,
            validation_fraction=settings.validation_fraction,
            require_matched_triplets=settings.require_matched_triplets,
        )
        if (replay.dataset_id == prepared.dataset_id
                or replay.guidance != prepared.guidance
                or (replay.horizon, replay.dt, replay.footprint_radius)
                != (prepared.horizon, prepared.dt, prepared.footprint_radius)
                or any(row.track != architecture.track for row in replay.fit)):
            raise ValueError("Replay dataset has incompatible geometry, guidance or identity")
        count = round(len(prepared.fit) * replay_fraction / (1 - replay_fraction))
        if count < 1 or count > len(replay.fit):
            raise ValueError("Insufficient independent replay fitting targets")
        indices = np.random.default_rng(settings.seed + 17).choice(
            len(replay.fit), size=count, replace=False,
        )
        selected = []
        for index in indices:
            row = replay.fit[int(index)]
            group = f"{replay.dataset_id}:{row.group_id}"
            selected.append(replace(row, group_id=group,
                                    context=replace(row.context, episode=group)))
        selected = tuple(selected)
        replay_groups = tuple(sorted({row.group_id for row in selected}))
        source_groups = tuple(sorted({replay.fit[int(i)].group_id for i in indices}))
        prepared = replace(
            prepared,
            fit=prepared.fit + selected,
            fit_groups=tuple(sorted((*prepared.fit_groups, *replay_groups))),
            replay_provenance={
                "dataset_id": replay.dataset_id,
                "manifest_sha256": replay.manifest_sha256,
                "fit_groups": source_groups,
                "fit_targets": len(selected),
                "requested_fraction": replay_fraction,
            },
        )
    with threadpool_limits(limits=settings.threads):
        trained = run_flow_training(
            prepared, architecture, settings, output,
            initial_checkpoint=initial_checkpoint, initial_sha256=initial_sha256,
        )
        model = trained.pop("model")
        report = run_flow_evaluation(
            model, prepared, seed=settings.seed + 1000,
            steps=evaluation["steps"], solver=evaluation["solver"],
            footprint_radius=prepared.footprint_radius,
        )
    write_json(output / "evaluation.json", report)
    return {
        "checkpoint": str(output / "weights.pt"),
        "sha256": trained["weights_sha256"],
        "best_validation_mse": trained["best_validation_mse"],
        "validation": report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-sha256", type=str)
    parser.add_argument("--replay-dataset", type=Path)
    parser.add_argument("--replay-fraction", type=float, default=0.0)
    args = parser.parse_args(argv)
    print(json.dumps(run_trajectory_flow_pipeline(
        args.dataset, args.config, args.output,
        initial_checkpoint=args.initial_checkpoint, initial_sha256=args.initial_sha256,
        replay_dataset=args.replay_dataset, replay_fraction=args.replay_fraction,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
