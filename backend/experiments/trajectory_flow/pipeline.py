"""Train a new trajectory-space flow from the existing expert-trajectory dataset."""

import argparse
import json
import tomllib
from dataclasses import fields
from pathlib import Path

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


def run_trajectory_flow_pipeline(dataset: Path, config: Path, output: Path) -> dict:
    """Compose public stage entry points; no changes to detector or MPC stages."""
    architecture, settings, evaluation = read_configuration(config)
    prepared = run_training_preparation(
        dataset, seed=settings.seed, validation_fraction=settings.validation_fraction
    )
    with threadpool_limits(limits=settings.threads):
        trained = run_flow_training(prepared, architecture, settings, output)
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
    args = parser.parse_args(argv)
    print(json.dumps(run_trajectory_flow_pipeline(args.dataset, args.config, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
