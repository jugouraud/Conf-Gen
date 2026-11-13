"""Generate a fresh expert dataset, train its vector field, then launch the browser."""

import argparse
import json
from pathlib import Path

from backend.experiments.expert_trajectories.pipeline import run_dataset_generation
from backend.experiments.trajectory_flow.pipeline import run_trajectory_flow_pipeline
from frontend.expert_browser import SPLITS
from frontend.expert_browser import main as run_expert_browser


def run_generation_training_visualization(
    expert_config: Path,
    flow_config: Path,
    dataset_output: Path,
    training_output: Path,
    *,
    workers: int | None = None,
    split: str = "train",
    port: int = 8081,
    launch_browser: bool = True,
) -> dict:
    """Compose generation, vector-field training, and optional local visualization."""
    if split not in SPLITS:
        raise ValueError("Unknown visualization split")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Visualization port must be in 1..65535")
    dataset = run_dataset_generation(expert_config, dataset_output, workers=workers)
    if dataset["status"] != "complete":
        raise ValueError("Dataset has no training targets; vector-field training skipped")
    training = run_trajectory_flow_pipeline(dataset_output, flow_config, training_output)
    report = {"dataset": dataset, "training": training}
    print(json.dumps(report, indent=2), flush=True)
    if launch_browser:
        result = run_expert_browser([
            "--dataset", str(dataset_output), "--split", split,
            "--checkpoint", str(training_output), "--sha256", training["sha256"],
            "--port", str(port),
        ])
        if result:
            raise RuntimeError("Trajectory browser exited unsuccessfully")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert-config", type=Path, default=Path(__file__).with_name("expert_training.toml")
    )
    parser.add_argument(
        "--flow-config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    parser.add_argument(
        "--dataset-output", type=Path, default=Path(".cache/silverstone-trajs-50k")
    )
    parser.add_argument(
        "--training-output", type=Path,
        default=Path(".cache/trained-silverstone-50k-48-256"),
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    run_generation_training_visualization(
        args.expert_config, args.flow_config, args.dataset_output, args.training_output,
        workers=args.workers, split=args.split, port=args.port,
        launch_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
