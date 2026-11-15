"""Run the three-phase Silverstone flow experiment and optionally visualize Phase 3."""

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


def _generate_dataset(config: Path, output: Path, workers: int | None) -> dict:
    """Generate a fresh dataset; full audit remains an explicit optional command."""
    generation = run_dataset_generation(config, output, workers=workers)
    if generation["status"] != "complete":
        raise ValueError("Dataset has no training targets; vector-field training skipped")
    return generation


def run_three_phase_experiment(
    phase1_expert_config: Path,
    phase1_flow_config: Path,
    phase2_flow_config: Path,
    phase3_expert_config: Path,
    phase3_flow_config: Path,
    phase1_dataset_output: Path,
    phase1_training_output: Path,
    phase2_training_output: Path,
    phase3_dataset_output: Path,
    phase3_training_output: Path,
    *,
    workers: int | None = None,
    split: str = "train",
    port: int = 8081,
    launch_browser: bool = True,
) -> dict:
    """Run fresh FM, verified mixed-loss transfer, then verified d* transfer."""
    if split not in SPLITS:
        raise ValueError("Unknown visualization split")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Visualization port must be in 1..65535")

    phase1_dataset = _generate_dataset(
        phase1_expert_config, phase1_dataset_output, workers
    )
    phase1_training = run_trajectory_flow_pipeline(
        phase1_dataset_output, phase1_flow_config, phase1_training_output
    )
    phase2_training = run_trajectory_flow_pipeline(
        phase1_dataset_output, phase2_flow_config, phase2_training_output,
        initial_checkpoint=phase1_training_output,
        initial_sha256=phase1_training["sha256"],
    )
    phase3_dataset = _generate_dataset(
        phase3_expert_config, phase3_dataset_output, workers
    )
    phase3_training = run_trajectory_flow_pipeline(
        phase3_dataset_output, phase3_flow_config, phase3_training_output,
        initial_checkpoint=phase2_training_output,
        initial_sha256=phase2_training["sha256"],
        replay_dataset=phase1_dataset_output, replay_fraction=0.2,
    )
    report = {
        "phase1": {"dataset": phase1_dataset, "training": phase1_training},
        "phase2": {"training": phase2_training},
        "phase3": {"dataset": phase3_dataset, "training": phase3_training},
    }
    print(json.dumps(report, indent=2), flush=True)
    if launch_browser:
        result = run_expert_browser([
            "--dataset", str(phase3_dataset_output), "--split", split,
            "--checkpoint", str(phase3_training_output),
            "--sha256", phase3_training["sha256"], "--port", str(port),
        ])
        if result:
            raise RuntimeError("Trajectory browser exited unsuccessfully")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-expert-config", type=Path,
        default=Path(__file__).with_name("expert_training.toml"),
    )
    parser.add_argument(
        "--phase1-flow-config", type=Path,
        default=Path(__file__).with_name("phase1_fm.toml"),
    )
    parser.add_argument(
        "--phase2-flow-config", type=Path,
        default=Path(__file__).with_name("phase2_mixed.toml"),
    )
    parser.add_argument(
        "--phase3-expert-config", type=Path,
        default=Path(__file__).with_name("expert_phase3_dstar.toml"),
    )
    parser.add_argument(
        "--phase3-flow-config", type=Path,
        default=Path(__file__).with_name("phase3_dstar.toml"),
    )
    parser.add_argument(
        "--phase1-dataset-output", type=Path,
        default=Path(".cache/three-phase-centerline-nominal-20k"),
    )
    parser.add_argument("--phase1-training-output", type=Path, default=Path(".cache/three-phase-1-fm-nominal-20k"))
    parser.add_argument("--phase2-training-output", type=Path, default=Path(".cache/three-phase-2-mixed-nominal-20k"))
    parser.add_argument("--phase3-dataset-output", type=Path, default=Path(".cache/three-phase-dstar-nominal-48k"))
    parser.add_argument("--phase3-training-output", type=Path, default=Path(".cache/three-phase-3-dstar-nominal-48k"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    run_three_phase_experiment(
        args.phase1_expert_config, args.phase1_flow_config, args.phase2_flow_config,
        args.phase3_expert_config, args.phase3_flow_config,
        args.phase1_dataset_output, args.phase1_training_output,
        args.phase2_training_output, args.phase3_dataset_output, args.phase3_training_output,
        workers=args.workers, split=args.split, port=args.port,
        launch_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
