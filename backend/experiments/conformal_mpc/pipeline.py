"""Reproducible synthetic conformal diffusion-MPC experiment entry point."""

import argparse
import json
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from backend.experiments.conformal_mpc.evaluate import run_simulation_evaluation
from backend.experiments.conformal_mpc.freeze import run_policy_freezing
from backend.experiments.conformal_mpc.prepare import run_simulation_preparation
from backend.experiments.conformal_mpc.report import run_simulation_reporting
from backend.src.control.calibrate import run_detector_calibration
from backend.src.control.coverage import run_output_calibration
from backend.src.control.probes import run_probe_construction


def run_simulation_pipeline(config: Path, output: Path) -> dict:
    experiment = run_simulation_preparation(config)
    cfg = experiment.config
    probes = run_probe_construction(
        experiment.model,
        experiment.groups,
        count=cfg["probe_count"],
        steps=cfg["probe_steps"],
        seed=cfg["seed"],
    )
    detector, calibration = run_detector_calibration(
        experiment.model,
        probes,
        experiment.groups,
        alpha_safe=cfg["alpha_safe"],
        alpha_unsafe=cfg["alpha_unsafe"],
        aggregation=cfg["aggregation"],
        bandwidth=cfg["bandwidth"],
    )
    policy_id = run_policy_freezing(experiment, detector, calibration, output)
    calibration_results = run_simulation_evaluation(
        experiment, detector, split="output_calibration"
    )
    coverage = {}
    forbidden = [
        row.context.episode
        for split in ("reference", "detector_calibration", "test")
        for row in experiment.groups[split]
    ]
    for mode, result in calibration_results.items():
        coverage[mode] = {}
        for target in ("reference", "executed"):
            coverage[mode][target] = run_output_calibration(
                {e["episode"]: e[target + "_score"] for e in result["episodes"]},
                policy_id=policy_id + ":" + mode,
                target=target,
                alpha=cfg["alpha_output"],
                forbidden_episodes=forbidden,
            )
    test_results = run_simulation_evaluation(experiment, detector, split="test")
    return run_simulation_reporting(
        output, policy_id, calibration_results, test_results, coverage
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.toml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Resource bounds are fixed before numerical work, including native BLAS.
    from backend.src.common.storage import read_toml_config

    threads = read_toml_config(args.config)["threads"]
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("threads must be in [1,32]")
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        report = run_simulation_pipeline(args.config, args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
