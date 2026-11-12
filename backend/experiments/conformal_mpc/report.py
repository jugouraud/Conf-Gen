"""Publish complete calibration evidence and paired held-out measurements."""

import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
import torch

from backend.experiments.conformal_mpc.freeze import json_values
from backend.src.common.storage import calculate_file_sha256, write_json
from backend.src.control.coverage import CoverageCalibration


def run_simulation_reporting(
    output: Path,
    policy_id: str,
    calibration_results: dict,
    test_results: dict,
    coverage: dict[str, dict[str, CoverageCalibration]],
) -> dict:
    summary = {}
    for mode, targets in coverage.items():
        metrics = dict(test_results[mode]["metrics"])
        metrics["coverage"] = {}
        for target, fit in targets.items():
            scores = {
                e["episode"]: e[target + "_score"]
                for e in test_results[mode]["episodes"]
            }
            metrics["coverage"][target] = {
                **asdict(fit),
                "marginal_score_coverage": fit.coverage(
                    scores, policy_id=policy_id + ":" + mode, target=target
                ),
                "test_scores": scores,
                "vacuous": bool(np.isinf(fit.cutoff)),
            }
        summary[mode] = metrics
    write_json(
        output / "output_calibration.json",
        json_values(
            {
                "policy_id": policy_id,
                "fits": {
                    mode: {target: asdict(fit) for target, fit in targets.items()}
                    for mode, targets in coverage.items()
                },
                "episodes": calibration_results,
            }
        ),
    )
    write_json(output / "evaluation.json", json_values(test_results))
    report = json_values(
        {
            "schema": "conformal-mpc-v1",
            "status": "complete",
            "policy_id": policy_id,
            "synthetic": True,
            "summary": summary,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
            },
            "artifacts": {
                name: calculate_file_sha256(output / name)
                for name in (
                    "frozen_policy.json",
                    "output_calibration.json",
                    "evaluation.json",
                )
            },
            "limitations": [
                "Synthetic analytic field, no trained trajectory checkpoint or hardware validation.",
                "Marginal episode score coverage requires exchangeability and a fixed policy.",
                "No conditional safety after release, closed-loop invariance or joint safe-FPR alpha bound.",
                "Deadlines are cooperative; native calls are not preempted.",
                "Disc footprints, piecewise-linear predictions and bounded uncertainty are assumptions.",
                "A no-reference episode receives +inf reference score; infinite cutoffs are vacuous.",
                "Nominal still uses geometric and MPC release gates; nominal flags are diagnostic only.",
            ],
        }
    )
    write_json(output / "report.json", report)
    return report
