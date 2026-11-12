"""All four policies, real constrained MPC and atomic calibration/report artifacts."""

import json
from pathlib import Path

import pytest
from threadpoolctl import threadpool_limits

from backend.experiments.conformal_mpc.pipeline import run_simulation_pipeline
from backend.src.common.storage import calculate_file_sha256

pytestmark = pytest.mark.e2e


def test_complete_synthetic_experiment_publishes_separate_calibration_and_paired_evaluation(
    tmp_path,
):
    source = Path("backend/experiments/conformal_mpc/config.toml").read_text()
    for old, new in (
        ("reference_episodes = 48", "reference_episodes = 18"),
        ("detector_episodes = 48", "detector_episodes = 18"),
        ("output_episodes = 20", "output_episodes = 3"),
        ("test_episodes = 12", "test_episodes = 3"),
        ("physical_steps = 24", "physical_steps = 2"),
        ("probe_count = 12", "probe_count = 3"),
    ):
        source = source.replace(old, new)
    config = tmp_path / "config.toml"
    config.write_text(source)
    output = tmp_path / "run"
    with threadpool_limits(limits=1):
        report = run_simulation_pipeline(config, output)
    assert report["status"] == "complete" and report["synthetic"]
    for name, digest in report["artifacts"].items():
        assert calculate_file_sha256(output / name) == digest
    frozen = json.loads((output / "frozen_policy.json").read_text())
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert set(evaluation) == {"nominal", "safe_only", "joint", "repaired"}
    paired = [
        [(e["episode"], e["seed"]) for e in result["episodes"]]
        for result in evaluation.values()
    ]
    assert all(rows == paired[0] for rows in paired)
    for mode, metrics in report["summary"].items():
        assert set(metrics["coverage"]) == {"reference", "executed"}
        assert (
            metrics["coverage"]["reference"]["policy_id"]
            == frozen["policy_id"] + ":" + mode
        )
        assert metrics["coverage"]["executed"]["marginal_score_coverage"] >= 0
    with pytest.raises(ValueError, match="fresh output"):
        run_simulation_pipeline(config, output)
