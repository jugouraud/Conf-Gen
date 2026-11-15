"""Three-phase orchestration preserves verified training handoffs."""

from pathlib import Path

import pytest

from backend.experiments.trajectory_flow import orchestrator

pytestmark = pytest.mark.unit


def test_three_phase_orchestrator_chains_weight_receipts_without_full_dataset_audits(
    tmp_path, monkeypatch
):
    generated, trained = [], []

    def generate(config: Path, output: Path, *, workers: int | None):
        generated.append((config, output, workers))
        return {"status": "complete"}

    def train(dataset: Path, config: Path, output: Path, **handoff):
        trained.append((dataset, config, output, handoff))
        return {"sha256": f"sha-{len(trained)}", "best_validation_mse": 0.1,
                "checkpoint": str(output / "weights.pt"), "validation": {}}

    monkeypatch.setattr(orchestrator, "run_dataset_generation", generate)
    monkeypatch.setattr(orchestrator, "run_trajectory_flow_pipeline", train)
    outputs = [tmp_path / name for name in (
        "phase1-data", "phase1-model", "phase2-model", "phase3-data", "phase3-model"
    )]
    report = orchestrator.run_three_phase_experiment(
        *(tmp_path / f"config-{index}.toml" for index in range(5)), *outputs,
        workers=2, launch_browser=False,
    )

    assert [entry[1] for entry in generated] == [outputs[0], outputs[3]]
    assert trained[0][0:3] == (outputs[0], tmp_path / "config-1.toml", outputs[1])
    assert trained[1][3] == {
        "initial_checkpoint": outputs[1], "initial_sha256": "sha-1"
    }
    assert trained[2][3] == {
        "initial_checkpoint": outputs[2], "initial_sha256": "sha-2"
    }
    assert report["phase3"]["training"]["sha256"] == "sha-3"
