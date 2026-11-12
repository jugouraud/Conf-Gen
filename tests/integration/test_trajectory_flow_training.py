"""Real expert JSONL -> learned weights -> frozen adapter -> sampled reference."""

from pathlib import Path

import numpy as np
import pytest

from backend.src.control.model import TorchTrajectoryAdapter
from backend.src.trajectory_flow.data import run_training_preparation
from backend.src.trajectory_flow.model import FlowArchitecture
from backend.src.trajectory_flow.sample import run_flow_sampling
from backend.src.trajectory_flow.train import (
    TrainingSettings,
    load_flow_checkpoint,
    run_flow_training,
)

pytestmark = pytest.mark.integration


def test_train_on_checked_in_expert_targets_and_replay_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[2]
    dataset = root / "backend/experiments/expert_trajectories/example_dataset"
    prepared = run_training_preparation(dataset, seed=13, validation_fraction=0.2)
    assert len(prepared.fit) >= 2
    assert len(prepared.validation) >= 1
    assert not set(prepared.fit_groups) & set(prepared.validation_groups)
    settings = TrainingSettings(epochs=3, batch_size=4, patience=3, seed=13, threads=1)
    trained = run_flow_training(
        prepared,
        FlowArchitecture(prepared.horizon, prepared.dt, width=32, depth=1, context_width=16),
        settings,
        tmp_path / "flow_run",
    )
    checkpoint = tmp_path / "flow_run"
    assert (checkpoint / "weights.pt").is_file()
    assert (checkpoint / "metadata.json").is_file()
    loaded = load_flow_checkpoint(checkpoint, expected_sha256=trained["weights_sha256"])
    sample = run_flow_sampling(
        loaded, prepared.validation[0].context, prepared.validation[0].guidance,
        seed=77, steps=4, solver="euler",
    )
    assert sample.reference.shape == (prepared.horizon + 1, 2)
    assert np.isfinite(sample.reference).all()
    adapter = TorchTrajectoryAdapter(loaded, loaded.revision)
    probe = np.zeros(2 * (prepared.horizon + 1))
    current = prepared.validation[0]
    signature = adapter.drift(probe, 0.5, current.context, current.guidance) - adapter.baseline(probe, 0.5)
    assert signature.shape == probe.shape
    assert np.isfinite(signature).all()
    with pytest.raises(ValueError, match="identity mismatch"):
        load_flow_checkpoint(checkpoint, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="frozen"):
        loaded.train()
        loaded.assert_frozen()
