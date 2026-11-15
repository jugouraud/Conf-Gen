"""Phase 3 replay draws only prior training groups, preserving held-out sets."""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.experiments.trajectory_flow import pipeline
from backend.src.control.contracts import Context
from backend.src.trajectory_flow.data import FlowExample, PreparedTraining

pytestmark = pytest.mark.unit


def _prepared(dataset_id: str, count: int) -> PreparedTraining:
    def make_rows(prefix: str, size: int):
        return tuple(FlowExample(
            Context(f"{prefix}-{i}", 0, (0., 0.), (1., 0.), (), 0.4, 2,
                    features=(0., 0., -2., 2., -2., 2.)),
            "nominal", np.zeros((3, 2)), f"{prefix}-{i}", None)
            for i in range(size))
    return PreparedTraining(
        make_rows("fit", count), make_rows("valid", 2),
        {"nominal": {"side": 0, "speed_scale": 1., "obstacle_index": 0}},
        dataset_id, f"sha-{dataset_id}",
        tuple(f"fit-{i}" for i in range(count)), ("valid-0", "valid-1"),
        2, 0.4, 0.15,
    )


def test_replay_is_20_percent_and_has_no_validation_groups(tmp_path, monkeypatch):
    primary = _prepared("primary", 8)
    replay = _prepared("replay", 10)
    prepared = {tmp_path / "primary": primary, tmp_path / "replay": replay}
    selected = []
    monkeypatch.setattr(pipeline, "read_configuration", lambda path: (
        SimpleNamespace(track=None),
        SimpleNamespace(seed=123, validation_fraction=0.2, require_matched_triplets=False,
                        threads=1), {"solver": "heun", "steps": 4},
    ))
    monkeypatch.setattr(pipeline, "run_training_preparation", lambda path, **kw: prepared[path])
    monkeypatch.setattr(pipeline, "run_flow_training", lambda rows, *a, **kw:
                        selected.append(rows) or {
                            "model": object(), "weights_sha256": "a" * 64,
                            "best_validation_mse": 0.1,
                        })
    monkeypatch.setattr(pipeline, "run_flow_evaluation", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "write_json", lambda *a: None)
    pipeline.run_trajectory_flow_pipeline(
        tmp_path / "primary", tmp_path / "config", tmp_path / "out",
        replay_dataset=tmp_path / "replay", replay_fraction=0.2,
    )
    training = selected[0]
    assert len(training.fit) == 10  # 8 primary + 2 selected replay rows
    assert training.validation == primary.validation
    assert training.replay_provenance["fit_targets"] == 2
    assert all(row.group_id.startswith("replay:") and row.context.episode == row.group_id
               for row in training.fit[len(primary.fit):])
    assert not set(training.validation_groups) & set(training.fit_groups)
    assert all("valid-" not in row.group_id for row in training.fit[len(primary.fit):])


def test_replay_requires_distinct_dataset_and_positive_fraction(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Replay requires"):
        pipeline.run_trajectory_flow_pipeline(
            tmp_path, tmp_path, tmp_path, replay_fraction=0.2,
        )
