"""Checkpoint cross-evaluation reuses held-out scenes and identical sampler seeds."""

from types import SimpleNamespace

import pytest

from backend.experiments.trajectory_flow import compare

pytestmark = pytest.mark.unit


def test_paired_checkpoint_comparison_uses_same_validation_and_seed(tmp_path, monkeypatch):
    prepared = SimpleNamespace(
        dataset_id="dataset", manifest_sha256="hash", horizon=96, dt=0.4,
        footprint_radius=0.15, validation_groups=("v",),
        validation=(SimpleNamespace(track={"layout": "test"}, guidance="nominal"),),
    )
    seen = []
    monkeypatch.setattr(compare, "run_training_preparation", lambda *a, **kw: prepared)
    monkeypatch.setattr(compare, "load_flow_checkpoint", lambda directory, **kw:
                        SimpleNamespace(architecture=SimpleNamespace(
                            horizon=96, dt=0.4, track={"layout": "test"}),
                            guidance={"nominal": {}}, revision=str(directory)))
    monkeypatch.setattr(compare, "read_record", lambda path: {
        "dataset_id": "dataset", "validation_groups": ["v"], "fit_groups": ["f"],
    })
    monkeypatch.setattr(compare, "run_flow_evaluation", lambda model, rows, **kw:
                        seen.append((rows, kw["seed"], kw["steps"])) or {"collision_fraction": 0})
    monkeypatch.setattr(compare, "write_json", lambda path, value: None)
    report = compare.run_checkpoint_comparison(
        tmp_path, [(tmp_path / "a", "a" * 64), (tmp_path / "b", "b" * 64)],
        tmp_path / "report.json",
    )
    assert len(report["checkpoints"]) == 2
    assert seen == [(prepared, 20262112, 32), (prepared, 20262112, 32)]


def test_comparison_rejects_validation_leakage(tmp_path, monkeypatch):
    prepared = SimpleNamespace(
        dataset_id="dataset", manifest_sha256="hash", horizon=96, dt=0.4,
        footprint_radius=0.15, validation_groups=("v",),
        validation=(SimpleNamespace(track=None, guidance="nominal"),),
    )
    monkeypatch.setattr(compare, "run_training_preparation", lambda *a, **kw: prepared)
    monkeypatch.setattr(compare, "load_flow_checkpoint", lambda *a, **kw:
                        SimpleNamespace(architecture=SimpleNamespace(horizon=96, dt=0.4, track=None),
                                        guidance={"nominal": {}}, revision="x"))
    monkeypatch.setattr(compare, "read_record", lambda path: {
        "dataset_id": "dataset", "validation_groups": ["v"], "fit_groups": ["v"],
    })
    with pytest.raises(ValueError, match="overlaps fitting"):
        compare.run_checkpoint_comparison(tmp_path,
            [(tmp_path / "a", "a" * 64), (tmp_path / "b", "b" * 64)], tmp_path / "report.json")
