"""Fixed expert dataset browsing with verified multi-path overlays."""

import json
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pytest

from backend.experiments.expert_trajectories.pipeline import read_configuration
from backend.experiments.expert_trajectories.scenes import run_scene_sampling
from backend.src.common.storage import calculate_file_sha256
from backend.src.expert.scenes import Scene
from backend.src.expert.storage import publish_record
from frontend.expert_browser import (
    load_prediction_model, read_expert_catalog, render_expert_svg, sample_predicted_path,
)

pytestmark = pytest.mark.unit
CONFIG = Path(__file__).resolve().parents[2] / "backend/experiments/expert_trajectories/config.toml"


def test_browser_renders_fixed_track_and_multiple_checked_paths(tmp_path):
    config, settings = read_configuration(CONFIG)
    rows, _ = run_scene_sampling(settings, config)
    dataset_id = "test-dataset"
    publish_record(tmp_path / "frozen.json", {
        "schema": "expert-dataset-v1", "dataset_id": dataset_id,
        "config": config.to_record(),
    })
    ids, files = [], {}
    (tmp_path / "targets").mkdir()
    for index, row in enumerate(rows[:2]):
        scene = Scene.from_record(row)
        name = f"s{index:04d}-g0-i0-r0"
        ids.append(name)
        # A synthetic validated-label receipt tests the browser contract only.
        path = np.linspace(scene.initial_state[:2], scene.goal,
                           config.planner.horizon + 1)
        destination = tmp_path / "targets" / f"{name}.json"
        publish_record(destination, {
            "schema": "expert-target-v1", "dataset_id": dataset_id,
            "target_id": name, "split": "train", "observation": row,
            "validation": {"valid": True}, "y_ref_star": path.tolist(),
            "guidance": asdict(config.guidance[0]),
        })
        files[f"targets/{name}.json"] = calculate_file_sha256(destination)
    publish_record(tmp_path / "index.json", {
        "dataset_id": dataset_id, "target_ids": ids, "files": files,
    })
    catalog = read_expert_catalog(tmp_path)
    assert len(catalog.paths) == 2
    first = render_expert_svg(catalog, (0,))
    both = render_expert_svg(catalog, (0, 1))
    assert catalog.track.layout == "silverstone"
    assert 'viewBox="' in first
    assert first.split('viewBox="')[1].split('"')[0] == both.split('viewBox="')[1].split('"')[0]
    assert both.count('<polyline') == 2
    assert both.count('data-kind="expert"') == 2
    assert "#38bdf8" in both and "#f472b6" in both
    assert both.count('<circle') >= 8  # two scene obstacles + path endpoints
    assert both.count('data-kind="obstacle"') == 2
    assert first.count('data-kind="obstacle"') == 1
    # Browsing an unchecked trajectory shows its obstacle, not the first one's.
    focused = render_expert_svg(catalog, (), focus_index=1)
    assert focused.count('data-kind="obstacle"') == 1
    marker = rows[1]["obstacles"][0]["centers"][0]
    assert f'cx="{marker[0]:.4f}" cy="{-marker[1]:.4f}"' in focused
    assert f'cx="{marker[0]:.4f}" cy="{-marker[1]:.4f}"' in both
    with pytest.raises(ValueError, match="out of range"):
        render_expert_svg(catalog, (999,))
    target = json.loads((tmp_path / "targets" / f"{ids[0]}.json").read_text())
    target["validation"]["valid"] = False
    publish_record(tmp_path / "targets" / f"{ids[0]}.json", target)
    with pytest.raises(ValueError, match="digest"):
        read_expert_catalog(tmp_path)
    # A correctly rehashed target with a moving obstacle still violates this viewer's contract.
    target["validation"]["valid"] = True
    target["observation"]["obstacles"][0]["centers"][-1][0] += 0.2
    publish_record(tmp_path / "targets" / f"{ids[0]}.json", target)
    index = json.loads((tmp_path / "index.json").read_text())
    index["files"][f"targets/{ids[0]}.json"] = calculate_file_sha256(
        tmp_path / "targets" / f"{ids[0]}.json"
    )
    publish_record(tmp_path / "index.json", index)
    with pytest.raises(ValueError, match="moving obstacles"):
        read_expert_catalog(tmp_path)


def test_predictions_share_the_expert_scene_and_have_distinct_dashed_overlays(tmp_path, monkeypatch):
    import torch

    from backend.src.trajectory_flow.model import FlowArchitecture, TrajectoryFlowModel
    from backend.src.trajectory_flow.sample import run_flow_sampling
    from frontend.visualization import context_from_scene

    config, settings = read_configuration(CONFIG)
    rows, _ = run_scene_sampling(settings, config)
    row = rows[0]
    scene = Scene.from_record(row)
    dataset_id = "test-dataset"
    publish_record(tmp_path / "frozen.json", {
        "schema": "expert-dataset-v1", "dataset_id": dataset_id,
        "config": config.to_record(),
    })
    target_id = "s0000-g0-i0-r0"
    path = np.linspace(scene.initial_state[:2], scene.goal, config.planner.horizon + 1)
    (tmp_path / "targets").mkdir()
    destination = tmp_path / "targets" / f"{target_id}.json"
    publish_record(destination, {
        "schema": "expert-target-v1", "dataset_id": dataset_id,
        "target_id": target_id, "split": "train", "observation": row,
        "validation": {"valid": True}, "guidance": asdict(config.guidance[0]),
        "y_ref_star": path.tolist(),
    })
    publish_record(tmp_path / "index.json", {
        "dataset_id": dataset_id, "target_ids": [target_id],
        "files": {f"targets/{target_id}.json": calculate_file_sha256(destination)},
    })
    catalog = read_expert_catalog(tmp_path)
    torch.manual_seed(3)
    guidance = config.guidance[0]
    model = TrajectoryFlowModel(
        FlowArchitecture(catalog.horizon, catalog.dt, width=32, depth=1, context_width=16),
        {guidance.name: {"side": guidance.side, "speed_scale": guidance.speed_scale,
                         "obstacle_index": guidance.obstacle_index}},
    )
    model.freeze("browser-test")
    prediction = sample_predicted_path(model, catalog, 0)
    expected = run_flow_sampling(
        model, context_from_scene(scene, catalog.horizon, catalog.dt),
        guidance.name, seed=42, steps=32, solver="heun",
    )
    np.testing.assert_array_equal(prediction, expected.reference)
    svg = render_expert_svg(catalog, (0,), {0: prediction})
    assert svg.count('data-kind="expert"') == 1
    assert svg.count('data-kind="predicted"') == 1
    assert 'stroke-dasharray="0.12 0.08"' in svg
    assert render_expert_svg(catalog, (), {0: prediction}).count('<polyline') == 1
    assert render_expert_svg(catalog, (0,), {}).count('data-kind="predicted"') == 0
    assert svg.split('viewBox="')[1].split('"')[0] == render_expert_svg(catalog, (0,)).split('viewBox="')[1].split('"')[0]
    with pytest.raises(ValueError, match="out of range"):
        sample_predicted_path(model, catalog, 1)
    with pytest.raises(ValueError, match="predicted trajectory"):
        render_expert_svg(catalog, (), {0: np.ones((2, 2))})
    with pytest.raises(ValueError, match="predicted trajectory"):
        render_expert_svg(catalog, (), {0: np.full_like(prediction, np.nan)})

    # The checkpoint wrapper validates dataset identity and guidance before plotting.
    from backend.src.trajectory_flow import train as train_module
    monkeypatch.setattr(train_module, "load_flow_checkpoint", lambda *args, **kwargs: model)
    (tmp_path / "metadata.json").write_text(json.dumps({"dataset_id": "wrong"}))
    with pytest.raises(ValueError, match="different expert dataset"):
        load_prediction_model(tmp_path, "0" * 64, catalog)
    (tmp_path / "metadata.json").write_text(json.dumps({"dataset_id": dataset_id}))
    assert load_prediction_model(tmp_path, "0" * 64, catalog) is model
