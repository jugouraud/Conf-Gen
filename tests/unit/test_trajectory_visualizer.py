"""Physical-scene contract and frozen-reference visualization checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from backend.src.expert.scenes import Scene, TimedObstacle
from backend.src.expert.storage import publish_record, read_record
from backend.src.trajectory_flow.model import FlowArchitecture, TrajectoryFlowModel
from backend.src.trajectory_flow.sample import run_flow_sampling
from frontend.visualization import (
    ReferenceOverlay,
    context_from_scene,
    read_scene_catalog,
    render_scene_svg,
    sample_references,
)

pytestmark = pytest.mark.unit


def _moving_scene() -> Scene:
    return Scene(
        scene_id="demo", group_id="group-a", map_id="rectangular-map",
        bounds=(-1, 4, -2, 2), start_time=10.0, initial_state=(0.0, 0.0),
        initial_velocity=(0.0, 0.0), goal=(3.0, 0.0),
        obstacles=(TimedObstacle(
            "moving-1", np.array([10.0, 11.5, 12.0]),
            np.array([[1.5, 0.0], [1.5, 0.7], [1.5, 0.0]]), 0.2,
        ),),
    )


def test_catalog_reads_ready_observations_without_expert_labels(tmp_path):
    root = Path(__file__).resolve().parents[2]
    original = read_record(root / "backend/experiments/expert_trajectories/example_dataset/frozen.json")
    (tmp_path / "frozen.json").write_text(json.dumps(original))
    catalog = read_scene_catalog(tmp_path)
    assert catalog.split == "train"
    assert catalog.footprint_radius == 0.15
    assert catalog.scenes
    ready_ids = {row["scene_id"] for row in original["scenes"]
                 if row["split"] == "train" and row["status"] == "ready"}
    assert {scene.scene_id for scene in catalog.scenes} == ready_ids
    assert not (tmp_path / "splits").exists()  # No dependence on y_ref_star labels.

    original["config"]["vehicle"]["kind"] = "bicycle"
    publish_record(tmp_path / "frozen.json", original)
    with pytest.raises(ValueError, match="velocity_disc"):
        read_scene_catalog(tmp_path)


def test_context_obstacle_predictions_match_physical_time():
    scene = _moving_scene()
    context = context_from_scene(scene, horizon=2, dt=1.0)
    np.testing.assert_allclose(context.obstacles[0].centers,
                               scene.obstacles[0].at(np.array([10.0, 11.0, 12.0])))
    assert context.state == scene.initial_state
    assert context.features == (*scene.initial_velocity, *scene.bounds)
    assert context.horizon == 2 and context.dt == 1.0


def test_reference_generation_uses_frozen_field_and_paired_seeds():
    import torch

    scene = _moving_scene()
    torch.manual_seed(4)
    model = TrajectoryFlowModel(
        FlowArchitecture(2, 1.0, width=32, depth=1, context_width=16),
        {"left": {"side": 1, "speed_scale": 1.0, "obstacle_index": 0},
         "right": {"side": -1, "speed_scale": 1.0, "obstacle_index": 0}},
    )
    model.freeze("visualizer-test")
    overlays = sample_references(
        model, scene, ("left", "right"), seed=12, samples_per_guidance=2,
        steps=4, solver="heun", footprint_radius=0.15,
    )
    assert [(row.guidance, row.seed) for row in overlays] == [
        ("left", 12), ("right", 12), ("left", 13), ("right", 13),
    ]
    assert all(row.positions.shape == (3, 2) for row in overlays)
    assert all(np.isfinite(row.positions).all() for row in overlays)
    expected = run_flow_sampling(
        model, context_from_scene(scene, 2, 1.0), "left", seed=12,
        steps=4, solver="heun",
    )
    np.testing.assert_array_equal(overlays[0].positions, expected.reference)
    assert all(row.obstacle_clearance is not None for row in overlays)
    with pytest.raises(ValueError, match="limits"):
        sample_references(model, scene, ("left", "right"), seed=12,
                          samples_per_guidance=4, steps=4, solver="heun",
                          footprint_radius=0.15)


def test_fixed_track_svg_draws_the_actual_circuit():
    from backend.src.expert.track import TrackSpec, track_for

    spec = TrackSpec(layout="silverstone", radii=(32.0, 36.0),
                     left_width=0.7, right_width=0.7)
    circuit = track_for(spec)
    start, goal = circuit.point(np.array([0.0, 10.0]))
    scene = Scene("silverstone", "g", "fixed-silverstone-v1",
                  (-33, 33, -37, 37), 0.0, tuple(start), (0, 0), tuple(goal),
                  (), track=spec)
    svg = render_scene_svg(scene, (), footprint_radius=0.15,
                           index=0, horizon=2, dt=1.0)
    assert 'data-kind="track-asphalt"' in svg
    assert 'data-kind="track-outer"' in svg
    assert 'data-kind="track-inner"' in svg
    assert 'data-kind="track-centerline"' in svg
    assert svg.count('fill-rule="evenodd"') == 1


def test_svg_renders_moving_obstacle_at_selected_time_and_unclipped_reference():
    scene = _moving_scene()
    reference = ReferenceOverlay(
        "left", 5, np.array([[0.0, 0.0], [1.5, 2.5], [3.0, 0.0]]),
        -0.05, -0.15, 0.1,
    )
    svg = render_scene_svg(
        scene, (reference,), footprint_radius=0.15, index=1, horizon=2, dt=1.0,
    )
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert 'cy="-0.46667"' in svg  # Interpolated obstacle, not a static circle.
    assert "1.50000,-2.50000" in svg  # Raw out-of-bounds reference stays visible.
    assert "START" in svg and "GOAL" in svg
    assert "<script" not in svg
    with pytest.raises(ValueError, match="index"):
        render_scene_svg(scene, (reference,), footprint_radius=0.15,
                         index=3, horizon=2, dt=1.0)
