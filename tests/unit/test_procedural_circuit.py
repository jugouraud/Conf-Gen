"""Fixed-circuit scene contracts and validated procedural alternatives."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from backend.experiments.expert_trajectories.pipeline import read_configuration
from backend.experiments.expert_trajectories.scenes import run_scene_sampling
from backend.src.expert.config import Guidance
from backend.src.expert.procedural import run_expert_procedural
from backend.src.expert.scenes import Scene
from backend.src.expert.track import TrackSpec, track_for
from backend.src.expert.validate import run_expert_validation, validate_initial_scene
from backend.src.trajectory_flow.data import example_from_target

pytestmark = pytest.mark.unit
CONFIG = Path(__file__).resolve().parents[2] / "backend/experiments/expert_trajectories/config.toml"


def test_track_is_closed_and_offset_boundaries_are_checked():
    track = track_for(TrackSpec())
    np.testing.assert_allclose(track.point(0), track.point(track.length), atol=1e-12)
    base = track.point(2.0)
    normal = track.normal(2.0)
    assert track.boundary_clearance(np.array([base]), 0.15).min() > 1.0
    assert track.boundary_clearance(np.array([base + 1.4 * normal]), 0.15).min() < 0
    with pytest.raises(ValueError, match="curvature"):
        TrackSpec(radii=(5.0, 2.0), left_width=1.0)


def test_fixed_scene_sampling_is_reproducible_and_grouped():
    config, settings = read_configuration(CONFIG)
    rows, source = run_scene_sampling(settings, config)
    repeat, _ = run_scene_sampling(settings, config)
    assert rows == repeat
    assert source["kind"] == "fixed_circuit"
    assert len({row["map_id"] for row in rows}) == 1
    assert len({str(row["track"]) for row in rows}) == 1
    assert len({tuple(row["initial_state"]) for row in rows}) > 1
    assert len({tuple(row["goal"]) for row in rows}) > 1
    assert len({tuple(row["obstacles"][0]["centers"][0]) for row in rows}) > 1
    assert len({row["group_id"] for row in rows}) == len(rows)
    for row in rows:
        scene = Scene.from_record(row)
        import json
        assert json.loads(json.dumps(scene.to_record())) == row
        assert not validate_initial_scene(scene, config)


def test_opposite_guidance_produces_independently_validated_reference():
    config, settings = read_configuration(CONFIG)
    settings = dict(settings, moving_fraction=0.0, invalid_initial_fraction=0.0)
    settings["obstacle_radius_range"] = [0.12, 0.12]
    settings["obstacle_lateral_range"] = [0.0, 0.0]
    rows, _ = run_scene_sampling(settings, config)
    # Several independently sampled attempts permit rejection without relabelling it.
    successes = []
    for row in rows:
        scene = Scene.from_record(row)
        alternatives = []
        for side in (1, -1):
            guidance = Guidance("left" if side == 1 else "right", side=side)
            candidates = [
                run_expert_procedural(scene, guidance, config, guess=j, seed=j + 10)
                for j in range(config.planner.procedural_candidates)
            ]
            accepted = next((r for r in candidates if r["eligible"]), None)
            if accepted is None:
                break
            assert run_expert_validation(
                scene, guidance, config, accepted["states"], accepted["controls"],
                accepted["timestamps"], accepted["y_ref_star"]
            )["valid"]
            alternatives.append(accepted)
        if len(alternatives) == 2:
            successes = alternatives
            break
    assert len(successes) == 2, "No feasible left/right pair in test scenes"
    left, right = (np.asarray(r["y_ref_star"]) for r in successes)
    assert np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1))) > 0.2

    row = rows[0]
    scene = Scene.from_record(row)
    guidance = Guidance("left", side=1)
    first = run_expert_procedural(scene, guidance, config, guess=0, seed=300)
    repeat = run_expert_procedural(scene, guidance, config, guess=0, seed=300)
    assert first["status"] == repeat["status"]
    assert first["y_ref_star"] == repeat["y_ref_star"]
    if first["eligible"]:
        target = dict(
            schema="expert-target-v1", split="train", dataset_id="test-dataset",
            usage="expert_training_target", validation={"valid": True},
            vehicle=asdict(config.vehicle), planner=asdict(config.planner),
            observation=row, scene_id=row["scene_id"], group_id=row["group_id"],
            guidance=asdict(guidance), timestamps=first["timestamps"],
            y_ref_star=first["y_ref_star"],
        )
        example = example_from_target(target, config.planner.horizon,
                                      config.planner.dt, "test-dataset")
        assert example.target.shape == (config.planner.horizon + 1, 2)
        assert len(example.context.features) == 6


def test_rejects_initial_overlap_and_track_exit():
    config, settings = read_configuration(CONFIG)
    rows, _ = run_scene_sampling(settings, config)
    row = rows[0]
    start = row["initial_state"][:2]
    bad = dict(row, obstacles=[dict(row["obstacles"][0],
                               centers=[start, *row["obstacles"][0]["centers"][1:]])])
    scene = Scene.from_record(bad)
    assert "initial_obstacle_clearance" in validate_initial_scene(scene, config)
    result = run_expert_procedural(scene, Guidance("left", 1), config, guess=0, seed=1)
    assert not result["eligible"]
    track = track_for(scene.track)
    outside = track.point(1.0) + 2 * track.normal(1.0)
    assert track.boundary_clearance(np.array([outside]), config.vehicle.footprint_radius).min() < 0


def test_moving_obstacle_predictions_have_safe_validated_alternatives():
    config, settings = read_configuration(CONFIG)
    settings = dict(settings, moving_fraction=1.0, invalid_initial_fraction=0.0)
    rows, _ = run_scene_sampling(settings, config)
    accepted = 0
    for row in rows:
        scene = Scene.from_record(row)
        obstacle = scene.obstacles[0]
        assert not np.array_equal(obstacle.centers[0], obstacle.centers[-1])
        for guidance in (Guidance("left", 1), Guidance("right", -1)):
            result = run_expert_procedural(scene, guidance, config, guess=0, seed=100)
            if result["eligible"]:
                accepted += 1
                assert result["validation"]["minimum_clearance"] >= config.planner.required_clearance
                assert result["validation"]["valid"]
    assert accepted > 0
