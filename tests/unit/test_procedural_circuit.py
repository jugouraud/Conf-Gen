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
from backend.src.trajectory_flow.representation import SplineRepresentation

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


def test_silverstone_track_is_closed_projectable_and_has_nonfolding_boundaries():
    spec = TrackSpec(layout="silverstone", radii=(32.0, 36.0),
                     left_width=0.7, right_width=0.7)
    circuit = track_for(spec)
    assert 250 < circuit.length < 290
    np.testing.assert_allclose(circuit.point(0), circuit.point(circuit.length))
    np.testing.assert_allclose(circuit.point(-2), circuit.point(circuit.length - 2))
    arcs = np.linspace(0.0, circuit.length, 50, endpoint=False)
    points = circuit.point(arcs)
    projected, lateral = circuit.project(points)
    np.testing.assert_allclose(projected, arcs, atol=1e-8)
    np.testing.assert_allclose(lateral, 0.0, atol=1e-8)
    assert np.min(circuit.boundary_clearance(points[:1], 0.15)) > 0.5
    outside = points[:1] + 0.8 * circuit.normal(arcs[:1])
    assert np.min(circuit.boundary_clearance(outside, 0.15)) < 0
    with pytest.raises(ValueError, match="curvature"):
        track_for(TrackSpec(layout="silverstone", radii=(32.0, 36.0),
                            left_width=1.0, right_width=1.0))
    # Missing layout in older frozen artifacts must retain their ellipse meaning.
    assert TrackSpec(**{"radii": [5.0, 3.5]}).layout == "ellipse"


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
    assert all(row["start_time"] == 0.0 for row in rows)
    assert all(all(center == row["obstacles"][0]["centers"][0]
                   for center in row["obstacles"][0]["centers"]) for row in rows)
    assert len({row["group_id"] for row in rows}) == len(rows)
    track = track_for(TrackSpec(**settings["track"]))
    forward_distances = []
    start_offsets, goal_offsets = [], []
    wrapped = 0
    for row in rows:
        scene = Scene.from_record(row)
        import json
        assert json.loads(json.dumps(scene.to_record())) == row
        assert not validate_initial_scene(scene, config)
        start_s, start_d = track.project(np.asarray(scene.initial_state[:2]))
        goal_s, goal_d = track.project(np.asarray(scene.goal))
        margin = config.vehicle.footprint_radius + config.planner.required_clearance
        right_limit = scene.track.right_width - margin
        left_limit = scene.track.left_width - margin
        for lateral in (start_d, goal_d):
            assert -right_limit - 1e-6 <= lateral <= left_limit + 1e-6
        start_offsets.append(float(start_d))
        goal_offsets.append(float(goal_d))
        obstacle_s, obstacle_d = track.project(scene.obstacles[0].centers[0])
        distance = float((goal_s - start_s) % track.length)
        relative = float((obstacle_s - start_s) % track.length)
        assert settings["goal_arc_range"][0] - 1e-4 <= distance
        assert distance <= settings["goal_arc_range"][1] + 1e-4
        assert 0 < relative < distance
        assert abs(float(obstacle_d)) <= 0.11
        assert all(np.array_equal(scene.obstacles[0].centers[0], center)
                   for center in scene.obstacles[0].centers)
        forward_distances.append(distance)
        wrapped += float(start_s) + distance >= track.length
    assert min(forward_distances) > 6.9
    assert max(forward_distances) > 15
    assert min(start_offsets) < -0.15 < max(start_offsets)
    assert min(goal_offsets) < -0.15 < max(goal_offsets)
    assert wrapped > 0  # Some scenes cross the zero-arc point.


def test_obstacle_free_reference_uses_dstar_for_off_center_endpoints():
    config, settings = read_configuration(CONFIG)
    rows, _ = run_scene_sampling(settings, config)
    scene = Scene.from_record(dict(rows[0], obstacles=[]))
    track = track_for(scene.track)
    result = run_expert_procedural(scene, Guidance("nominal"), config, guess=0, seed=7)
    assert result["eligible"], result["validation"]
    phase = np.linspace(0.0, 1.0, config.planner.horizon + 1)
    progress = phase**3 * (10 + phase * (-15 + 6 * phase))
    start = np.asarray(scene.initial_state[:2])
    goal = np.asarray(scene.goal)
    arcs, laterals = track.project(np.stack((start, goal)))
    distance = float((arcs[1] - arcs[0]) % track.length)
    s = float(arcs[0]) + distance * progress
    d_star = laterals[0] + (laterals[1] - laterals[0]) * progress
    expected = track.point(s) + d_star[:, None] * track.normal(s)
    expected[0], expected[-1] = start, goal
    np.testing.assert_allclose(result["y_ref_star"], expected, atol=1e-8)
    assert np.max(np.abs(d_star)) > 0.1
    representation = SplineRepresentation(config.planner.horizon, 8, asdict(scene.track))
    zeros = np.r_[start, goal, np.zeros(2 * representation.knots)]
    np.testing.assert_allclose(representation.decode(zeros[None])[0], expected, atol=1e-8)


def test_obstacle_avoidance_modulates_dstar_without_moving_endpoints():
    config, settings = read_configuration(CONFIG)
    rows, _ = run_scene_sampling(settings, config)
    scene = Scene.from_record(rows[0])
    result = run_expert_procedural(scene, Guidance("left", side=1), config, guess=0, seed=7)
    assert result["eligible"], result["validation"]
    phase = np.linspace(0.0, 1.0, config.planner.horizon + 1)
    progress = phase**3 * (10 + phase * (-15 + 6 * phase))
    baseline = track_for(scene.track).reference(
        np.asarray(scene.initial_state[:2]), np.asarray(scene.goal), progress
    )
    actual = np.asarray(result["y_ref_star"])
    np.testing.assert_allclose(actual[[0, -1]], baseline[[0, -1]], atol=1e-9)
    assert np.max(np.linalg.norm(actual - baseline, axis=1)) > 0.1


def test_opposite_guidance_produces_independently_validated_reference():
    config, settings = read_configuration(CONFIG)
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


def test_nominal_shortest_route_is_valid_and_not_longer_than_forced_sides():
    config, settings = read_configuration(CONFIG)
    rows, _ = run_scene_sampling(settings, config)
    for row in rows:
        scene = Scene.from_record(row)
        results = [run_expert_procedural(
            scene, Guidance("nominal" if side == 0 else str(side), side),
            config, guess=0, seed=100,
        ) for side in (0, 1, -1)]
        nominal = results[0]
        assert nominal["eligible"], nominal["validation"]
        assert nominal["validation"]["valid"]
        assert nominal["validation"]["minimum_clearance"] >= config.planner.required_clearance
        for forced in results[1:]:
            if forced["eligible"]:
                assert nominal["path_length"] <= forced["path_length"] + 1e-8


def test_procedural_rejects_moving_obstacles_without_relabelling():
    config, settings = read_configuration(CONFIG)
    rows, _ = run_scene_sampling(settings, config)
    row = rows[0]
    obstacle = row["obstacles"][0]
    moved = [list(center) for center in obstacle["centers"]]
    moved[-1][0] += 0.1
    scene = Scene.from_record(dict(row, obstacles=[dict(obstacle, centers=moved)]))
    result = run_expert_procedural(scene, Guidance("nominal", 0), config, guess=0, seed=1)
    assert not result["eligible"]
    assert "fixed obstacles" in result["validation"]["reasons"][0]


def test_sampling_rejects_obstacle_outside_forward_window():
    config, settings = read_configuration(CONFIG)
    for name, value in (
        ("obstacle_fraction_range", [0.0, 0.5]),
        ("obstacle_fraction_range", [0.2, 1.0]),
        ("goal_arc_range", [7.0, 30.0]),
        ("obstacle_lateral_range", [-1.3, 0.1]),
    ):
        invalid = dict(settings, **{name: value})
        with pytest.raises(ValueError):
            run_scene_sampling(invalid, config)
