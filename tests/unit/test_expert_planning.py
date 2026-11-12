"""Goal optimization, physical timing and independently checked constraints."""

from dataclasses import replace

import numpy as np
import pytest

from backend.src.expert.config import Guidance, VehicleSettings
from backend.src.expert.geometry import timed_clearances
from backend.src.expert.normalize import training_statistics
from backend.src.expert.optimize import run_expert_optimization
from backend.src.expert.scenes import Scene, TimedObstacle, assign_groups
from backend.src.expert.validate import run_expert_validation, validate_initial_scene
from backend.src.expert.vehicle import Vehicle

pytestmark = pytest.mark.unit


def simple_solution(config, raw):
    config = replace(config, planner=replace(config.planner, horizon=3, dt=1.0))
    scene = Scene.from_record(raw)
    controls = np.array([[1.0, 0], [1.0, 0], [0.0, 0]])
    vehicle = Vehicle(config.vehicle, 1.0)
    states = vehicle.rollout(np.asarray(scene.initial_state), controls)
    times = scene.start_time + np.arange(4)
    return config, scene, vehicle, states, controls, times


def test_expert_solves_scene_and_goal_without_a_generated_reference(
    expert_config, expert_scenes
):
    scene = Scene.from_record(expert_scenes[0])
    result = run_expert_optimization(
        scene, expert_config.guidance[0], expert_config, guess=0, seed=10
    )
    assert result["eligible"], result
    assert (
        result["validation"]["goal_error"]
        <= expert_config.planner.goal_tolerance + 1e-6
    )
    assert result["validation"]["maximum_constraint_residual"] <= 1e-6
    np.testing.assert_allclose(
        np.asarray(result["states"])[1:] - np.asarray(result["states"])[:-1],
        np.asarray(result["controls"]) * 0.5,
        atol=1e-12,
    )
    np.testing.assert_array_equal(result["timestamps"], 100 + np.arange(9) * 0.5)


def test_opposite_passages_preserve_scene_and_describe_distinct_solutions(
    expert_config, expert_scenes
):
    raw = dict(
        expert_scenes[0],
        obstacles=[
            {
                "obstacle_id": "gate",
                "times": [100.0, 104.0],
                "centers": [[1.0, 0.0], [1.0, 0.0]],
                "radius": 0.12,
            }
        ],
    )
    scene = Scene.from_record(raw)
    config = replace(
        expert_config,
        planner=replace(expert_config.planner, initial_guess_offsets=(0.7,)),
    )
    references = []
    for side in (1, -1):
        result = run_expert_optimization(
            scene, Guidance(str(side), side), config, guess=0, seed=10
        )
        assert result["eligible"], result["validation"]
        assert all(
            side * x > 0.3 for x in result["validation"]["passage_lateral_offsets"]
        )
        references.append(np.asarray(result["y_ref_star"]))
    assert np.sqrt(np.mean((references[0] - references[1]) ** 2)) > 0.2
    np.testing.assert_array_equal(scene.obstacles[0].centers, [[1, 0], [1, 0]])
    assert scene.goal == (2, 0)


def test_obstacle_turn_between_optimizer_knots_is_checked_at_absolute_time():
    obstacle = TimedObstacle(
        "moving",
        np.array([100, 100.5, 101]),
        np.array([[0.5, 1], [0.5, 0], [0.5, 1]]),
        0.1,
    )
    clearance = timed_clearances(
        np.array([100, 101]), np.array([[0, 0], [1, 0]]), (obstacle,), 0.15
    )
    assert clearance.min() == pytest.approx(-0.25)
    with pytest.raises(ValueError, match="cover"):
        timed_clearances(
            np.array([0, 1]), np.array([[0, 0], [1, 0]]), (obstacle,), 0.15
        )


def test_swept_reference_rejects_static_collision_between_clear_endpoints(
    expert_config, expert_scenes
):
    cfg, scene, vehicle, states, controls, times = simple_solution(
        expert_config, expert_scenes[0]
    )
    scene = replace(
        scene,
        obstacles=(TimedObstacle("static", [100, 103], [[0.5, 0], [0.5, 0]], 0.1),),
    )
    result = run_expert_validation(
        scene, Guidance("free"), cfg, states, controls, times, vehicle.output(states)
    )
    assert not result["valid"] and "clearance" in result["reasons"]
    assert result["minimum_clearance"] == pytest.approx(-0.25)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("dimensions", "reference_dimensions_or_nonfinite_values"),
        ("time", "physical_timestamps_mismatch"),
        ("nan", "reference_dimensions_or_nonfinite_values"),
        ("dynamics", "dynamics"),
        ("reference", "reference_output"),
        ("initial", "start"),
        ("goal", "goal_or_progress"),
        ("limits", "acceleration_limit"),
        ("bounds", "map_boundary"),
    ],
)
def test_final_export_checks_reject_invalid_targets(
    expert_config, expert_scenes, mutation, reason
):
    cfg, scene, vehicle, states, controls, times = simple_solution(
        expert_config, expert_scenes[0]
    )
    reference = vehicle.output(states)
    if mutation == "dimensions":
        reference = reference[:-1]
    elif mutation == "time":
        times = times + 0.1
    elif mutation == "nan":
        reference[1, 0] = np.nan
    elif mutation == "dynamics":
        states[1, 0] += 0.1
    elif mutation == "reference":
        reference[1, 1] += 0.1
    elif mutation == "initial":
        states[0, 0] = 0.1
    elif mutation == "goal":
        scene = replace(scene, goal=(3, 0))
    elif mutation == "limits":
        controls[1] = [-2, 0]
    elif mutation == "bounds":
        states[1, 1] = 3
        reference = vehicle.output(states)
    result = run_expert_validation(
        scene, Guidance("free"), cfg, states, controls, times, reference
    )
    assert not result["valid"] and reason in result["reasons"]


def test_progress_mode_has_explicit_terminal_condition(expert_config, expert_scenes):
    cfg, scene, vehicle, states, controls, times = simple_solution(
        expert_config, expert_scenes[0]
    )
    scene = replace(scene, goal=(3.5, 0))
    cfg = replace(
        cfg,
        planner=replace(cfg.planner, terminal_mode="progress", minimum_progress=1.5),
    )
    result = run_expert_validation(
        scene, Guidance("free"), cfg, states, controls, times, vehicle.output(states)
    )
    assert result["valid"] and result["progress"] == 2


def test_bicycle_unwrapped_heading_sincos_output_and_actuators(
    expert_config, expert_scenes
):
    vehicle_config = VehicleSettings(
        kind="bicycle",
        output="xy_sincos",
        state_lower=(-5, -5, -100, 0, -0.6),
        state_upper=(5, 5, 100, 2, 0.6),
        input_lower=(-2, -1),
        input_upper=(2, 1),
    )
    cfg = replace(
        expert_config,
        vehicle=vehicle_config,
        planner=replace(expert_config.planner, horizon=2, dt=0.5),
    )
    scene = Scene.from_record(
        dict(
            expert_scenes[0],
            initial_state=[0, 0, 2 * np.pi, 1, 0],
            initial_velocity=[1, 0],
            goal=[0.5, 0],
            goal_heading=0.0,
        )
    )
    vehicle = Vehicle(cfg.vehicle, cfg.planner.dt)
    controls = np.array([[-1, 0], [-1, 0]], dtype=float)
    states = vehicle.rollout(np.asarray(scene.initial_state), controls)
    reference = vehicle.output(states)
    np.testing.assert_allclose(states[:, 0], [0, 0.375, 0.5], atol=1e-12)
    np.testing.assert_allclose(reference[:, 2:], np.tile([0, 1], (3, 1)), atol=1e-12)
    times = 100 + np.arange(3) * 0.5
    result = run_expert_validation(
        scene, Guidance("free"), cfg, states, controls, times, reference
    )
    assert result["valid"], result
    bad_controls = controls.copy()
    bad_controls[0, 1] = 0.9
    bad = run_expert_validation(
        scene, Guidance("free"), cfg, states, bad_controls, times, reference
    )
    assert "steering_limits" in bad["reasons"]
    states[-1, 2] = 0
    wrapped = run_expert_validation(
        scene, Guidance("free"), cfg, states, controls, times, vehicle.output(states)
    )
    assert "dynamics" in wrapped["reasons"]


def test_invalid_initial_conditions_and_solver_deadline_are_explicit(
    expert_config, expert_scenes
):
    scene = Scene.from_record(
        dict(
            expert_scenes[0],
            obstacles=[
                {
                    "obstacle_id": "bad",
                    "times": [100, 104],
                    "centers": [[0, 0], [0, 0]],
                    "radius": 0.2,
                }
            ],
        )
    )
    assert "initial_obstacle_clearance" in validate_initial_scene(scene, expert_config)
    assert (
        run_expert_optimization(
            scene, Guidance("free"), expert_config, guess=0, seed=1
        )["status"]
        == "invalid_scene_or_guidance"
    )
    times = iter([0.0, expert_config.planner.solver_timeout + 1])
    result = run_expert_optimization(
        Scene.from_record(expert_scenes[0]),
        Guidance("free"),
        expert_config,
        guess=0,
        seed=1,
        clock=lambda: next(times, expert_config.planner.solver_timeout + 1),
    )
    assert result["status"] == "solver_timeout" and not result["eligible"]


def test_group_partition_and_normalization_ignore_heldout_variants(
    expert_config, expert_scenes
):
    groups = assign_groups(["a", "a", "b", "c", "d", "e"], expert_config.dataset)
    assert set(groups.values()) == {"train", "reference", "future_calibration", "test"}
    assert groups == assign_groups(["e", "d", "c", "b", "a"], expert_config.dataset)
    first = {
        "target_id": "t",
        "split": "train",
        "group_id": "a",
        "observation": expert_scenes[0],
        "y_ref_star": [[0, 0], [2, 0]],
        "controls": [[1, 0]],
    }
    heldout = dict(
        first, target_id="h", group_id="b", split="test", y_ref_star=[[1e6, 1e6]]
    )
    fit = training_statistics([first, heldout], expert_config)
    assert fit == training_statistics([first], expert_config)
    assert fit["statistics"]["y_ref_star"]["mean"] == [1, 0]
    assert fit["groups"] == ["a"]
