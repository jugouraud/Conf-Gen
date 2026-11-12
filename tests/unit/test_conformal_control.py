"""Mathematical contracts and finite-search behavior of contextual control."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from backend.experiments.conformal_mpc.synthetic import SyntheticTrajectoryModel
from backend.src.control.calibrate import run_detector_calibration
from backend.src.control.contracts import (
    Anchor,
    Context,
    ControlState,
    Detector,
    Example,
    PreviousReference,
    ProbeBank,
    RepairChoice,
    Sampling,
)
from backend.src.control.coverage import run_output_calibration, upper_score_quantile
from backend.src.control.partition import validate_partitions
from backend.src.control.repair import choose_schedule, run_drift_repair
from backend.src.control.sample import integrate_field
from backend.src.control.signatures import (
    evaluate_signature,
    mix_signatures,
    score_signature,
    signature_distances,
)
from backend.src.detection.calibration import calculate_conformal_cutoff
from backend.src.mpc.geometry import Obstacle, minimum_clearance
from backend.src.repair.steering import (
    build_constant_steering_schedule,
    build_window_steering_schedule,
)

pytestmark = pytest.mark.unit


def scalar_detector(safe=(0.0,), unsafe=(2.0,), r_safe=0.1, r_unsafe=0.2) -> Detector:
    probes = ProbeBank(np.zeros((1, 1)), np.array([0.5]), ("train",), "scalar")

    def anchors(values):
        return tuple(
            Anchor("train", str(v), "default", np.array([[v]])) for v in values
        )

    return Detector(probes, anchors(safe), anchors(unsafe), r_safe, r_unsafe)


@pytest.mark.parametrize(
    "scores,alpha,expected",
    [
        ([1, 2, 3, 4], 0.2, 4),
        ([1, 2, 3, 4], 0.19, np.inf),
        ([2, 2, 2], 0.5, 2),
        ([7], 0.5, 7),
        ([7], 0.49, np.inf),
    ],
)
def test_quantile_rank_boundaries_and_ties(scores, alpha, expected):
    assert calculate_conformal_cutoff(scores, alpha) == expected
    assert upper_score_quantile(scores, alpha) == expected


def test_infinite_failed_output_scores_are_not_dropped():
    assert upper_score_quantile([], 0.1) == np.inf
    assert upper_score_quantile([0, 1, np.inf], 0.25) == np.inf
    fit = run_output_calibration(
        {"a": 0, "b": np.inf},
        policy_id="p",
        target="reference",
        alpha=0.5,
        forbidden_episodes=["train"],
    )
    assert fit.coverage({"test": np.inf}, policy_id="p", target="reference") == 1
    with pytest.raises(ValueError, match="changed"):
        fit.coverage({"test": 0}, policy_id="other", target="reference")
    with pytest.raises(ValueError, match="changed"):
        fit.coverage({"test": 0}, policy_id="p", target="executed")
    with pytest.raises(ValueError, match="reused"):
        fit.coverage({"a": 0}, policy_id="p", target="reference")
    with pytest.raises(ValueError, match="fresh"):
        run_output_calibration(
            {"train": 0},
            policy_id="p",
            target="executed",
            alpha=0.2,
            forbidden_episodes=["train"],
        )


def test_detector_preserves_inclusive_safe_and_closed_unsafe_ties():
    detector = scalar_detector(r_safe=1.0, r_unsafe=1.0)
    tie = score_signature(np.array([[1.0]]), detector)
    assert not tie.safe_flag
    assert tie.unsafe_flag and tie.flagged
    assert not score_signature(np.array([[0.99]]), detector).flagged
    assert score_signature(np.array([[1.01]]), detector).flagged
    assert score_signature(
        np.array([[0.0]]), replace(detector, r_unsafe=np.inf)
    ).flagged


@pytest.mark.parametrize("aggregation,expected", [("nearest", 1), ("mean", 2)])
def test_existing_anchor_aggregation_is_reused(aggregation, expected):
    detector = replace(scalar_detector(safe=(0.0, 4.0)), aggregation=aggregation)
    assert score_signature(np.array([[1.0]]), detector).d_safe == expected


def test_exponential_aggregation_and_rms_normalization():
    detector = replace(
        scalar_detector(safe=(0.0, 4.0)), aggregation="exponential", bandwidth=2
    )
    expected = (np.exp(-1 / 8) + 3 * np.exp(-9 / 8)) / (np.exp(-1 / 8) + np.exp(-9 / 8))
    assert score_signature(np.array([[1.0]]), detector).d_safe == pytest.approx(
        expected
    )
    assert signature_distances(np.zeros((2, 2)), np.array([[[3, 4], [0, 0]]]))[
        0
    ] == pytest.approx(5 / np.sqrt(2))


def test_nearly_equal_large_signatures_preserve_zero_and_small_distances():
    query = np.full((3, 10), 1e6)
    references = np.stack((query, query + 1e-5))
    expected = np.linalg.norm(references - query, axis=(1, 2)) / np.sqrt(3)
    np.testing.assert_allclose(
        signature_distances(query, references), expected, rtol=1e-12, atol=0
    )


def test_search_checks_disconnected_feasible_candidates_and_chooses_minimum():
    detector = scalar_detector(safe=(0, 2, 4), unsafe=(1, 3))
    schedules = [
        build_constant_steering_schedule(v) for v in (1, 0.8, 0.6, 0.4, 0.2, 0)
    ]
    choice = choose_schedule(np.array([[-1.0]]), np.array([[4.0]]), detector, schedules)
    assert choice.evaluated == 6
    assert choice.schedule.integrate_strength() == pytest.approx(0.2)
    assert not choice.decision.flagged


def test_search_minimizes_integrated_strength_across_profiles():
    detector = scalar_detector()
    window = build_window_steering_schedule("window", [0.5], 0.4)
    choice = choose_schedule(
        np.array([[2.0]]),
        np.array([[0.0]]),
        detector,
        [build_constant_steering_schedule(1), window],
    )
    assert choice.schedule == window
    assert choice.schedule.integrate_strength() < 1
    impossible = choose_schedule(
        np.array([[2.0]]), np.array([[2.0]]), detector, [window]
    )
    assert impossible.status == "repair_infeasible"


def test_signature_identity_matches_actual_mixed_field_and_context_is_immutable():
    model = SyntheticTrajectoryModel()
    context = Context("episode", 0, (0, 0), (4, 0), (), 0.4, 8)
    probes = ProbeBank(
        np.ones((3, 9, 2)), np.array([0.1, 0.5, 0.9]), ("train",), model.revision
    )
    nominal = evaluate_signature(model, probes, context, "direct")
    anchor = evaluate_signature(model, probes, context, "upper")
    schedule = build_window_steering_schedule("window", [0.5], 0.7)
    measured = np.stack(
        [
            (1 - schedule(t)) * model.drift(x, t, context, "direct")
            + schedule(t) * model.drift(x, t, context, "upper")
            - model.baseline(x, t)
            for x, t in zip(probes.states, probes.times, strict=True)
        ]
    )
    np.testing.assert_allclose(
        mix_signatures(nominal, anchor, probes.times, schedule), measured, atol=1e-14
    )
    with pytest.raises(FrozenInstanceError):
        context.goal = (9, 9)
    with pytest.raises(ValueError):
        probes.states.flags.writeable = True


def test_anchor_is_evaluated_in_current_context_and_unflagged_query_is_unsteered():
    model = SyntheticTrajectoryModel()
    current = Context("query", 0, (0, 0), (4, 0), (), 0.4, 8)
    probes = ProbeBank(np.zeros((1, 9, 2)), np.array([0.5]), ("train",), model.revision)
    nominal = evaluate_signature(model, probes, current, "direct")
    approved = evaluate_signature(model, probes, current, "upper")
    # Stored anchor context differs; returning its stored signature for repair
    # would produce a different candidate and fail this behavioral assertion.
    safe = Anchor("train", "upper", "default", approved + 0.01)
    unsafe = Anchor("train", "direct", "default", nominal)
    detector = Detector(probes, (safe,), (unsafe,), 0.1, 0.1)
    state = ControlState(
        current,
        0,
        100,
        lambda: 0,
        signature=nominal,
        decision=score_signature(nominal, detector),
    )
    repaired = run_drift_repair(
        state, model, detector, [build_constant_steering_schedule(1)], "repaired"
    )
    assert repaired.repair.status == "repaired" and repaired.context is current
    assert repaired.repair.decision.d_safe == pytest.approx(np.sqrt(18) * 0.01)
    accepted = replace(
        state, signature=approved, decision=score_signature(approved, detector)
    )
    assert (
        run_drift_repair(
            accepted, model, detector, [], "repaired"
        ).repair.schedule.integrate_strength()
        == 0
    )
    missing = replace(state, context=replace(current, compatibility="other"))
    assert (
        run_drift_repair(
            missing, model, detector, [build_constant_steering_schedule(1)], "repaired"
        ).failure
        == "missing_anchor"
    )


def test_episode_separation_and_probe_lineage_are_enforced():
    model = SyntheticTrajectoryModel()
    context = Context("r", 0, (0, 0), (4, 0), (), 0.4, 8)
    groups = {
        name: (Example(replace(context, episode=name), "direct", True),)
        for name in ("reference", "detector_calibration", "output_calibration", "test")
    }
    validate_partitions(groups)
    leaked = dict(groups, test=groups["reference"])
    with pytest.raises(ValueError, match="leakage"):
        validate_partitions(leaked)
    probes = ProbeBank(np.zeros((1, 9, 2)), np.array([0.5]), ("test",), model.revision)
    with pytest.raises(ValueError, match="leaked"):
        run_detector_calibration(
            model, probes, groups, alpha_safe=0.2, alpha_unsafe=0.2
        )


def test_absent_calibration_class_retains_infinite_radius_without_capping():
    model = SyntheticTrajectoryModel()
    context = Context("reference", 0, (0, 0), (4, 0), (), 0.4, 8)
    groups = {
        name: (Example(replace(context, episode=name), "upper", True),)
        for name in ("reference", "detector_calibration", "output_calibration", "test")
    }
    groups["reference"] += (
        Example(replace(context, episode="unsafe-reference"), "direct", False),
    )
    probes = ProbeBank(
        np.zeros((1, 9, 2)), np.array([0.5]), ("reference",), model.revision
    )
    detector, calibration = run_detector_calibration(
        model, probes, groups, alpha_safe=0.5, alpha_unsafe=0.2
    )
    assert detector.r_safe == 0 and detector.r_unsafe == np.inf
    assert calibration["unsafe_scores"] == []
    assert score_signature(detector.safe[0].signature, detector).flagged


def test_moving_obstacle_crossing_is_caught_between_safe_waypoints():
    obstacle = Obstacle(((0, -1), (0, 1)), 0.1)
    reference = np.array([[-1, 0], [1, 0]])
    assert minimum_clearance(reference, (obstacle,), 0.2) == pytest.approx(-0.3)
    assert minimum_clearance(reference, (), 0.2) == np.inf


@pytest.mark.parametrize("method", ["ode", "sde"])
def test_sampling_replays_paired_randomness_and_warm_protocol(method):
    model = SyntheticTrajectoryModel()
    context = Context("episode", 1, (0, 0), (4, 0), (), 0.4, 8)
    choice = RepairChoice("unsteered", build_constant_steering_schedule())
    state = ControlState(context, 0, 100, lambda: 0, repair=choice)
    sampling = Sampling(16, method)
    first = integrate_field(model, state, "direct", sampling, 123)
    np.testing.assert_array_equal(
        first, integrate_field(model, state, "direct", sampling, 123)
    )
    assert not np.array_equal(
        first, integrate_field(model, state, "direct", sampling, 124)
    )
    previous = PreviousReference(replace(context, k=0), model.target(context, "upper"))
    warmed = integrate_field(
        model, state, "direct", replace(sampling, warm_start="shifted"), 123, previous
    )
    assert not np.array_equal(first, warmed)
    with pytest.raises(ValueError, match="incompatible"):
        previous.shifted(replace(context, episode="other"))
    with pytest.raises(ValueError, match="incompatible"):
        previous.shifted(replace(context, goal=(9, 9)))


def test_paired_sde_noise_is_unchanged_by_drift_repair():
    model = SyntheticTrajectoryModel()
    context = Context("episode", 0, (0, 0), (4, 0), (), 0.4, 8)
    anchor = Anchor("train", "upper", "default", np.zeros((1, 9, 2)))
    differences = []
    for method in ("ode", "sde"):
        results = []
        for strength in (0, 0.5):
            state = ControlState(
                context,
                0,
                100,
                lambda: 0,
                repair=RepairChoice(
                    "repaired", build_constant_steering_schedule(strength), anchor
                ),
            )
            results.append(
                integrate_field(model, state, "direct", Sampling(32, method), 42)
            )
        differences.append(results[1] - results[0])
    np.testing.assert_allclose(differences[0], differences[1], atol=2e-15)


def test_nonlinear_mixed_field_is_integrated_not_decoded_path_interpolation():
    class NonlinearModel(SyntheticTrajectoryModel):
        def initialize(self, c, rng, warm):
            return np.ones((2, 2)) * 0.1

        def drift(self, x, t, c, a):
            return x * x if a == "upper" else np.ones_like(x)

    model = NonlinearModel()
    context = Context("episode", 0, (0, 0), (1, 1), (), 1, 1)
    anchor = Anchor("train", "upper", "default", np.zeros((1, 2, 2)))
    state = ControlState(
        context,
        0,
        100,
        lambda: 0,
        repair=RepairChoice("repaired", build_constant_steering_schedule(0.5), anchor),
    )
    mixed = integrate_field(model, state, "direct", Sampling(32), 1)
    nominal = integrate_field(
        model,
        replace(
            state, repair=RepairChoice("unsteered", build_constant_steering_schedule())
        ),
        "direct",
        Sampling(32),
        1,
    )
    approved = integrate_field(
        model,
        replace(
            state, repair=RepairChoice("unsteered", build_constant_steering_schedule())
        ),
        "upper",
        Sampling(32),
        1,
    )
    assert not np.allclose(mixed, (nominal + approved) / 2, atol=1e-3)
