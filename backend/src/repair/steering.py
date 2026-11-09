"""Repair schedules, field geometry, constraint search, and sampling."""

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from backend.src.detection.calibration import aggregate_anchor_distances
from backend.src.fields.signatures import evaluate_probe_field


def calculate_per_time_field_distances(query, anchor, bank):
    return (
        (query.double() - anchor.double())
        .flatten(1)
        .norm(dim=1)
        .reshape(bank["time_points"], bank["trajectories"])
    )


def build_temporal_steering_geometry(query, anchor, experiment):
    """Sufficient statistics of freshly measured fields, exact at frozen states.

    Each coefficient sums latent coordinates and the original probe weights;
    temporal scheduling never changes the detector's aggregate score.
    """

    accelerated = getattr(experiment.features, "build_temporal_geometry", None)
    if accelerated is not None:
        return accelerated(query, anchor, experiment)
    bank = experiment.bank
    weights = bank["weights"].double()
    k, m = bank["time_points"], bank["trajectories"]
    direction = anchor.double() - query.double()
    quadratic = (
        (direction.flatten(1).square().sum(1) * weights).reshape(k, m).sum(1).numpy()
    )
    geometry = {}
    for label in ("safe", "unsafe"):
        linear, constant = [], []
        split = f"{label}_reference"
        for row in experiment.groups[split]:
            reference = experiment.features.load_signature(split, row)
            difference = query.double() - reference.double()
            constant.append(
                (difference.flatten(1).square().sum(1) * weights)
                .reshape(k, m)
                .sum(1)
                .numpy()
            )
            linear.append(
                (2 * (difference * direction).flatten(1).sum(1) * weights)
                .reshape(k, m)
                .sum(1)
                .numpy()
            )
        geometry[label] = (np.asarray(constant), np.asarray(linear), quadratic)
    return TemporalSteeringGeometry(
        geometry, bank["midpoints"], experiment.calibration, query.numel()
    )


def score_field_constraints(values, experiment, mode):

    scores = {}
    weights = experiment.bank["weights"].double()
    for label in ("safe", "unsafe"):
        distances = []
        split = f"{label}_reference"
        for row in experiment.groups[split]:
            reference = experiment.features.load_signature(split, row)
            squared = (values.double() - reference.double()).flatten(1).square().sum(1)
            distances.append(float((squared * weights).sum().sqrt()))
        scores[label] = float(
            aggregate_anchor_distances(distances, {"method": "euclidean"})
        )
    cutoff = (
        experiment.calibration["unsafe_cutoff"]
        if mode == "safe_attraction_unsafe_avoidance"
        else None
    )
    return RepairConstraintScores(
        scores["safe"], scores["unsafe"], experiment.calibration["safe_cutoff"], cutoff
    )


def score_verified_field_constraints(
    values, query, anchor, schedule, geometry, experiment, mode
):
    """Direct nearest-reference scores of actual model output.

    Exact affine statistics locate possible nearest references. The measured
    residual and triangle inequality exclude all other references before I/O;
    distances to the remaining candidates use the directly evaluated field.
    No interpolated score is returned as a measured field score.
    """

    weights = experiment.bank["weights"].double()
    strengths = torch.tensor(
        [schedule(float(time)) for time in experiment.bank["times"]],
        dtype=torch.float64,
    )
    strengths = strengths.reshape((-1,) + (1,) * (values.ndim - 1))
    predicted = query.double() + strengths * (anchor.double() - query.double())
    residual = float(
        ((values.double() - predicted).flatten(1).square().sum(1) * weights)
        .sum()
        .sqrt()
    )
    scores = {}
    counts = {}
    for label in ("safe", "unsafe"):
        indexes = geometry.select_possible_nearest_references(schedule, label, residual)
        distances = []
        split = f"{label}_reference"
        for index in indexes:
            reference = experiment.features.load_signature(
                split, experiment.groups[split][int(index)]
            )
            squared = (values.double() - reference.double()).flatten(1).square().sum(1)
            distances.append(float((squared * weights).sum().sqrt()))
        scores[label] = min(distances)
        counts[label] = len(indexes)
    cutoff = (
        experiment.calibration["unsafe_cutoff"]
        if mode == "safe_attraction_unsafe_avoidance"
        else None
    )
    return RepairConstraintScores(
        scores["safe"], scores["unsafe"], experiment.calibration["safe_cutoff"], cutoff
    ), {"affine_model_residual": residual, "direct_reference_candidates": counts}


def select_nearest_approved_anchor(experiment, split, index, mode):
    safe = experiment.distances[f"safe__{split}"][index]
    eligible = (
        experiment.distances["safe__safe_reference"].min(axis=1)
        <= experiment.calibration["safe_cutoff"]
    )
    if mode == "safe_attraction_unsafe_avoidance":
        eligible &= (
            experiment.distances["unsafe__safe_reference"].min(axis=1)
            > experiment.calibration["unsafe_cutoff"]
        )
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        return None
    return experiment.groups["safe_reference"][
        int(candidates[np.argmin(safe[candidates])])
    ]


def build_constant_steering_schedule(strength=0.0, name="R0"):

    return SteeringSchedule(name, (0.0, 1.0), (float(strength), float(strength)))


def build_window_steering_schedule(name, centers, width, ramp_fraction=0.2):
    """Union of contiguous windows with smooth ramps, including domain boundaries."""

    if not 0 < width <= 1 or not 0 < ramp_fraction <= 0.5:
        raise ValueError("Invalid intervention window or ramp width")
    if any(not 0 <= float(center) <= 1 for center in centers):
        raise ValueError("Window centers must lie in [0,1]")
    left_edges = [
        min(max(0.0, float(center) - width / 2), 1 - width) for center in centers
    ]
    windows = [(left, left + width) for left in left_edges]
    knots = {0.0, 1.0}
    for left, right in windows:
        ramp = (right - left) * ramp_fraction
        knots.update((left, left + ramp, right - ramp, right))
    times = tuple(sorted(knots))
    values = []
    for flow_time in times:
        levels = []
        for left, right in windows:
            ramp = (right - left) * ramp_fraction
            level = min(1.0, (flow_time - left) / ramp, (right - flow_time) / ramp)
            if left == 0 and flow_time < left + ramp:
                level = 1.0
            if right == 1 and flow_time > right - ramp:
                level = 1.0
            if level > 1 - 1e-12:
                level = 1.0
            levels.append(max(0.0, level))
        values.append(max(levels, default=0.0))
    return SteeringSchedule(name, times, tuple(values))


def calculate_normalized_deviation_profiles(query, anchor, bank, normalization):
    """Normalize per-probe latent norms using safe-only per-time location/scale."""
    differences = (query.double() - anchor.double()).flatten(1).norm(dim=1).numpy()
    deviations = differences.reshape(bank["time_points"], bank["trajectories"])
    normalized = np.maximum(
        0.0,
        (deviations - np.asarray(normalization["location"])[:, None])
        / np.asarray(normalization["scale"])[:, None],
    )
    descending = np.sort(normalized, axis=1)[:, ::-1]
    return {
        "max": descending[:, 0],
        "mean": normalized.mean(axis=1),
        **{
            f"top{percent}": descending[
                :, : max(1, math.ceil(normalized.shape[1] * percent / 100))
            ].mean(axis=1)
            for percent in (10, 25)
        },
    }


def build_strategy_steering_schedules(methods, bank, profiles, settings):
    """Return all preregistered variants; held-out outcomes never choose a width."""
    schedules = []
    if "R1" in methods:
        schedules.append(build_constant_steering_schedule(1.0, "R1"))
    times = np.asarray(bank["midpoints"], dtype=float)
    for width in settings["window_widths"]:
        if "R2" in methods:
            for label, center in (
                ("early", width / 2),
                ("middle", 0.5),
                ("late", 1 - width / 2),
            ):
                schedules.append(
                    build_window_steering_schedule(
                        f"R2_{label}_w{width:g}",
                        [center],
                        width,
                        settings["ramp_fraction"],
                    )
                )
        if "R3" in methods:
            center = times[int(np.argmax(profiles["max"]))]
            schedules.append(
                build_window_steering_schedule(
                    f"R3_max_w{width:g}", [center], width, settings["ramp_fraction"]
                )
            )
        if "R4" in methods:
            for statistic in ("mean", "top10", "top25"):
                values = profiles[statistic]
                left_edges = np.clip(times - width / 2, 0.0, 1 - width)
                averages = [
                    float(values[(times >= left) & (times <= left + width)].mean())
                    for left in left_edges
                ]
                selected = []
                for index in np.argsort(-np.asarray(averages), kind="stable"):
                    if all(abs(times[index] - center) >= width for center in selected):
                        selected.append(float(times[index]))
                    if len(selected) == settings["top_k"]:
                        break
                schedules.append(
                    build_window_steering_schedule(
                        f"R4_{statistic}_w{width:g}",
                        selected,
                        width,
                        settings["ramp_fraction"],
                    )
                )
    return schedules


def evaluate_repaired_probes(config, field, bank):
    """Reevaluate the chosen schedule through the model, never a cached mixture."""
    remote_evaluator = getattr(field.adapter, "evaluate_repaired_probes", None)
    if remote_evaluator is not None:
        return remote_evaluator(config, field, bank)
    return evaluate_probe_field(config, field, bank, None)


def sample_repaired_trajectory(field, seed, steps):
    if steps < 1:
        raise ValueError("Inference steps must be positive")
    adapter = field.adapter
    remote_sampler = getattr(adapter, "sample_trajectory", None)
    if remote_sampler is not None:
        return remote_sampler(field, seed, steps)
    noise = torch.randn(
        (1, *adapter.shape), generator=torch.Generator("cpu").manual_seed(int(seed))
    )
    states = noise.to(adapter.device)
    strengths = []
    perturbation = 0.0
    # One common Euler grid across all methods; schedule knots never alter it.
    nodes = np.linspace(0.0, 1.0, steps + 1)
    cuda = str(adapter.device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    evaluations_before = field.model_evaluations
    diagnostic_evaluations = 0
    for current, following in itertools.pairwise(nodes):
        strength = field.schedule(current)
        velocity = field.evaluate_guided_velocity(states, current)
        if strength:
            cached = getattr(adapter, "cached_diagnostic_velocity", None)
            original = (
                cached(states, current, field.prompt_embeddings)
                if cached is not None
                else None
            )
            if original is None:
                original = adapter.evaluate_guided_velocity(
                    states, current, field.prompt_embeddings
                )
                diagnostic_evaluations += 1
            perturbation += float(following - current) * float(
                (velocity - original).double().norm()
            )
        states = states + float(following - current) * velocity
        if not torch.isfinite(states).all():
            raise ValueError("Non-finite repaired trajectory")
        strengths.append(strength)
    if cuda:
        torch.cuda.synchronize()
    return states.detach().cpu(), {
        "inference_seconds": time.perf_counter() - started,
        "model_evaluations": field.model_evaluations - evaluations_before,
        "diagnostic_model_evaluations": diagnostic_evaluations,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if cuda
        else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        "mean_strength": float(np.mean(strengths)),
        "modified_step_fraction": float(np.mean(np.asarray(strengths) > 0)),
        "integrated_field_perturbation": perturbation,
        "sampler": "Euler",
        "steps": steps,
    }


def search_schedule_strength(shape, score, settings):
    """Check the entire grid, refine every detected entry, verify candidates.

    Disconnected feasible intervals are retained. Intervals narrower than grid
    resolution can be missed: this is a bounded numerical search, not a proof
    of global minimality or infeasibility.
    """

    grid = np.linspace(0.0, 1.0, settings["grid_points"])
    outcomes = [score(shape.scale_strength(float(value))) for value in grid]
    evaluations = len(outcomes)
    intervals, candidates = [], []
    inside = np.asarray([outcome.satisfies_constraints for outcome in outcomes])
    for index in np.flatnonzero(inside & ~np.r_[False, inside[:-1]]):
        left, right = float(grid[max(0, index - 1)]), float(grid[index])
        for _ in range(settings["refinement_steps"]):
            midpoint = (left + right) / 2
            feasible = score(shape.scale_strength(midpoint)).satisfies_constraints
            evaluations += 1
            if feasible:
                right = midpoint
            else:
                left = midpoint
        end = index
        while end + 1 < len(grid) and inside[end + 1]:
            end += 1
        intervals.append(
            {"entry": right, "last_feasible_grid_strength": float(grid[end])}
        )
        schedule = shape.scale_strength(right)
        verified = score(schedule)
        evaluations += 1
        if verified.satisfies_constraints:
            candidates.append((schedule.integrate_strength(), schedule, verified))
    if not candidates:
        schedule = shape.scale_strength(0.0)
        return RepairScheduleChoice(
            schedule, outcomes[0], evaluations, intervals, "no_feasible_grid_interval"
        )
    _, schedule, verified = min(candidates, key=lambda item: item[0])
    return RepairScheduleChoice(schedule, verified, evaluations, intervals)


def optimize_schedule_knots(score, settings, initial_choices=()):
    """Deterministic multistart coordinate search over smooth bounded knots.

    The penalty guides exploration, while feasible incumbents are ranked by
    intervention plus smoothness. Penalty satisfaction never substitutes for
    explicit safe/unsafe inequality checks.
    """

    times = tuple(
        float(x) for x in np.linspace(0.0, 1.0, settings["optimization_knots"])
    )
    starts = [np.zeros(len(times)), np.ones(len(times))]
    starts.extend(
        np.asarray([choice.schedule(t) for t in times])
        for choice in initial_choices
        if choice.scores.satisfies_constraints
    )
    best = None
    evaluations = 0
    for start in starts:
        current = start.copy()
        current_objective = float("inf")
        for step in settings["coordinate_steps"]:
            for _ in range(settings["optimization_sweeps"]):
                changed = False
                for index in range(len(times)):
                    proposals = [current.copy()]
                    for delta in (-step, step):
                        proposal = current.copy()
                        proposal[index] = np.clip(proposal[index] + delta, 0.0, 1.0)
                        proposals.append(proposal)
                    for proposal in proposals:
                        schedule = SteeringSchedule(
                            "R5", times, tuple(float(x) for x in proposal)
                        )
                        scores = score(schedule)
                        evaluations += 1
                        cost = (
                            schedule.integrate_strength()
                            + settings["smoothness_weight"]
                            * schedule.integrate_squared_derivative()
                        )
                        violation = max(0.0, scores.safe / scores.safe_cutoff - 1) ** 2
                        if scores.unsafe_cutoff is not None:
                            violation += (
                                max(
                                    0.0,
                                    1
                                    + settings["unsafe_margin"]
                                    - scores.unsafe / scores.unsafe_cutoff,
                                )
                                ** 2
                            )
                        objective = cost + settings["constraint_penalty"] * violation
                        if scores.satisfies_constraints and (
                            best is None or cost < best[0]
                        ):
                            best = (cost, schedule, scores)
                        if objective < current_objective - 1e-12:
                            current, current_objective = proposal, objective
                            changed = True
                if not changed:
                    break
    if best is None:
        schedule = build_constant_steering_schedule(0.0, "R5")
        return RepairScheduleChoice(
            schedule, score(schedule), evaluations + 1, status="optimization_unresolved"
        )
    # Reverify independently of the optimizer's penalty objective.
    verified = score(best[1])
    return RepairScheduleChoice(
        best[1],
        verified,
        evaluations + 1,
        status="verified" if verified.satisfies_constraints else "verification_failed",
    )


def match_schedule_intervention_budget(shape, budget):
    magnitude = shape.integrate_strength()
    if magnitude == 0 or budget > magnitude:
        return None
    return shape.scale_strength(budget / magnitude)


@dataclass(frozen=True)
class RepairConstraintScores:
    """Safe and unsafe distance scores together with their admission cutoffs."""

    safe: float
    unsafe: float
    safe_cutoff: float
    unsafe_cutoff: float | None

    @property
    def satisfies_constraints(self):
        """Output: safe admission and, when enabled, strict unsafe exclusion."""
        return self.safe <= self.safe_cutoff and (
            self.unsafe_cutoff is None or self.unsafe > self.unsafe_cutoff
        )


@dataclass
class RepairScheduleChoice:
    """Selected steering schedule, constraint scores, and search verification details."""

    schedule: Any
    scores: RepairConstraintScores
    evaluations: int
    intervals: list = field(default_factory=list)
    status: str = "verified"

    @property
    def has_intervention(self):
        return self.schedule.integrate_strength() > 0


class TemporalSteeringGeometry:
    """Per-time affine field statistics for schedule scoring and reference pruning."""

    def __init__(self, coefficients, times, calibration, coordinate_count=1):
        self.coefficients = coefficients
        self.times = [float(t) for t in times]
        self.calibration = calibration
        self.coordinate_count = coordinate_count

    def calculate_reference_distances(self, schedule, label):
        """Input: schedule and reference class. Output: distance to every reference signature."""
        strengths = np.asarray([schedule(t) for t in self.times])
        constant, linear, quadratic = self.coefficients[label]
        squared = (constant + linear * strengths + quadratic * strengths**2).sum(axis=1)
        return np.sqrt(np.maximum(squared, 0))

    def select_possible_nearest_references(self, schedule, label, residual):
        """Triangle inequality bounds which references can still be nearest.

        A conservative reduction roundoff allowance broadens the candidate set.
        Large verification residuals naturally fall back to scanning references.
        """
        constant, linear, quadratic = self.coefficients[label]
        rounding_squared = (
            64
            * np.finfo(float).eps
            * self.coordinate_count
            * (np.abs(constant) + np.abs(linear) + np.abs(quadratic)).sum(axis=1)
        )
        radius = residual + float(np.sqrt(rounding_squared.max()))
        distances = self.calculate_reference_distances(schedule, label)
        return np.flatnonzero(distances <= distances.min() + 2 * radius)

    def score_schedule_constraints(self, schedule, mode):
        """Input: schedule and admission mode. Output: safe/unsafe scores and applicable cutoffs."""
        strengths = np.asarray([schedule(t) for t in self.times])
        scores = {}
        for label, (constant, linear, quadratic) in self.coefficients.items():
            squared = (constant + linear * strengths + quadratic * strengths**2).sum(
                axis=1
            )
            scores[label] = float(
                aggregate_anchor_distances(
                    np.sqrt(np.maximum(squared, 0)), {"method": "euclidean"}
                )
            )
        cutoff = (
            self.calibration["unsafe_cutoff"]
            if mode == "safe_attraction_unsafe_avoidance"
            else None
        )
        return RepairConstraintScores(
            scores["safe"], scores["unsafe"], self.calibration["safe_cutoff"], cutoff
        )


@dataclass(frozen=True)
class SteeringSchedule:
    """Bounded steering strengths interpolated by smoothstep over flow time."""

    name: str
    times: tuple
    strengths: tuple

    def __post_init__(self):
        times, strengths = np.asarray(self.times), np.asarray(self.strengths)
        if (
            len(times) < 2
            or times.shape != strengths.shape
            or not np.isfinite(times).all()
            or not np.isfinite(strengths).all()
            or times[0] != 0
            or times[-1] != 1
            or (np.diff(times) <= 0).any()
            or (strengths < 0).any()
            or (strengths > 1).any()
        ):
            raise ValueError(
                "Schedules require ordered [0,1] times and bounded strengths"
            )

    def __call__(self, time):
        if not 0 <= float(time) <= 1:
            raise ValueError("Schedule time must be in [0,1]")
        index = min(
            np.searchsorted(self.times, time, side="right") - 1, len(self.times) - 2
        )
        fraction = (float(time) - self.times[index]) / (
            self.times[index + 1] - self.times[index]
        )
        smooth = fraction * fraction * (3 - 2 * fraction)
        return float(
            (1 - smooth) * self.strengths[index] + smooth * self.strengths[index + 1]
        )

    def scale_strength(self, strength):
        """Input: strength multiplier. Output: a new schedule with scaled knot strengths."""
        return SteeringSchedule(
            self.name, self.times, tuple(float(strength) * x for x in self.strengths)
        )

    def integrate_strength(self):
        """Output: exact integral of steering strength over flow time [0, 1]."""
        # Smoothstep has integral 1/2 on each segment, so this is exact.
        return float(
            np.sum(
                np.diff(self.times)
                * (np.asarray(self.strengths[:-1]) + self.strengths[1:])
                / 2
            )
        )

    def integrate_squared_derivative(self):
        """Output: exact integral of squared strength derivative, used as a smoothness cost."""
        # Integral of squared derivative of a smoothstep interpolant.
        return float(1.2 * np.sum(np.diff(self.strengths) ** 2 / np.diff(self.times)))

    def to_record(self):
        """Output: JSON-compatible schedule record using the persisted artifact schema."""
        return {
            "name": self.name,
            "times": list(self.times),
            "strengths": list(self.strengths),
            "interpolation": "smoothstep",
            "integrated_strength": self.integrate_strength(),
        }


class RepairedVelocityField:
    """One approved condition is held fixed for the entire generation."""

    def __init__(self, adapter, prompt_embeddings, anchor_embeddings, schedule):
        self.adapter = adapter
        self.prompt_embeddings = prompt_embeddings
        self.anchor_embeddings = anchor_embeddings
        self.schedule = schedule
        self.device = adapter.device
        self.model_evaluations = 0

    def evaluate_guided_velocity(self, states, time):
        strength = self.schedule(time)
        accelerated = getattr(self.adapter, "evaluate_repaired_guided_velocity", None)
        if accelerated is not None:
            result = accelerated(
                states, time, self.prompt_embeddings, self.anchor_embeddings, strength
            )
            self.model_evaluations += getattr(
                self.adapter, "last_guided_forward_calls", 1
            )
            return result
        if strength == 0:
            self.model_evaluations += 1
            return self.adapter.evaluate_guided_velocity(
                states, time, self.prompt_embeddings
            )
        if self.anchor_embeddings is None:
            raise ValueError("Nonzero steering requires one approved safe condition")
        if strength == 1:
            self.model_evaluations += 1
            return self.adapter.evaluate_guided_velocity(
                states, time, self.anchor_embeddings
            )
        original = self.adapter.evaluate_guided_velocity(
            states, time, self.prompt_embeddings
        )
        approved = self.adapter.evaluate_guided_velocity(
            states, time, self.anchor_embeddings
        )
        self.model_evaluations += 2
        return (1 - strength) * original + strength * approved

    def evaluate_prompt_effect(self, states, time, embeddings=None):
        strength = self.schedule(time)
        if strength == 0:
            self.model_evaluations += 1
            return self.adapter.evaluate_prompt_effect(
                states, time, self.prompt_embeddings
            )
        if strength == 1:
            self.model_evaluations += 1
            return self.adapter.evaluate_prompt_effect(
                states, time, self.anchor_embeddings
            )
        original = self.adapter.evaluate_prompt_effect(
            states, time, self.prompt_embeddings
        )
        approved = self.adapter.evaluate_prompt_effect(
            states, time, self.anchor_embeddings
        )
        self.model_evaluations += 2
        return (1 - strength) * original + strength * approved
