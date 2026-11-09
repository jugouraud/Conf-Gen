"""H1: search, verify, and sample paired repairs for one explicit cohort."""

import copy
import importlib
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from backend.src.common.storage import calculate_file_sha256, read_json, write_json
from backend.src.data.prompts import select_detected_unsafe_repair_cases
from backend.src.fields.signatures import (
    calculate_model_signature,
    evaluate_probe_field,
)
from backend.src.inference.flow import FlowModelAdapter, decode_latents_to_image
from backend.src.repair.contracts import RepairExperiment, RepairSearchResult
from backend.src.repair.steering import (
    RepairConstraintScores,
    RepairedVelocityField,
    RepairScheduleChoice,
    build_constant_steering_schedule,
    build_strategy_steering_schedules,
    build_temporal_steering_geometry,
    calculate_normalized_deviation_profiles,
    evaluate_repaired_probes,
    match_schedule_intervention_budget,
    optimize_schedule_knots,
    sample_repaired_trajectory,
    score_verified_field_constraints,
    search_schedule_strength,
    select_nearest_approved_anchor,
)


def select_execution_cases(experiment, stage):
    options = experiment.settings["experiment"]
    if options.get("cohort_selection") == "detected_unsafe_all_modes":
        return select_detected_unsafe_repair_cases(
            experiment.groups,
            experiment.distances,
            experiment.calibration,
            options,
            stage,
        )
    splits = (
        ("safe_validation", "unsafe_validation")
        if stage == "tune"
        else ("safe_test", "unsafe_test", "additional_safe_test")
    )
    limit = (
        options["tuning_per_category"]
        if stage == "tune"
        else options["pilot_per_category"]
    )
    rng = np.random.default_rng(options["seed"])
    selected = []
    for split in splits:
        categories = (
            options.get(
                "categories",
                tuple(
                    dict.fromkeys(row["category"] for row in experiment.groups[split])
                ),
            )
            if split.startswith("unsafe") or split == "additional_safe_test"
            else ("safe",)
        )
        for category in categories:
            indexes = [
                i
                for i, row in enumerate(experiment.groups[split])
                if row.get("category", "safe") == category or split.startswith("safe_")
            ]
            if not indexes:
                raise ValueError(f"No prompts in required stratum {split}/{category}")
            rng.shuffle(indexes)
            if stage != "full":
                indexes = indexes[:limit]
            selected.extend(
                (split, i, experiment.groups[split][i]) for i in sorted(indexes)
            )
    return selected


def _validate_stage_prerequisites(experiment, stage):
    if stage != "full":
        return
    for prerequisite in ("tune", "pilot"):
        status = experiment.output / prerequisite / "status.json"
        if not status.exists() or read_json(status).get("status") != "complete":
            raise ValueError(f"Complete {prerequisite} before full held-out evaluation")


def _load_generation_model(experiment):
    if experiment.adapter is not None:
        adapter = experiment.adapter
    else:
        config = copy.deepcopy(experiment.config)
        config["model"]["revision"] = experiment.bank["metadata"]["revision_sha"]
        adapter = FlowModelAdapter(config)
    if adapter.signature != calculate_model_signature(experiment.bank["metadata"]):
        raise ValueError("Generation model differs from the frozen detector model")
    pipe = getattr(adapter, "pipe", None)
    if pipe is not None:
        for name in (
            "transformer",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
        ):
            component = getattr(pipe, name, None)
            if component is not None:
                component.requires_grad_(False)
    return adapter


def _load_image_evaluator(experiment):
    if experiment.evaluator is not None:
        return experiment.evaluator
    settings = experiment.settings["evaluation"]
    if not settings["factory"]:
        return None
    module_name, function_name = settings["factory"].split(":", 1)
    evaluator = getattr(importlib.import_module(module_name), function_name)(settings)
    if not callable(getattr(evaluator, "evaluate_image", None)) or not getattr(
        evaluator, "metadata", None
    ):
        raise ValueError(
            "Image evaluator must expose evaluate_image() and versioned metadata"
        )
    return evaluator


def _freeze_evaluator_metadata(experiment):
    evaluator = experiment.evaluator
    metadata = (
        evaluator.metadata if evaluator is not None else {"status": "not_evaluated"}
    )
    if metadata.get("model_id") == experiment.bank["metadata"]["model_id"]:
        raise ValueError("The image safety evaluator must use an independent model")
    path = experiment.output / "image_evaluator.json"
    receipt = {"metadata": metadata}
    factory = experiment.settings["evaluation"]["factory"]
    if factory:
        module = importlib.import_module(factory.split(":", 1)[0])
        receipt["factory_source_sha256"] = calculate_file_sha256(Path(module.__file__))
    if path.exists() and read_json(path) != receipt:
        raise ValueError("Image evaluator changed; use a fresh output")
    write_json(path, receipt)


def _read_cached_case(experiment, stage, split, row):
    path = experiment.output / stage / "records" / f"{split}_{row['id']}.json"
    if not path.exists():
        return None
    record = read_json(path)
    if record["experiment_sha256"] != calculate_file_sha256(
        experiment.output / "experiment.json"
    ):
        raise ValueError("Case receipt belongs to a different experiment")
    for relative, expected in record["artifacts"].items():
        artifact = experiment.output / relative
        if not artifact.is_file() or calculate_file_sha256(artifact) != expected:
            raise ValueError(f"Generated artifact changed: {artifact}")
    return record["rows"]


def _measure_original_field(experiment, embeddings, split, row):
    started = time.perf_counter()
    field = evaluate_probe_field(
        experiment.config, experiment.adapter, experiment.bank, embeddings
    )
    cached = experiment.features.load_signature(split, row)
    matches = torch.allclose(field, cached, rtol=1e-3, atol=1e-3)
    policy = experiment.settings.get("execution", {}).get(
        "fresh_field_policy", "require_close"
    )
    if policy not in ("require_close", "record_drift"):
        raise ValueError("Unknown fresh-field reproducibility policy")
    if not matches and policy == "require_close":
        raise ValueError(
            "Fresh prompt field differs from the retained extraction; cannot reuse detection decisions"
        )
    batches = experiment.bank["time_points"] * int(
        np.ceil(
            experiment.bank["trajectories"]
            / experiment.config["experiment"].get("probe_batch_size", 1)
        )
    )
    batches = getattr(experiment.adapter, "frozen_probe_forward_calls", batches)
    delta = field.double() - cached.double()
    return field, {
        "fresh_field_policy": policy,
        "matches_cached_field": bool(matches),
        "cached_field_rms_difference": float(delta.square().mean().sqrt()),
        "cached_field_max_difference": float(delta.abs().max()),
        "model_evaluations": batches,
        "seconds": time.perf_counter() - started,
        "sharing": "once per prompt, shared by modes and variants",
    }


def _read_detection_scores(experiment, split, index, mode):
    safe = float(experiment.distances[f"safe__{split}"][index].min())
    unsafe = float(experiment.distances[f"unsafe__{split}"][index].min())
    cutoff = (
        experiment.calibration["unsafe_cutoff"]
        if mode == "safe_attraction_unsafe_avoidance"
        else None
    )
    return RepairConstraintScores(
        safe, unsafe, experiment.calibration["safe_cutoff"], cutoff
    )


def _measure_anchor_field(experiment, embeddings, anchor):
    field = evaluate_probe_field(
        experiment.config, experiment.adapter, experiment.bank, embeddings
    )
    cached = experiment.features.load_signature("safe_reference", anchor)
    matches = torch.allclose(field, cached, rtol=1e-3, atol=1e-3)
    policy = experiment.settings.get("execution", {}).get(
        "fresh_field_policy", "require_close"
    )
    if not matches and policy == "require_close":
        raise ValueError(
            "Fresh safe field differs from the retained approved reference"
        )
    delta = field.double() - cached.double()
    return field, {
        "matches_cached_field": bool(matches),
        "fresh_field_policy": policy,
        "cached_field_rms_difference": float(delta.square().mean().sqrt()),
        "cached_field_max_difference": float(delta.abs().max()),
    }


def _search_strategy_shapes(experiment, geometry, shapes, mode):
    settings = experiment.settings["steering"]
    choices, timings = [], {}
    for shape in shapes:
        started = time.perf_counter()
        choices.append(
            search_schedule_strength(
                shape,
                lambda schedule: geometry.score_schedule_constraints(schedule, mode),
                settings,
            )
        )
        method = shape.name.split("_")[0]
        timings[method] = timings.get(method, 0.0) + time.perf_counter() - started
    if "R5" in experiment.settings["experiment"]["methods"]:
        started = time.perf_counter()
        optimized = optimize_schedule_knots(
            lambda schedule: geometry.score_schedule_constraints(schedule, mode),
            settings,
            choices,
        )
        choices.append(optimized)
        timings["R5"] = time.perf_counter() - started
    geometry.method_search_seconds = timings
    return choices


def _construct_budget_choices(experiment, geometry, shapes, choices, mode):
    variants = []
    optimized = [
        choice.schedule
        for choice in choices
        if choice.schedule.name == "R5" and choice.scores.satisfies_constraints
    ]
    for shape in shapes + optimized:
        for budget in experiment.settings["steering"]["matched_budgets"]:
            schedule = match_schedule_intervention_budget(shape, budget)
            if schedule is not None:
                variants.append(
                    (
                        RepairScheduleChoice(
                            schedule,
                            geometry.score_schedule_constraints(schedule, mode),
                            1,
                            status="matched_budget",
                        ),
                        f"budget_{budget:g}",
                    )
                )
    return variants


def _select_generation_variants(experiment, variants, stats):
    policy = experiment.settings.get("execution", {}).get("generation_policy", "all")
    if policy == "all":
        return variants, stats
    if policy != "minimum_per_method":
        raise ValueError("Unknown generation policy")
    candidates = [
        {
            "variant": choice.schedule.name,
            "schedule": choice.schedule.to_record(),
            "safe_score": choice.scores.safe,
            "unsafe_score": choice.scores.unsafe,
            "admissible": choice.scores.satisfies_constraints,
            "status": choice.status,
        }
        for choice, budget in variants
        if budget == "minimum"
    ]
    selected = []
    for method in experiment.settings["experiment"]["methods"]:
        matches = [
            (choice, budget)
            for choice, budget in variants
            if budget == "minimum" and choice.schedule.name.split("_")[0] == method
        ]
        feasible = [
            (choice, budget)
            for choice, budget in matches
            if choice.scores.satisfies_constraints
        ]
        selected.append(
            min(
                feasible or matches,
                key=lambda pair: pair[0].schedule.integrate_strength(),
            )
        )
    return selected, {
        **stats,
        "generation_policy": policy,
        "searched_variants": candidates,
    }


def _construct_unchanged_choices(experiment, original, reason):
    # Preserve the full variant design for unflagged prompts and anchor failures.
    empty = {
        key: np.zeros(experiment.bank["time_points"])
        for key in ("max", "mean", "top10", "top25")
    }
    shapes = build_strategy_steering_schedules(
        experiment.settings["experiment"]["methods"],
        experiment.bank,
        empty,
        experiment.settings["steering"],
    )
    names = [shape.name for shape in shapes]
    if "R5" in experiment.settings["experiment"]["methods"]:
        names.append("R5")
    variants = [
        (
            RepairScheduleChoice(
                build_constant_steering_schedule(0.0, name), original, 0, status=reason
            ),
            "minimum",
        )
        for name in names
    ]
    for shape in shapes:
        for budget in experiment.settings["steering"]["matched_budgets"]:
            if match_schedule_intervention_budget(shape, budget) is not None:
                variants.append(
                    (
                        RepairScheduleChoice(
                            build_constant_steering_schedule(0.0, shape.name),
                            original,
                            0,
                            status=reason,
                        ),
                        f"budget_{budget:g}",
                    )
                )
    return variants


def _verify_schedule(
    experiment, choice, embeddings, anchor_embeddings, mode, geometry_context
):
    started = time.perf_counter()
    query, anchor, geometry = geometry_context
    policy = experiment.settings.get("execution", {}).get(
        "probe_verification", "direct_model"
    )
    if policy not in ("direct_model", "fresh_affine_with_global_check"):
        raise ValueError("Unknown frozen-probe verification policy")
    field = RepairedVelocityField(
        experiment.adapter, embeddings, anchor_embeddings, choice.schedule
    )
    if policy == "fresh_affine_with_global_check" and choice.schedule.name != "R1":
        times = experiment.bank["times"]
        dimensions = (-1,) + (1,) * (query.ndim - 1)
        original_weights = torch.tensor(
            [1 - choice.schedule(float(t)) for t in times], dtype=query.dtype
        ).reshape(dimensions)
        anchor_weights = torch.tensor(
            [choice.schedule(float(t)) for t in times], dtype=anchor.dtype
        ).reshape(dimensions)
        values = original_weights * query + anchor_weights * anchor
        basis = (
            "exact affine identity of fresh model fields; no cached signature steering"
        )
    else:
        values = evaluate_repaired_probes(experiment.config, field, experiment.bank)
        basis = "direct model reevaluation"
    scores, verification = score_verified_field_constraints(
        values, query, anchor, choice.schedule, geometry, experiment, mode
    )
    verification["basis"] = basis
    return scores, field.model_evaluations, time.perf_counter() - started, verification


def _sample_generation(experiment, embeddings, anchor_embeddings, schedule, seed):
    field = RepairedVelocityField(
        experiment.adapter, embeddings, anchor_embeddings, schedule
    )
    return sample_repaired_trajectory(
        field, seed, experiment.settings["experiment"]["inference_steps"]
    )


def _decode_generation(experiment, latents):
    if hasattr(experiment.adapter, "decode_latents_to_image"):
        return experiment.adapter.decode_latents_to_image(latents)
    return decode_latents_to_image(experiment.adapter, latents)


def _assess_generated_image(experiment, image, original, prompt):
    if experiment.evaluator is None:
        return {
            "status": "not_evaluated",
            "unsafe": None,
            "semantic_preservation": None,
            "quality": None,
            "artifacts": None,
            "evaluator": None,
        }
    result = dict(
        experiment.evaluator.evaluate_image(
            image=image, prompt=prompt, original_image=original
        )
    )
    if type(result.get("unsafe")) is not bool:
        raise ValueError(
            "Independent image evaluator must return a boolean unsafe decision"
        )
    for key in ("semantic_preservation", "quality", "artifacts"):
        value = result.get(key)
        if value is not None and (
            not isinstance(value, (int, float)) or not np.isfinite(value)
        ):
            raise ValueError(f"Evaluator returned an invalid {key} metric")
    result.update(status="evaluated", evaluator=experiment.evaluator.metadata)
    return result


def _measure_image_similarity(image, original):
    values, reference = (
        np.asarray(image).astype(float) / 255.0,
        np.asarray(original).astype(float) / 255.0,
    )
    return {
        "pixel_rmse": float(np.sqrt(np.mean((values - reference) ** 2))),
        "meaning": "paired pixel difference; not a semantic or image-quality metric",
    }


def _save_generation(experiment, identity, image, latents):
    path = experiment.output / identity
    path.parent.mkdir(parents=True, exist_ok=True)
    image_path = path.parent / (path.name + ".png")
    latent_path = path.parent / (path.name + ".pt")
    image.save(image_path)
    torch.save(latents, latent_path)
    return {
        "image": image_path.relative_to(experiment.output).as_posix(),
        "latents": latent_path.relative_to(experiment.output).as_posix(),
    }


def _package_generation_record(
    context,
    choice,
    budget,
    scores,
    verification_calls,
    generated,
    evaluation,
    similarity,
    paths,
):
    experiment = context["experiment"]
    split, _index, row = context["case"]
    search = context["search"]
    return {
        "id": row["id"],
        "split": split,
        "category": row.get("category", "safe"),
        "prompt": row["prompt"],
        "cohort": context["stage"],
        "mode": context["mode"],
        "seed": context["seed"],
        "method": choice.schedule.name.split("_")[0],
        "variant": choice.schedule.name.split("_")[0]
        if experiment.settings.get("execution", {}).get("generation_policy")
        == "minimum_per_method"
        else choice.schedule.name,
        "selected_schedule_variant": choice.schedule.name,
        "budget": budget,
        "flagged": not search.original.satisfies_constraints,
        "anchor_id": search.anchor["id"] if search.anchor else None,
        "schedule": choice.schedule.to_record(),
        "search_status": choice.status,
        "search_evaluations": choice.evaluations,
        "feasible_intervals": choice.intervals,
        "search": search.stats,
        "original_scores": {
            "safe": search.original.safe,
            "unsafe": search.original.unsafe,
        },
        "frozen_probe": {
            "safe_score": scores.safe,
            "unsafe_score": scores.unsafe,
            "safe_admission": scores.safe <= scores.safe_cutoff,
            "unsafe_exclusion": scores.unsafe > experiment.calibration["unsafe_cutoff"],
            "admissible": scores.satisfies_constraints,
            "direct_verification_model_evaluations": verification_calls,
        },
        "independent_probe": {"status": "not_evaluated"},
        "deviation_profiles": {
            key: np.asarray(value).tolist() for key, value in search.profiles.items()
        },
        "probe_times": experiment.bank["midpoints"].tolist(),
        "generation": generated,
        "image_evaluation": evaluation,
        "image_similarity": similarity,
        **paths,
    }


def _save_prompt_repair_records(experiment, stage, split, row, records):
    artifacts = {record[key] for record in records for key in ("image", "latents")}
    receipt = {
        "experiment_sha256": calculate_file_sha256(
            experiment.output / "experiment.json"
        ),
        "rows": records,
        "artifacts": {
            relative: calculate_file_sha256(experiment.output / relative)
            for relative in sorted(artifacts)
        },
    }
    path = experiment.output / stage / "records" / f"{split}_{row['id']}.json"
    temporary = path.with_suffix(".tmp")
    write_json(temporary, receipt)
    os.replace(temporary, path)


def _announce_completed_case(stage, case, records):
    split, _index, row = case
    print(
        f"{stage}: {split}/{row['id']}: {len(records)} paired generation records",
        flush=True,
    )


def _create_baseline_schedule_choice(original):
    return RepairScheduleChoice(
        build_constant_steering_schedule(), original, 0, status="baseline"
    )


def _measure_search_cost(experiment, started, choices, budgets):
    count = experiment.bank["time_points"] * int(
        np.ceil(
            experiment.bank["trajectories"]
            / experiment.config["experiment"].get("probe_batch_size", 1)
        )
    )
    return {
        "seconds": time.perf_counter() - started,
        "candidate_score_evaluations": sum(choice.evaluations for choice in choices)
        + len(budgets),
        "fresh_query_probe_model_evaluations": getattr(
            experiment.adapter, "frozen_probe_forward_calls", count
        ),
        "fresh_anchor_probe_model_evaluations": getattr(
            experiment.adapter, "frozen_probe_forward_calls", count
        ),
        "search_basis": "exact affine geometry of fresh model fields at frozen states",
    }


def _reject_unverified_schedule(choice, original):
    return RepairScheduleChoice(
        build_constant_steering_schedule(0.0, choice.schedule.name),
        original,
        choice.evaluations,
        choice.intervals,
        "direct_verification_failed",
    )


def _write_repair_execution_status(experiment, stage, status, **details):
    write_json(experiment.output / stage / "status.json", {"status": status, **details})


def _check_compute_availability(experiment):
    device = experiment.config["model"]["device"]
    return (
        experiment.adapter is not None
        or not device.startswith("cuda")
        or torch.cuda.is_available()
    )


def initialize_repair_execution(experiment, stage):
    _validate_stage_prerequisites(experiment, stage)
    available = _check_compute_availability(experiment)
    if not available:
        _write_repair_execution_status(
            experiment,
            stage,
            "blocked",
            reason="CUDA unavailable; real image pilot has not run",
        )
        return False
    experiment.adapter = _load_generation_model(experiment)
    experiment.evaluator = _load_image_evaluator(experiment)
    _freeze_evaluator_metadata(experiment)
    _write_repair_execution_status(experiment, stage, "running")
    return True


def prepare_prompt_repair(experiment, stage, case):
    split, _index, row = case
    cached = _read_cached_case(experiment, stage, split, row)
    if cached is not None:
        return cached, None, None
    embeddings = experiment.adapter.encode_prompt(row["prompt"])
    query = _measure_original_field(experiment, embeddings, split, row)
    return None, embeddings, query


def search_prompt_repairs(experiment, case, query, mode):
    query, query_cost = query
    split, index, _row = case
    original = _read_detection_scores(experiment, split, index, mode)
    baseline = _create_baseline_schedule_choice(original)
    started = time.perf_counter()
    if original.satisfies_constraints:
        anchor = select_nearest_approved_anchor(experiment, split, index, mode)
        profiles = {}
        if anchor is not None:
            anchor_field = experiment.features.load_signature("safe_reference", anchor)
            profiles = calculate_normalized_deviation_profiles(
                query, anchor_field, experiment.bank, experiment.normalization
            )
        variants, stats = _select_generation_variants(
            experiment,
            [(baseline, "minimum")]
            + _construct_unchanged_choices(experiment, original, "accepted_unchanged"),
            {"query_probe": query_cost},
        )
        return RepairSearchResult(
            original=original,
            anchor=anchor,
            profiles=profiles,
            variants=variants,
            stats=stats,
        )
    anchor = select_nearest_approved_anchor(experiment, split, index, mode)
    if anchor is None:
        variants, stats = _select_generation_variants(
            experiment,
            [(baseline, "minimum")]
            + _construct_unchanged_choices(experiment, original, "no_approved_anchor"),
            {"query_probe": query_cost},
        )
        return RepairSearchResult(
            original=original, anchor=None, profiles={}, variants=variants, stats=stats
        )
    anchor_embeddings = experiment.adapter.encode_prompt(anchor["prompt"])
    anchor_field, anchor_drift = _measure_anchor_field(
        experiment, anchor_embeddings, anchor
    )
    geometry = build_temporal_steering_geometry(query, anchor_field, experiment)
    profiles = calculate_normalized_deviation_profiles(
        query, anchor_field, experiment.bank, experiment.normalization
    )
    shapes = build_strategy_steering_schedules(
        experiment.settings["experiment"]["methods"],
        experiment.bank,
        profiles,
        experiment.settings["steering"],
    )
    choices = _search_strategy_shapes(experiment, geometry, shapes, mode)
    budgets = _construct_budget_choices(experiment, geometry, shapes, choices, mode)
    variants = (
        [(baseline, "minimum")] + [(choice, "minimum") for choice in choices] + budgets
    )
    stats = _measure_search_cost(experiment, started, choices, budgets)
    stats["method_search_seconds"] = geometry.method_search_seconds
    stats["geometry_seconds"] = getattr(geometry, "geometry_seconds", None)
    stats["query_probe"] = query_cost
    stats["anchor_probe_drift"] = anchor_drift
    fresh_original = geometry.score_schedule_constraints(baseline.schedule, mode)
    baseline.scores = fresh_original
    stats["fresh_original_scores"] = {
        "safe": fresh_original.safe,
        "unsafe": fresh_original.unsafe,
        "admissible": fresh_original.satisfies_constraints,
    }
    variants, stats = _select_generation_variants(experiment, variants, stats)
    return RepairSearchResult(
        original=original,
        anchor=anchor,
        anchor_embeddings=anchor_embeddings,
        profiles=profiles,
        variants=variants,
        stats=stats,
        geometry_context=(query, anchor_field, geometry),
    )


def generate_prompt_comparison(experiment, stage, case, embeddings, search, mode):
    split, _index, row = case
    records = []
    baseline = _create_baseline_schedule_choice(search.original)
    for seed in experiment.settings["experiment"]["generation_seeds"]:
        baseline_latents, baseline_stats = _sample_generation(
            experiment, embeddings, None, baseline.schedule, seed
        )
        baseline_image = _decode_generation(experiment, baseline_latents)
        for choice, budget in search.variants:
            verification_calls = 0
            verification_seconds = 0.0
            verification_details = {}
            if choice.has_intervention:
                (
                    scores,
                    verification_calls,
                    verification_seconds,
                    verification_details,
                ) = _verify_schedule(
                    experiment,
                    choice,
                    embeddings,
                    search.anchor_embeddings,
                    mode,
                    search.geometry_context,
                )
                if not scores.satisfies_constraints and budget == "minimum":
                    choice = _reject_unverified_schedule(choice, baseline.scores)
                    scores = baseline.scores
            else:
                scores = choice.scores
            if not choice.has_intervention:
                latents, generated, image = (
                    baseline_latents,
                    dict(baseline_stats),
                    baseline_image,
                )
                generated["reused_baseline"] = choice.schedule.name != "R0"
            else:
                latents, generated = _sample_generation(
                    experiment,
                    embeddings,
                    search.anchor_embeddings,
                    choice.schedule,
                    seed,
                )
                image = _decode_generation(experiment, latents)
            generated["direct_verification_seconds"] = verification_seconds
            generated["direct_verification"] = verification_details
            evaluation = _assess_generated_image(
                experiment, image, baseline_image, row["prompt"]
            )
            similarity = _measure_image_similarity(image, baseline_image)
            identity = f"{stage}/generations/{split}/{row['id']}/{mode}/{choice.schedule.name}_{budget}_{seed}"
            paths = _save_generation(experiment, identity, image, latents)
            context = {
                "experiment": experiment,
                "stage": stage,
                "case": case,
                "mode": mode,
                "seed": seed,
                "search": search,
            }
            records.append(
                _package_generation_record(
                    context,
                    choice,
                    budget,
                    scores,
                    verification_calls,
                    generated,
                    evaluation,
                    similarity,
                    paths,
                )
            )
    return records


def persist_prompt_comparison(experiment, stage, case, records):
    split, _index, row = case
    _save_prompt_repair_records(experiment, stage, split, row, records)
    _announce_completed_case(stage, case, records)


def complete_repair_execution(experiment, stage, records):
    _write_repair_execution_status(
        experiment,
        stage,
        "complete",
        generation_records=len(records),
        image_evaluation="complete"
        if experiment.evaluator is not None
        else "not_evaluated",
    )
    return records


def run_repair_execution(
    experiment: RepairExperiment,
    stage: str,
    *,
    cases: Sequence[tuple[str, int, dict]] | None = None,
    on_case_completed: Callable[[tuple[str, int, dict], list[dict], int, int], None]
    | None = None,
) -> list[dict]:
    """Search and sample an explicit cohort, committing one receipt per case.

    Cases retain their source split/index order. The optional completion hook
    runs after a verified cached or newly persisted receipt, including on resume.
    Hook failures fail the stage; a later run verifies receipts before reusing them.
    """
    if stage not in {"tune", "pilot", "full"}:
        raise ValueError("Unknown repair stage")
    try:
        cases = list(
            select_execution_cases(experiment, stage) if cases is None else cases
        )
        allowed_splits = (
            {"safe_validation", "unsafe_validation"}
            if stage == "tune"
            else {"safe_test", "unsafe_test", "additional_safe_test"}
        )
        seen = set()
        for split, index, row in cases:
            if split not in allowed_splits or type(index) is not int or index < 0:
                raise ValueError("Repair case is outside the requested stage")
            rows = experiment.groups[split]
            if index >= len(rows) or any(
                row.get(key) != value for key, value in rows[index].items()
            ):
                raise ValueError("Repair case differs from its frozen split/index")
            if (
                row.get("split", split) != split
                or row.get("source_index", index) != index
            ):
                raise ValueError(
                    "Repair case selection metadata differs from its split/index"
                )
            identity = (split, row["id"])
            if identity in seen:
                raise ValueError("Duplicate repair case")
            seen.add(identity)
        if not initialize_repair_execution(experiment, stage):
            return []
        records = []
        for completed, case in enumerate(cases, 1):
            cached, embeddings, query = prepare_prompt_repair(experiment, stage, case)
            if cached is not None:
                records.extend(cached)
                if on_case_completed is not None:
                    on_case_completed(case, cached, completed, len(cases))
                continue
            prompt_records = []
            for mode in experiment.settings["experiment"]["modes"]:
                search = search_prompt_repairs(experiment, case, query, mode)
                prompt_records.extend(
                    generate_prompt_comparison(
                        experiment, stage, case, embeddings, search, mode
                    )
                )
            persist_prompt_comparison(experiment, stage, case, prompt_records)
            records.extend(prompt_records)
            if on_case_completed is not None:
                on_case_completed(case, prompt_records, completed, len(cases))
        return complete_repair_execution(experiment, stage, records)
    except Exception as error:
        _write_repair_execution_status(
            experiment,
            stage,
            "failed",
            reason=str(error),
            error_type=type(error).__name__,
        )
        raise
