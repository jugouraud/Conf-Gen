"""Prepare repair inputs through one API for paired or additional-control artifacts."""

import copy
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from backend.src.common.archives import (
    collect_python_sources,
    create_manifested_archive,
)
from backend.src.common.paths import PROJECT_ROOT
from backend.src.common.storage import (
    calculate_file_sha256,
    read_json,
    read_jsonl,
    read_toml_config,
    relocate_path,
    write_json,
)
from backend.src.data.prompts import (
    normalize_prompt_text,
    select_detected_unsafe_repair_cases,
)
from backend.src.detection.calibration import (
    calculate_conformal_cutoff,
    cap_unsafe_cutoff,
)
from backend.src.fields.signatures import (
    FrozenSignatureStore,
    load_signature_space_config,
    read_probe_bank,
)
from backend.src.repair.contracts import PreparedRepair, RepairExperiment
from backend.src.repair.normalization import fit_safe_deviation_normalization


def _validate_repair_settings(settings, mode):
    options, steering = settings["experiment"], settings["steering"]
    if options.get("cohort_selection", "stratified") not in (
        "stratified",
        "detected_unsafe_all_modes",
        "matched_pairs",
        "unsafe_test_only",
    ):
        raise ValueError("Unknown cohort_selection")
    if mode == "paired" and options.get("cohort_selection", "matched_pairs") not in {
        "matched_pairs",
        "unsafe_test_only",
    }:
        raise ValueError(
            "Paired mode requires matched_pairs or unsafe_test_only selection"
        )
    if mode == "controls" and options.get("cohort_selection", "stratified") not in {
        "stratified",
        "detected_unsafe_all_modes",
    }:
        raise ValueError(
            "Controls mode requires stratified or detected_unsafe_all_modes selection"
        )
    if (
        not options["methods"]
        or not set(options["methods"]) <= {f"R{i}" for i in range(6)}
        or "R0" not in options["methods"]
        or len(set(options["methods"])) != len(options["methods"])
    ):
        raise ValueError("Choose unique R0–R5 methods, including paired baseline R0")
    modes = {"safe_attraction", "safe_attraction_unsafe_avoidance"}
    if not options["modes"] or not set(options["modes"]) <= modes:
        raise ValueError("Unknown admission mode")
    for key in (
        "inference_steps",
        "pilot_per_category",
        "tuning_per_category",
        "normalization_limit",
        "threads",
    ):
        if not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not options["generation_seeds"] or any(
        not isinstance(seed, int) or seed < 0 for seed in options["generation_seeds"]
    ):
        raise ValueError("Provide nonnegative generation seeds")
    for key, minimum in (
        ("grid_points", 2),
        ("refinement_steps", 0),
        ("optimization_knots", 2),
        ("optimization_sweeps", 1),
        ("top_k", 1),
    ):
        if not isinstance(steering[key], int) or steering[key] < minimum:
            raise ValueError(f"Invalid {key}")
    if (
        not steering["window_widths"]
        or any(not 0 < w <= 1 for w in steering["window_widths"])
        or not 0 < steering["ramp_fraction"] <= 0.5
        or any(not 0 <= b <= 1 for b in steering["matched_budgets"])
        or not steering["coordinate_steps"]
        or any(not 0 < s <= 1 for s in steering["coordinate_steps"])
        or any(
            not np.isfinite(steering[k]) or steering[k] < 0
            for k in ("smoothness_weight", "constraint_penalty", "unsafe_margin")
        )
    ):
        raise ValueError("Invalid steering search settings")
    return settings


def _resolve_repair_paths(settings):
    paths = {key: relocate_path(value) for key, value in settings["paths"].items()}
    output = paths["output"]
    for key in ("source", "controls"):
        if key not in paths:
            continue
        parent = paths[key]
        if output == parent or output in parent.parents or parent in output.parents:
            raise ValueError(
                "Repair output must be separate from retained input directories"
            )
    return paths


def _read_probe_bank(source):
    config = load_signature_space_config(source, "trajectory")
    bank = read_probe_bank(config)
    bank["rpc_bank_sha256"] = calculate_file_sha256(
        source / "trajectory/probes/probe_bank.pt"
    )
    return config, bank


def _configure_repair_execution(config, settings):
    """Memory/batching overrides preserve the detector's mathematical definition."""
    config = copy.deepcopy(config)
    execution = settings.get("execution", {})
    if "cpu_offload" in execution:
        if type(execution["cpu_offload"]) is not bool:
            raise ValueError("cpu_offload must be a boolean")
        config["model"]["cpu_offload"] = execution["cpu_offload"]
    if "probe_batch_size" in execution:
        batch_size = execution["probe_batch_size"]
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("probe_batch_size must be a positive integer")
        config["experiment"]["probe_batch_size"] = batch_size
    return config


def _read_repair_prompt_splits(paths, manifest):
    source, controls = paths["source"], paths["controls"]
    groups = {
        split: read_jsonl(source / "trajectory/splits" / f"{split}.jsonl")
        for split in manifest["split_ids"]
    }
    groups["additional_safe_test"] = read_jsonl(
        controls / "trajectory/splits/additional_safe_test.jsonl"
    )
    control_manifest = read_json(controls / "experiment.json")
    if control_manifest["parent_experiment_sha256"] != calculate_file_sha256(
        source / "experiment.json"
    ):
        raise ValueError("Controls do not belong to this frozen parent")
    if [row["id"] for row in groups["additional_safe_test"]] != control_manifest[
        "split_ids"
    ]["additional_safe_test"]:
        raise ValueError("Control split order changed")
    return groups


def _read_frozen_distance_tables(paths, manifest):
    source = paths["source"]
    folder = source / "trajectory"
    receipt = read_json(folder / "two_mode_distances.provenance.json")
    if (
        receipt["experiment_sha256"]
        != calculate_file_sha256(source / "experiment.json")
        or receipt["distances_sha256"]
        != calculate_file_sha256(folder / "two_mode_distances.npz")
        or receipt["split_ids"] != manifest["split_ids"]
        or receipt["bank_sha256"]
        != calculate_file_sha256(folder / "probes/probe_bank.pt")
        or receipt["safe_receipt_sha256"]
        != calculate_file_sha256(folder / "safe_distances.provenance.json")
        or receipt["unsafe_receipt_sha256"]
        != calculate_file_sha256(folder / "unsafe_features.provenance.json")
    ):
        raise ValueError("Frozen distance provenance changed")
    with np.load(folder / "two_mode_distances.npz") as archive:
        distances = {key: archive[key].copy() for key in archive.files}
    controls = paths["controls"] / "trajectory"
    receipt = read_json(controls / "additional_distances.provenance.json")
    if (
        receipt["distances_sha256"]
        != calculate_file_sha256(controls / "additional_distances.npz")
        or receipt["bank_sha256"]
        != calculate_file_sha256(folder / "probes/probe_bank.pt")
        or receipt["experiment_sha256"]
        != calculate_file_sha256(paths["controls"] / "experiment.json")
        or receipt["query_receipt_sha256"]
        != calculate_file_sha256(controls / "additional_features.provenance.json")
        or receipt["reference_receipt_sha256"]
        != calculate_file_sha256(controls / "reference_features.provenance.json")
    ):
        raise ValueError("Control distances changed")
    with np.load(controls / "additional_distances.npz") as archive:
        for label in ("safe", "unsafe"):
            distances[f"{label}__additional_safe_test"] = archive[label].copy()
    return distances


def _validate_repair_partitions(groups, manifest, calibration, distances):
    seen_ids, seen_prompts = set(), set()
    for split, rows in groups.items():
        ids = [row["id"] for row in rows]
        prompts = [normalize_prompt_text(row["prompt"]) for row in rows]
        if (
            len(set(ids)) != len(ids)
            or len(set(prompts)) != len(prompts)
            or seen_ids.intersection(ids)
            or seen_prompts.intersection(prompts)
        ):
            raise ValueError("Frozen partitions overlap or contain duplicates")
        if split in manifest["split_ids"] and ids != manifest["split_ids"][split]:
            raise ValueError(f"Frozen split changed: {split}")
        if not rows:
            raise ValueError(f"Missing required partition: {split}")
        permitted_labels = (
            {"benign_intended"} if split == "additional_safe_test" else {"safe"}
        )
        if (split.startswith("safe_") or split == "additional_safe_test") and any(
            row["label"] not in permitted_labels for row in rows
        ):
            raise ValueError("Only safe prompts may enter safe partitions or controls")
        seen_ids.update(ids)
        seen_prompts.update(prompts)
        for label in ("safe", "unsafe"):
            matrix = distances[f"{label}__{split}"]
            if (
                matrix.shape != (len(rows), len(groups[f"{label}_reference"]))
                or not np.isfinite(matrix).all()
                or (matrix < 0).any()
            ):
                raise ValueError("Invalid frozen distance table dimensions or values")
    for label in ("safe", "unsafe"):
        for role, source_role in (
            ("reference", "reference"),
            ("calibration", "validation"),
        ):
            if (
                calibration[f"{label}_{role}_ids"]
                != manifest["split_ids"][f"{label}_{source_role}"]
            ):
                raise ValueError("Calibration reference/partition IDs changed")
        if calibration[f"{label}_rule"]["method"] != "euclidean":
            raise ValueError(
                "Repair requires the retained trajectory_euclidean detector"
            )
    safe_cutoff = calculate_conformal_cutoff(
        distances["safe__safe_validation"].min(axis=1),
        manifest["config"]["experiment"]["alpha"],
    )
    options = manifest["settings"]["experiment"]
    unsafe_conformal = calculate_conformal_cutoff(
        distances["unsafe__unsafe_validation"].min(axis=1), options["unsafe_alpha"]
    )
    unsafe_cutoff, _ = cap_unsafe_cutoff(
        unsafe_conformal,
        distances["unsafe__safe_validation"].min(axis=1),
        options["cap_alpha"],
    )
    if (
        safe_cutoff != calibration["safe_cutoff"]
        or unsafe_cutoff != calibration["unsafe_cutoff"]
    ):
        raise ValueError("Retained calibration thresholds differ from frozen data")
    if (
        not np.isfinite([safe_cutoff, unsafe_cutoff]).all()
        or min(safe_cutoff, unsafe_cutoff) <= 0
    ):
        raise ValueError("Positive finite cutoffs required")


def _bind_frozen_signature_store(paths, bank, groups):
    folder = paths["source"] / "trajectory"
    bank_hash = calculate_file_sha256(folder / "probes/probe_bank.pt")
    safe = read_json(folder / "safe_distances.provenance.json")
    unsafe = read_json(folder / "unsafe_features.provenance.json")
    controls = read_json(
        paths["controls"] / "trajectory/additional_features.provenance.json"
    )
    for receipt in (safe, unsafe, controls):
        if receipt["bank_sha256"] != bank_hash:
            raise ValueError("Feature receipts refer to a different probe bank")
    roots, receipts = {}, {}
    for split in groups:
        is_control = split == "additional_safe_test"
        roots[split] = (
            paths["controls"] if is_control else paths["source"]
        ) / "trajectory"
        receipts[split] = (
            controls if is_control else safe if split.startswith("safe_") else unsafe
        )["features"]
    return FrozenSignatureStore(bank, bank_hash, roots, receipts)


def _fingerprint_repair_inputs(paths, settings):
    files = [
        paths["source"] / "experiment.json",
        paths["source"] / "variants/trajectory_euclidean/calibration.json",
        paths["source"] / "trajectory/probes/probe_bank.pt",
        paths["controls"] / "experiment.json",
    ]
    for folder, names in (
        (
            paths["source"] / "trajectory",
            (
                "safe_distances.provenance.json",
                "unsafe_features.provenance.json",
                "two_mode_distances.provenance.json",
            ),
        ),
        (
            paths["controls"] / "trajectory",
            (
                "additional_features.provenance.json",
                "additional_distances.provenance.json",
            ),
        ),
    ):
        files.extend(folder / name for name in names)
    # Reporting and post-hoc image scoring have independent provenance.
    code = [
        PROJECT_ROOT / name
        for name in (
            "backend/src/repair/prepare.py",
            "backend/src/repair/execute.py",
            "backend/src/common/paths.py",
            "backend/src/repair/contracts.py",
            "backend/src/common/storage.py",
            "backend/src/data/prompts.py",
            "backend/src/fields/signatures.py",
            "backend/src/detection/calibration.py",
            "backend/src/repair/steering.py",
            "backend/src/inference/flow.py",
            "backend/src/common/archives.py",
            "backend/src/repair/normalization.py",
            "backend/src/fields/resident.py",
            "backend/src/repair/artifacts.py",
        )
    ]
    return {
        "settings": settings,
        "inputs": {str(path): calculate_file_sha256(path) for path in files},
        "implementation": {
            path.relative_to(PROJECT_ROOT).as_posix(): calculate_file_sha256(path)
            for path in code
        },
    }


def _write_frozen_repair_manifest(experiment):
    path = experiment.output / "experiment.json"
    manifest = {
        "fingerprint": experiment.fingerprint,
        "normalization": experiment.normalization,
        "detector": "trajectory_euclidean",
        "calibration": experiment.calibration,
        "claim_scope": "Frozen-probe feasibility does not establish image safety",
    }
    if (
        experiment.settings["experiment"].get("cohort_selection")
        == "detected_unsafe_all_modes"
    ):
        cases = select_detected_unsafe_repair_cases(
            experiment.groups,
            experiment.distances,
            experiment.calibration,
            experiment.settings["experiment"],
            "pilot",
        )
        manifest["selected_cohort"] = [
            {
                **row,
                "split": split,
                "source_index": index,
                "cached_safe_score": float(
                    experiment.distances[f"safe__{split}"][index].min()
                ),
                "cached_unsafe_score": float(
                    experiment.distances[f"unsafe__{split}"][index].min()
                ),
            }
            for split, index, row in cases
        ]
    if path.exists() and read_json(path) != manifest:
        raise ValueError(
            "Repair inputs, configuration, or implementation changed; use a fresh output"
        )
    write_json(path, manifest)


def _read_frozen_deviation_normalization(experiment):
    path = experiment.output / "experiment.json"
    if not path.exists():
        return None
    manifest = read_json(path)
    if manifest["fingerprint"] != experiment.fingerprint:
        raise ValueError(
            "Repair inputs, configuration, or implementation changed; use a fresh output"
        )
    return manifest["normalization"]


def load_frozen_repair_experiment(settings, paths):
    manifest = read_json(paths["source"] / "experiment.json")
    calibration = read_json(
        paths["source"] / "variants/trajectory_euclidean/calibration.json"
    )
    config, bank = _read_probe_bank(paths["source"])
    config = _configure_repair_execution(config, settings)
    groups = _read_repair_prompt_splits(paths, manifest)
    distances = _read_frozen_distance_tables(paths, manifest)
    _validate_repair_partitions(groups, manifest, calibration, distances)
    features = _bind_frozen_signature_store(paths, bank, groups)
    fingerprint = _fingerprint_repair_inputs(paths, settings)
    return RepairExperiment(
        settings,
        paths["output"],
        paths["source"],
        config,
        bank,
        calibration,
        groups,
        distances,
        features,
        fingerprint,
    )


def freeze_repair_inputs(experiment):
    experiment.normalization = _read_frozen_deviation_normalization(experiment)
    if experiment.normalization is None:
        experiment.normalization = fit_safe_deviation_normalization(experiment)
    _write_frozen_repair_manifest(experiment)
    return experiment


def select_matched_pairs(groups, per_category, seed):
    """Select scenarios before variants, without inspecting detection or image scores."""
    safe = {
        row["pair_id"]: (index, row) for index, row in enumerate(groups["safe_test"])
    }
    unsafe = {
        row["pair_id"]: (index, row) for index, row in enumerate(groups["unsafe_test"])
    }
    if set(safe) != set(unsafe):
        raise ValueError("Safe and unsafe test pairs differ")
    strata = defaultdict(lambda: defaultdict(list))
    for pair, (_, row) in safe.items():
        other = unsafe[pair][1]
        if any(
            row[key] != other[key] for key in ("category", "scenario_id", "variant_id")
        ):
            raise ValueError("Matched pair metadata differs")
        strata[row["category"]][row["scenario_id"]].append(pair)
    rng = random.Random(seed)
    selected = []
    for category, scenarios in sorted(strata.items()):
        if len(scenarios) < per_category:
            raise ValueError(f"Insufficient held-out scenarios in {category}")
        for scenario in sorted(rng.sample(sorted(scenarios), per_category)):
            pair = rng.choice(sorted(scenarios[scenario]))
            for label, lookup in (("safe", safe), ("unsafe", unsafe)):
                index, row = lookup[pair]
                selected.append(
                    {**row, "split": f"{label}_test", "source_index": index}
                )
    return selected


def select_repair_cases(groups, options):
    policy = options.get("cohort_selection", "matched_pairs")
    if policy not in {"matched_pairs", "unsafe_test_only"}:
        raise ValueError("Unknown paired repair selection policy")
    selected = select_matched_pairs(
        groups, options["pairs_per_category"], options["seed"]
    )
    if policy == "unsafe_test_only":
        selected = [row for row in selected if row["split"] == "unsafe_test"]
    return selected


def _prepare_paired_artifacts(settings: dict, output: Path) -> dict:
    """Freeze copied paired inputs and the source bundle for worker reconstruction."""
    source = relocate_path(settings["paths"]["source"])
    if output.exists():
        raise ValueError("Preparation requires a fresh output directory")
    parent = read_json(source / "experiment.json")
    for name, digest in {
        **parent["implementation"],
        **parent.get("experiment_implementation", {}),
    }.items():
        if calculate_file_sha256(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Parent implementation changed: {name}")
    groups = {}
    for split, digest in parent["split_sha256"].items():
        path = source / "trajectory/splits" / f"{split}.jsonl"
        if calculate_file_sha256(path) != digest:
            raise ValueError("Frozen parent split changed")
        groups[split] = read_jsonl(path)
    selected = select_repair_cases(groups, settings["experiment"])
    unsafe_only = settings["experiment"].get("cohort_selection") == "unsafe_test_only"
    pair_count = len({row["pair_id"] for row in selected})
    category_count = len({row["category"] for row in selected})
    expectations = settings.get("cohort", {})
    if pair_count != expectations.get(
        "expected_pairs", pair_count
    ) or category_count != expectations.get("expected_categories", category_count):
        raise ValueError(
            "Selected cohort differs from the configured experiment design"
        )
    calibration = read_json(source / "scores/trajectory/euclidean/calibration.json")
    if calibration["fit_policy"] != "validation_only":
        raise ValueError("Only validation-fitted thresholds may be retained")
    inputs = {}
    names = [
        "experiment.json",
        "trajectory/distances.npz",
        "trajectory/distances.provenance.json",
        "scores/trajectory/euclidean/calibration.json",
        *[f"trajectory/splits/{split}.jsonl" for split in groups],
    ]
    for name in names:
        path = source / name
        inputs[name] = calculate_file_sha256(path)
    provenance = read_json(source / "trajectory/distances.provenance.json")
    if (
        provenance["distances_sha256"] != inputs["trajectory/distances.npz"]
        or provenance["experiment_sha256"] != inputs["experiment.json"]
        or provenance["bank_sha256"]
        != read_json(source / "banks.provenance.json")["banks"]["trajectory"]
    ):
        raise ValueError("Parent distances changed")
    for name in names:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    files = collect_python_sources()
    code = {name: calculate_file_sha256(path) for name, path in files.items()}
    manifest = {
        "config": {"colab": settings["colab"]},
        "fingerprint": {"settings": settings, "inputs": inputs, "implementation": code},
        "detector": "trajectory_euclidean",
        "calibration": calibration,
        "parent_bank_sha256": read_json(source / "banks.provenance.json")["banks"][
            "trajectory"
        ],
        "selected_cohort": selected,
        "expected_pairs": pair_count,
        "expected_prompts": len(selected),
        "expected_generation_records": len(selected)
        * len(settings["experiment"]["modes"])
        * len(settings["experiment"]["methods"])
        * len(settings["experiment"]["generation_seeds"]),
        "selection_policy": f"{settings['experiment']['pairs_per_category']} distinct held-out scenarios per category; seeded variant; {'unsafe test members only' if unsafe_only else 'both pair members'}; no outcome filtering",
        "normalization_policy": f"fit on first {settings['experiment']['normalization_limit']} safe-validation prompts only; numerical values retained separately",
        "reconstruction_policy": "regenerate lost parent probes and fields; compare all rebuilt query/reference distances with retained parent values before any image generation",
        "claim_scope": "Frozen-probe feasibility and independent image-policy scores are separate outcomes",
    }
    if settings["execution"].get("optimized_gpu"):
        recovery = read_json(relocate_path(settings["paths"]["recovery"]))
        recovered = Path(recovery["source_directory"])
        bank = recovered / "trajectory/probes/probe_bank.pt"
        required = {
            row["id"]
            for split in ("safe_reference", "unsafe_reference")
            for row in groups[split]
        }
        required.update(
            row["id"]
            for row in groups["safe_validation"][
                : settings["experiment"]["normalization_limit"]
            ]
        )
        required.update(row["id"] for row in selected)
        archive = Path(recovery["archive"])
        if calculate_file_sha256(archive) != recovery["archive_sha256"]:
            raise ValueError("Recovered archive changed")
        with zipfile.ZipFile(archive) as snapshot:
            recovered_files = json.loads(snapshot.read("artifact_manifest.json"))
        allowed = {}
        prefixes = (
            "source/trajectory/probes/",
            "source/optimization/baseline.pt",
            "source/optimization/null_embeddings.pt",
            "source/optimization/trajectory.null.pt",
            "source/optimization/trajectory.compatibility.json",
        )
        for name, digest in recovered_files.items():
            if name.startswith(prefixes):
                allowed[name] = digest
            elif (
                name.startswith("source/optimization/embeddings/")
                and Path(name).stem in required
            ):
                allowed[name] = digest
            elif name.startswith("source/trajectory/features/"):
                prompt_id = Path(name).name.split(".")[0]
                if prompt_id in required:
                    allowed[name] = digest
        manifest["reuse"] = {
            "archive_sha256": recovery["archive_sha256"],
            "bank_sha256": calculate_file_sha256(bank),
            "files": allowed,
        }
    write_json(output / "experiment.json", manifest)
    write_json(output / "selection.json", {"cases": selected})
    for path in output.rglob("*"):
        if path.is_file():
            files[f"results/{path.relative_to(output).as_posix()}"] = path
    for name in ("pyproject.toml", "uv.lock"):
        files[name] = PROJECT_ROOT / name
    archive = output.parent / "gpu-inputs.zip"
    if settings["execution"].get("optimized_gpu"):
        archive = (
            Path.home()
            / ".cache/thesis-paired-repair-packages"
            / f"{output.parent.name}-{output.name}.zip"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
    create_manifested_archive(archive, files)
    write_json(
        output / "validation.json",
        {
            "status": "passed",
            "parent_sha256": inputs["experiment.json"],
            "pairs": pair_count,
            "prompts": len(selected),
            "distinct_scenarios": len({row["scenario_id"] for row in selected}),
            "selection_uses_image_outcomes": False,
            "parent_calibration_unchanged": True,
            "package_sha256": calculate_file_sha256(archive),
            "package_path": str(archive),
            "package_files": len(files),
        },
    )
    if settings["execution"].get("optimized_gpu"):
        write_json(
            output / "gpu/reuse_upload.json",
            {
                "archive": recovery["archive"],
                "parts_directory": str(Path(recovery["archive"]).parent),
                "metadata": str(Path(recovery["archive"]).parent / "metadata.json"),
            },
        )
    return {
        "pairs": pair_count,
        "prompts": len(selected),
        "records": manifest["expected_generation_records"],
    }


def run_repair_preparation(
    config: Path | dict, output: Path | None = None, *, mode: str = "paired"
) -> PreparedRepair:
    """Bind frozen detector inputs in paired or controls mode without refitting cutoffs.

    Paired mode publishes a reconstruction/worker bundle and explicit paired cohort.
    Controls mode binds saved fields plus separate benign controls for local execution.
    The returned record has one stable shape; only controls mode includes a runtime context.
    """
    if mode not in {"paired", "controls"}:
        raise ValueError("Repair preparation mode must be paired or controls")
    settings = (
        read_toml_config(config)
        if not isinstance(config, dict)
        else copy.deepcopy(config)
    )
    if output is not None:
        settings["paths"]["output"] = str(Path(output).resolve())
    _validate_repair_settings(settings, mode)
    paths = _resolve_repair_paths(settings)
    torch.set_num_threads(settings["experiment"]["threads"])
    if mode == "paired":
        summary = _prepare_paired_artifacts(settings, paths["output"])
        return PreparedRepair(mode, paths["output"], summary)
    experiment = load_frozen_repair_experiment(settings, paths)
    freeze_repair_inputs(experiment)
    return PreparedRepair(
        mode,
        paths["output"],
        {"prompts": sum(map(len, experiment.groups.values()))},
        experiment,
    )
