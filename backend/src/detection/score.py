"""H1: calibrate and score frozen detector artifacts without test-dependent fitting."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from backend.src.common.paths import PROJECT_ROOT
from backend.src.common.storage import (
    calculate_file_sha256,
    calculate_source_fingerprints,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
)
from backend.src.data.definitions import SAFE_SPLITS, SPLIT, UNSAFE_SPLITS
from backend.src.detection.calibration import (
    aggregate_anchor_distances,
    calculate_conformal_cutoff,
    calculate_detection_metrics,
    calibrate_paired_detection_thresholds,
    cap_unsafe_cutoff,
    evaluate_paired_detection_tests,
    fit_distance_rule,
)
from backend.src.detection.contracts import (
    QUERY_SPLITS,
    SPACE_METHODS,
    read_frozen_control_inputs,
    read_resident_detection_inputs,
)
from backend.src.detection.definitions import DISTANCE_METHODS, MODES
from backend.src.detection.rendering import (
    complete_probe_detection_results,
    complete_scoring_comparison,
    complete_source_comparison,
    write_control_comparison_report,
    write_detection_comparison_report,
)
from backend.src.fields.constraints import steer_query_signatures
from backend.src.fields.definitions import VARIANTS
from backend.src.fields.distances import calculate_streamed_reference_distances
from backend.src.fields.signatures import (
    load_signature_space_config,
    load_weighted_signature_vector,
    read_probe_bank,
    resolve_signature_path,
)


def _load_detection_score_inputs(output, space):
    manifest = read_json(output / "experiment.json")
    folder = output / space
    archive = folder / "two_mode_distances.npz"
    receipt_path = folder / "two_mode_distances.provenance.json"
    receipt = read_json(receipt_path)
    if (
        receipt["experiment_sha256"]
        != calculate_file_sha256(output / "experiment.json")
        or receipt["distances_sha256"] != calculate_file_sha256(archive)
        or receipt["split_ids"] != manifest["split_ids"]
    ):
        raise ValueError("Distance provenance differs from the frozen experiment")
    with np.load(archive) as data:
        safe = {s: data[f"safe__{s}"].copy() for s in SAFE_SPLITS + UNSAFE_SPLITS}
        unsafe = {s: data[f"unsafe__{s}"].copy() for s in SAFE_SPLITS + UNSAFE_SPLITS}
    groups = {
        s: read_jsonl(folder / "splits" / f"{s}.jsonl")
        for s in SAFE_SPLITS + UNSAFE_SPLITS
    }
    if any(
        [r["id"] for r in rows] != manifest["split_ids"][s]
        for s, rows in groups.items()
    ):
        raise ValueError("Frozen split order changed")
    return SimpleNamespace(
        output=output,
        space=space,
        manifest=manifest,
        safe=safe,
        unsafe=unsafe,
        groups=groups,
    )


def _calibrate_detection_method(inputs, method):
    safe, unsafe, manifest = inputs.safe, inputs.unsafe, inputs.manifest
    settings = manifest["settings"]["experiment"]
    safe_spec = fit_distance_rule(
        method, safe["safe_reference"], manifest["bandwidth_scale"]
    )
    unsafe_spec = fit_distance_rule(
        method, unsafe["unsafe_reference"], manifest["bandwidth_scale"]
    )
    safe_cutoff = calculate_conformal_cutoff(
        aggregate_anchor_distances(safe["safe_validation"], safe_spec),
        manifest["config"]["experiment"]["alpha"],
    )
    unsafe_cutoff = calculate_conformal_cutoff(
        aggregate_anchor_distances(unsafe["unsafe_validation"], unsafe_spec),
        settings["unsafe_alpha"],
    )
    unsafe_cap, cap_rank = cap_unsafe_cutoff(
        unsafe_cutoff,
        aggregate_anchor_distances(unsafe["safe_validation"], unsafe_spec),
        settings["cap_alpha"],
    )
    if not (
        np.isfinite([safe_cutoff, unsafe_cutoff, unsafe_cap]).all()
        and min(safe_cutoff, unsafe_cap) > 0
    ):
        raise ValueError("Normalized joint scores require positive finite cutoffs")
    return {
        "safe_cutoff": safe_cutoff,
        "unsafe_conformal_cutoff": unsafe_cutoff,
        "unsafe_cutoff": unsafe_cap,
        "cap_rank": cap_rank,
        "safe_rule": safe_spec,
        "unsafe_rule": unsafe_spec,
        "safe_calibration_ids": manifest["split_ids"]["safe_validation"],
        "unsafe_calibration_ids": manifest["split_ids"]["unsafe_validation"],
        "safe_reference_ids": manifest["split_ids"]["safe_reference"],
        "unsafe_reference_ids": manifest["split_ids"]["unsafe_reference"],
    }


def _measure_detection_method_scores(inputs, calibration):
    safe, unsafe, groups = inputs.safe, inputs.unsafe, inputs.groups
    safe_spec, unsafe_spec = calibration["safe_rule"], calibration["unsafe_rule"]
    s = np.concatenate(
        [
            aggregate_anchor_distances(safe[split], safe_spec)
            for split in ("safe_test", "unsafe_test")
        ]
    )
    u = np.concatenate(
        [
            aggregate_anchor_distances(unsafe[split], unsafe_spec)
            for split in ("safe_test", "unsafe_test")
        ]
    )
    labels = np.r_[
        np.zeros(len(groups["safe_test"])), np.ones(len(groups["unsafe_test"]))
    ]
    return s, u, labels


def _score_detection_mode(inputs, method, mode, calibration, scores):
    safe, unsafe, groups, manifest = (
        inputs.safe,
        inputs.unsafe,
        inputs.groups,
        inputs.manifest,
    )
    space = inputs.space
    safe_spec, unsafe_spec = calibration["safe_rule"], calibration["unsafe_rule"]
    safe_cutoff, unsafe_cap = calibration["safe_cutoff"], calibration["unsafe_cutoff"]
    s, u, labels = scores
    cutoff = None if mode == "safe_attraction" else unsafe_cap
    flags = s > safe_cutoff
    rank_scores = s / safe_cutoff
    if cutoff is not None:
        flags |= u <= cutoff
        rank_scores = np.maximum(rank_scores, 2 - u / cutoff)
    detection = calculate_detection_metrics(labels, flags, rank_scores)
    records, steering = steer_query_signatures(
        safe,
        unsafe,
        groups,
        safe_spec,
        unsafe_spec,
        safe_cutoff,
        cutoff,
        manifest["settings"]["steering"],
    )
    for r, flag, rank_score in zip(records, flags, rank_scores, strict=True):
        r.update(flagged=bool(flag), ranking_score=float(rank_score))
        r["inside_safe_after"] = r["safe_score"] <= safe_cutoff
        r["outside_unsafe_after"] = r["unsafe_score"] > unsafe_cap
        r["joint_conditions_after"] = (
            r["inside_safe_after"] and r["outside_unsafe_after"]
        )
    for split, summary in steering.items():
        rr = [r for r in records if r["split"] == split]
        summary.update(
            post_safe_membership_rate=float(
                np.mean([r["inside_safe_after"] for r in rr])
            ),
            post_unsafe_exclusion_rate=float(
                np.mean([r["outside_unsafe_after"] for r in rr])
            ),
            post_joint_stop_rate=float(
                np.mean([r["joint_conditions_after"] for r in rr])
            ),
        )
    # Explicit per-category metrics preserve overlap labels in the split
    # files, while evaluation strata use each row's unique assignment.
    category_rates = {
        category: float(
            np.mean(
                [
                    r["flagged"]
                    for r in records
                    if r["split"] == "unsafe_test" and r["category"] == category
                ]
            )
        )
        for category in sorted({row["category"] for row in groups["unsafe_test"]})
    }
    result = {
        "space": space,
        "method": method,
        "mode": mode,
        **detection,
        "unsafe_flagged_by_category": category_rates,
        "steering": steering,
        "calibration": calibration,
    }
    return result, records


def _persist_detection_mode(inputs, method, mode, result, records):
    variant = inputs.output / "variants" / f"{inputs.space}_{method}"
    write_csv(variant / mode / "decisions_and_steering.csv", records)
    write_json(variant / mode / "report.json", result)
    print(
        f"{inputs.space}/{method}/{mode}: safe FAR {result['safe_false_positive_rate']:.3f}, "
        f"unsafe flagged {result['unsafe_flagged_rate']:.3f}",
        flush=True,
    )


def score_signature_space(output, space):
    inputs = _load_detection_score_inputs(output, space)
    results = {}
    for method in VARIANTS[space]:
        calibration = _calibrate_detection_method(inputs, method)
        scores = _measure_detection_method_scores(inputs, calibration)
        write_json(
            inputs.output
            / "variants"
            / f"{inputs.space}_{method}"
            / "calibration.json",
            calibration,
        )
        for mode in MODES:
            result, records = _score_detection_mode(
                inputs, method, mode, calibration, scores
            )
            _persist_detection_mode(inputs, method, mode, result, records)
            results[f"{space}_{method}__{mode}"] = result
    return results


def load_paired_detection_scores(output):
    output = Path(output).resolve()
    manifest = read_json(output / "experiment.json")
    receipt = read_json(output / "trajectory/distances.provenance.json")
    target = output / "trajectory/distances.npz"
    if receipt["experiment_sha256"] != calculate_file_sha256(
        output / "experiment.json"
    ) or receipt["distances_sha256"] != calculate_file_sha256(target):
        raise ValueError("Distance provenance changed")
    groups = {}
    for split in manifest["counts"]:
        path = output / "trajectory/splits" / f"{split}.jsonl"
        if calculate_file_sha256(path) != manifest["split_sha256"][split]:
            raise ValueError("Split bytes changed")
        groups[split] = read_jsonl(path)
    with np.load(target) as archive:
        tables = {key: archive[key].copy() for key in archive.files}
    return output, manifest, groups, tables


def freeze_validation_detection_thresholds(output, manifest, tables):
    calibration = calibrate_paired_detection_thresholds(
        tables, manifest["config"]["experiment"]
    )
    calibration.update(
        {
            "reference_ids": {
                s: ids
                for s, ids in manifest["split_ids"].items()
                if s.endswith("reference")
            },
            "calibration_ids": {
                s: ids
                for s, ids in manifest["split_ids"].items()
                if s.endswith("validation")
            },
        }
    )
    write_json(output / "calibration.json", calibration)
    return calibration


def evaluate_held_out_paired_detection(output, groups, tables, calibration):
    reports, decisions = evaluate_paired_detection_tests(tables, groups, calibration)
    write_json(
        output / "report.json",
        {
            "detector": "trajectory_euclidean",
            "evaluation_split": "test",
            "calibration": calibration,
            "modes": reports,
            "scope": "CSV prompt labels; image safety unmeasured",
        },
    )
    write_csv(output / "test_decisions.csv", decisions)
    write_json(
        output / "report.provenance.json",
        {
            "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
            "distance_receipt_sha256": calculate_file_sha256(
                output / "trajectory/distances.provenance.json"
            ),
            "artifacts": {
                name: calculate_file_sha256(output / name)
                for name in ("calibration.json", "report.json", "test_decisions.csv")
            },
        },
    )
    write_json(
        output / "analysis_status.json",
        {"status": "complete", "phase": "detection_complete"},
    )
    return reports


def load_probe_scoring_inputs(output):
    output = Path(output).resolve()
    spaces = {}
    for space in VARIANTS:
        root = output / space
        measurement = read_json(root / "experiment.json")
        receipt = read_json(root / "distances.provenance.json")
        if (
            measurement["space"] != space
            or receipt["experiment_sha256"]
            != calculate_file_sha256(root / "experiment.json")
            or receipt["distances_sha256"]
            != calculate_file_sha256(root / "distances.npz")
            or receipt["probe_aggregation"] != measurement["probe_aggregation"]
        ):
            raise ValueError("Frozen probe-distance provenance changed")
        source = Path(measurement["source"])
        manifest = read_json(source / "experiment.json")
        if (
            calculate_file_sha256(source / "experiment.json")
            != measurement["source_experiment_sha256"]
            or calculate_file_sha256(source / space / "features.provenance.json")
            != measurement["features_receipt_sha256"]
            or calculate_file_sha256(source / space / "probes/probe_bank.pt")
            != measurement["bank_sha256"]
        ):
            raise ValueError("Frozen signature provenance changed")
        groups = {}
        for split, ids in manifest["split_ids"].items():
            path = source / space / "splits" / f"{split}.jsonl"
            if calculate_file_sha256(path) != manifest["split_sha256"][split]:
                raise ValueError("Frozen scoring split changed")
            groups[split] = read_jsonl(path)
            if [row["id"] for row in groups[split]] != ids:
                raise ValueError("Frozen scoring IDs changed")
        with np.load(root / "distances.npz") as archive:
            tables = {key: archive[key].copy() for key in archive.files}
        expected = {
            f"{label}__{split}"
            for label in ("safe", "unsafe")
            for split in (
                "safe_validation",
                "unsafe_validation",
                "safe_test",
                "unsafe_test",
                f"{label}_reference",
            )
        }
        if tables.keys() != expected:
            raise ValueError("Incomplete query/training distance tables")
        for key, table in tables.items():
            label, split = key.split("__")
            if (
                table.shape != (len(groups[split]), len(groups[f"{label}_reference"]))
                or not np.isfinite(table).all()
                or (table < 0).any()
            ):
                raise ValueError("Invalid probe distance table")
        spaces[space] = SimpleNamespace(
            root=root,
            measurement=measurement,
            manifest=manifest,
            groups=groups,
            tables=tables,
        )
    trajectory, source = spaces["trajectory"], spaces["source_slice"]
    if (
        trajectory.groups != source.groups
        or trajectory.manifest["config"]["experiment"]
        != source.manifest["config"]["experiment"]
    ):
        raise ValueError(
            "Five-method comparison requires identical splits and calibration settings"
        )
    aggregations = {item.measurement["probe_aggregation"] for item in spaces.values()}
    if len(aggregations) != 1:
        raise ValueError("Cannot combine different probe aggregations")
    provenance = {
        "probe_aggregation": aggregations.pop(),
        "distance_receipts": {
            space: calculate_file_sha256(item.root / "distances.provenance.json")
            for space, item in spaces.items()
        },
        "bandwidth_fit": "Full within-class training pairs only; median scale 1.0",
        "threshold_fit": "validation_only",
        "evaluation": "test_only",
        "scope": "detection_only",
        "implementation": calculate_source_fingerprints(
            [
                "backend/src/detection/score.py",
                "backend/src/data/paired.py",
                "backend/src/detection/calibration.py",
            ]
        ),
    }
    target = output / "experiment.json"
    if target.exists() and read_json(target) != provenance:
        raise ValueError("Existing scoring provenance differs; use a fresh output")
    write_json(target, provenance)
    return SimpleNamespace(
        output=output, spaces=spaces, aggregation=provenance["probe_aggregation"]
    )


def fit_probe_training_rules(inputs):
    rules = {}
    for space, item in inputs.spaces.items():
        rules[space] = {}
        for label in ("safe", "unsafe"):
            matrix = item.tables[f"{label}__{label}_reference"]
            if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-7):
                raise ValueError("Asymmetric training-reference geometry")
            rules[space][label] = {
                method: fit_distance_rule(method, matrix, 1.0)
                for method in VARIANTS[space]
            }
    return rules


def _evaluate_probe_method(inputs, item, space, method, rules):
    calibration = calibrate_paired_detection_thresholds(
        item.tables,
        item.manifest["config"]["experiment"],
        safe_rule=rules[space]["safe"][method],
        unsafe_rule=rules[space]["unsafe"][method],
    )
    calibration.update(
        safe_rule=rules[space]["safe"][method],
        unsafe_rule=rules[space]["unsafe"][method],
        probe_aggregation=inputs.aggregation,
        reference_ids={
            s: ids
            for s, ids in item.manifest["split_ids"].items()
            if s.endswith("reference")
        },
        calibration_ids={
            s: ids
            for s, ids in item.manifest["split_ids"].items()
            if s.endswith("validation")
        },
    )
    modes, decisions = evaluate_paired_detection_tests(
        item.tables, item.groups, calibration
    )
    root = item.root / "scores" / method
    report = {
        "detector": f"{space}_{method}",
        "probe_aggregation": inputs.aggregation,
        "evaluation_split": "test",
        "calibration": calibration,
        "modes": modes,
    }
    write_json(root / "calibration.json", calibration)
    write_json(root / "report.json", report)
    write_csv(root / "test_decisions.csv", decisions)
    return report


def evaluate_probe_detection_methods(inputs, rules):
    return {
        f"{space}_{method}": _evaluate_probe_method(inputs, item, space, method, rules)
        for space, item in inputs.spaces.items()
        for method in VARIANTS[space]
    }


def _score_resident_detection(output: Path) -> list[dict]:
    """Fit training rules and validation cutoffs; publish held-out RMS metrics."""
    output = Path(output).resolve()
    manifest, groups = read_resident_detection_inputs(output)
    summary = []
    for space, methods in SPACE_METHODS.items():
        path = output / space / "distances.npz"
        receipt = read_json(path.with_suffix(".provenance.json"))
        if receipt["experiment_sha256"] != calculate_file_sha256(
            output / "experiment.json"
        ) or receipt["distances_sha256"] != calculate_file_sha256(path):
            raise ValueError("Frozen distance provenance changed")
        with np.load(path) as archive:
            tables = {k: archive[k].copy() for k in archive.files}
        for key, matrix in tables.items():
            label, split = key.split("__")
            if matrix.shape != (len(groups[split]), len(groups[f"{label}_reference"])):
                raise ValueError("Distance table shape changed")
        for method in methods:
            rules = {
                label: fit_distance_rule(
                    method, tables[f"{label}__{label}_reference"], 1.0
                )
                for label in ("safe", "unsafe")
            }
            calibration = calibrate_paired_detection_thresholds(
                tables,
                manifest["config"]["experiment"],
                safe_rule=rules["safe"],
                unsafe_rule=rules["unsafe"],
            )
            calibration.update(
                safe_rule=rules["safe"],
                unsafe_rule=rules["unsafe"],
                calibration_ids={
                    s: manifest["split_ids"][s]
                    for s in QUERY_SPLITS
                    if s.endswith("validation")
                },
            )
            target = output / "scores" / space / method
            write_json(target / "calibration.json", calibration)
            reports, decisions = evaluate_paired_detection_tests(
                tables, groups, calibration
            )
            write_json(
                target / "report.json",
                {
                    "space": space,
                    "method": method,
                    "modes": reports,
                    "scope": "Prompt-label detection; image safety unmeasured",
                },
            )
            write_csv(target / "test_decisions.csv", decisions)
            summary.extend(
                {
                    "space": space,
                    "method": method,
                    "mode": mode,
                    **{
                        k: metrics[k]
                        for k in (
                            "safe_false_positive_rate",
                            "unsafe_flagged_rate",
                            "auroc_prompt_labels",
                            "auprc_prompt_labels",
                        )
                    },
                }
                for mode, metrics in reports.items()
            )
    write_json(
        output / "comparison.json", {"results": summary, "probe_aggregation": "rms"}
    )
    write_csv(output / "comparison.csv", summary)
    write_json(
        output / "status.json", {"status": "complete", "phase": "detection_complete"}
    )
    return summary


def load_source_scoring_inputs(output):
    output = Path(output).resolve()
    manifest = read_json(output / "experiment.json")
    root = output / "source_slice"
    receipt = read_json(root / "distances.provenance.json")
    features = read_json(root / "features.provenance.json")
    if (
        receipt["experiment_sha256"]
        != calculate_file_sha256(output / "experiment.json")
        or receipt["distances_sha256"] != calculate_file_sha256(root / "distances.npz")
        or receipt["bank_sha256"]
        != calculate_file_sha256(root / "probes/probe_bank.pt")
        or receipt["features_receipt_sha256"]
        != calculate_file_sha256(root / "features.provenance.json")
        or features["experiment_sha256"] != receipt["experiment_sha256"]
        or features["bank_sha256"] != receipt["bank_sha256"]
    ):
        raise ValueError("Source distance provenance changed")
    groups = {}
    for split, ids in manifest["split_ids"].items():
        path = root / "splits" / f"{split}.jsonl"
        if calculate_file_sha256(path) != manifest["split_sha256"][split]:
            raise ValueError("Frozen source split changed")
        groups[split] = read_jsonl(path)
        if [row["id"] for row in groups[split]] != ids:
            raise ValueError("Frozen source split order changed")
    with np.load(root / "distances.npz") as archive:
        tables = {key: archive[key].copy() for key in archive.files}
    for key, values in tables.items():
        label, split = key.split("__")
        if values.shape != (len(groups[split]), len(groups[f"{label}_reference"])):
            raise ValueError("Invalid source distance dimensions")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid source distances")
    return SimpleNamespace(
        output=output, manifest=manifest, groups=groups, tables=tables
    )


def fit_source_training_rules(inputs):
    rules = {}
    for label in ("safe", "unsafe"):
        matrix = inputs.tables[f"{label}__{label}_reference"]
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-7):
            raise ValueError("Asymmetric source training geometry")
        rules[label] = {
            method: fit_distance_rule(method, matrix, 1.0)
            for method in ("anchor_static", "anchor_adaptive")
        }
    return rules


def _evaluate_source_rule(inputs, rules, method):
    calibration = calibrate_paired_detection_thresholds(
        inputs.tables,
        inputs.manifest["config"]["experiment"],
        safe_rule=rules["safe"][method],
        unsafe_rule=rules["unsafe"][method],
    )
    calibration.update(
        safe_rule=rules["safe"][method],
        unsafe_rule=rules["unsafe"][method],
        reference_ids={
            s: ids
            for s, ids in inputs.manifest["split_ids"].items()
            if s.endswith("reference")
        },
        calibration_ids={
            s: ids
            for s, ids in inputs.manifest["split_ids"].items()
            if s.endswith("validation")
        },
    )
    reports, decisions = evaluate_paired_detection_tests(
        inputs.tables, inputs.groups, calibration
    )
    root = inputs.output / "scores" / method
    write_json(root / "calibration.json", calibration)
    report = {
        "detector": f"source_slice_{method}",
        "evaluation_split": "test",
        "calibration": calibration,
        "modes": reports,
    }
    write_json(root / "report.json", report)
    write_csv(root / "test_decisions.csv", decisions)
    return report


def evaluate_source_detection_rules(inputs, rules):
    return {
        method: _evaluate_source_rule(inputs, rules, method)
        for method in ("anchor_static", "anchor_adaptive")
    }


def load_scoring_comparison_inputs(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output in source.parents:
        raise ValueError("Use a distinct scoring-comparison output")
    manifest = read_json(source / "experiment.json")
    archive = source / "trajectory/distances.npz"
    receipt = read_json(source / "trajectory/distances.provenance.json")
    features = read_json(source / "trajectory/features.provenance.json")
    bank_path = source / "trajectory/probes/probe_bank.pt"
    if (
        receipt["experiment_sha256"]
        != calculate_file_sha256(source / "experiment.json")
        or receipt["distances_sha256"] != calculate_file_sha256(archive)
        or receipt["features_receipt_sha256"]
        != calculate_file_sha256(source / "trajectory/features.provenance.json")
        or receipt["bank_sha256"] != calculate_file_sha256(bank_path)
        or features["bank_sha256"] != receipt["bank_sha256"]
        or features["experiment_sha256"] != receipt["experiment_sha256"]
    ):
        raise ValueError("Frozen source provenance changed")
    groups = {}
    for split, ids in manifest["split_ids"].items():
        path = source / "trajectory/splits" / f"{split}.jsonl"
        if calculate_file_sha256(path) != manifest["split_sha256"][split]:
            raise ValueError("Source split changed")
        groups[split] = read_jsonl(path)
        if [row["id"] for row in groups[split]] != ids:
            raise ValueError("Source split order changed")
    with np.load(archive) as data:
        tables = {key: data[key].copy() for key in data.files}
    for key, values in tables.items():
        label, split = key.split("__")
        if values.shape != (len(groups[split]), len(groups[f"{label}_reference"])):
            raise ValueError("Distance matrix shape changed")
    provenance = {
        "source_experiment_sha256": receipt["experiment_sha256"],
        "source_distance_receipt_sha256": calculate_file_sha256(
            source / "trajectory/distances.provenance.json"
        ),
        "source_features_receipt_sha256": receipt["features_receipt_sha256"],
        "bank_sha256": receipt["bank_sha256"],
        "methods": list(DISTANCE_METHODS),
        "bandwidth_scale": 1.0,
        "bandwidth_policy": "Full within-class training pairwise median; no validation/test fit",
        "implementation": {
            p: calculate_file_sha256(PROJECT_ROOT / p)
            for p in (
                "backend/src/detection/score.py",
                "backend/src/fields/distances.py",
                "backend/src/data/paired.py",
                "backend/src/detection/calibration.py",
            )
        },
    }
    if (output / "experiment.json").exists() and read_json(
        output / "experiment.json"
    ) != provenance:
        raise ValueError("Existing comparison provenance differs; use a fresh output")
    write_json(output / "experiment.json", provenance)
    write_json(
        output / "status.json",
        {"status": "running", "phase": "loaded_frozen_distances"},
    )
    return SimpleNamespace(
        source=source,
        output=output,
        manifest=manifest,
        groups=groups,
        tables=tables,
        features=features,
        bank_hash=receipt["bank_sha256"],
    )


def fit_training_adaptive_geometry(inputs, working_directory):
    cfg = load_signature_space_config(inputs.source, "trajectory")
    bank = read_probe_bank(cfg)
    matrices = {}
    for label in ("safe", "unsafe"):
        target = inputs.output / "training_geometry" / f"{label}.npz"
        receipt = target.with_suffix(".provenance.json")
        if target.exists() and receipt.exists():
            saved = read_json(receipt)
            if saved["comparison_sha256"] != calculate_file_sha256(
                inputs.output / "experiment.json"
            ) or saved["distances_sha256"] != calculate_file_sha256(target):
                raise ValueError("Training geometry provenance changed")
            with np.load(target) as data:
                matrices[label] = data["distances"].copy()
            continue
        split = f"{label}_reference"

        def load_vector(row):
            path = resolve_signature_path(cfg, split, row["id"])
            if (
                calculate_file_sha256(path)
                != inputs.features["features"][f"{split}/{row['id']}.pt"]
            ):
                raise ValueError("Training signature bytes changed")
            return load_weighted_signature_vector(
                cfg, split, row, bank, inputs.bank_hash
            )

        def progress(phase, done, total):
            write_json(
                inputs.output / "status.json",
                {
                    "status": "running",
                    "phase": f"{label}/{phase}",
                    "completed": done,
                    "total": total,
                },
            )

        matrices[label] = calculate_streamed_reference_distances(
            inputs.groups[split],
            load_vector,
            working_directory=working_directory,
            progress=progress,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, distances=matrices[label])
        write_json(
            receipt,
            {
                "comparison_sha256": calculate_file_sha256(
                    inputs.output / "experiment.json"
                ),
                "distances_sha256": calculate_file_sha256(target),
                "reference_ids": inputs.manifest["split_ids"][split],
                "implementation": "FP64 feature-block Gram accumulation from unchanged weighted signatures",
            },
        )
    return matrices


def _evaluate_scoring_method(inputs, method, matrices):
    safe_rule = fit_distance_rule(method, matrices.get("safe"), 1.0)
    unsafe_rule = fit_distance_rule(method, matrices.get("unsafe"), 1.0)
    calibration = calibrate_paired_detection_thresholds(
        inputs.tables,
        inputs.manifest["config"]["experiment"],
        safe_rule=safe_rule,
        unsafe_rule=unsafe_rule,
    )
    calibration.update(
        safe_rule=safe_rule,
        unsafe_rule=unsafe_rule,
        reference_ids={
            s: ids
            for s, ids in inputs.manifest["split_ids"].items()
            if s.endswith("reference")
        },
        calibration_ids={
            s: ids
            for s, ids in inputs.manifest["split_ids"].items()
            if s.endswith("validation")
        },
    )
    root = inputs.output / method
    write_json(root / "calibration.json", calibration)
    reports, decisions = evaluate_paired_detection_tests(
        inputs.tables, inputs.groups, calibration
    )
    write_json(
        root / "report.json",
        {
            "detector": f"trajectory_{method}",
            "evaluation_split": "test",
            "calibration": calibration,
            "modes": reports,
        },
    )
    write_csv(root / "test_decisions.csv", decisions)
    return {"method": method, "modes": reports}


def evaluate_reused_distance_methods(inputs):
    return {
        method: _evaluate_scoring_method(inputs, method, {})
        for method in ("euclidean", "anchor_static")
    }


def _load_control_score_inputs(output, space, frozen_inputs):
    manifest, parent, frozen = frozen_inputs
    folder = output / space
    if (
        calculate_file_sha256(folder / "frozen_geometry.npz")
        != manifest["frozen_geometry_sha256"][space]
    ):
        raise ValueError("Parent geometry changed")
    receipt = read_json(folder / "additional_distances.provenance.json")
    if receipt["experiment_sha256"] != calculate_file_sha256(
        output / "experiment.json"
    ) or receipt["distances_sha256"] != calculate_file_sha256(
        folder / "additional_distances.npz"
    ):
        raise ValueError("Additional distance provenance changed")
    with np.load(folder / "additional_distances.npz") as data:
        query = {key: data[key].copy() for key in ("safe", "unsafe")}
    with np.load(folder / "frozen_geometry.npz") as data:
        geometry = {key: data[key].copy() for key in data.files}
    groups = {
        s: read_jsonl(folder / "splits" / f"{s}.jsonl")
        for s in ("safe_reference", "unsafe_test", SPLIT)
    }
    if [r["id"] for r in groups[SPLIT]] != manifest["split_ids"][SPLIT]:
        raise ValueError("Additional test ordering changed")
    safe = {"safe_reference": geometry["safe__safe_reference"], SPLIT: query["safe"]}
    unsafe = {
        "safe_reference": geometry["unsafe__safe_reference"],
        SPLIT: query["unsafe"],
    }
    return SimpleNamespace(
        output=output,
        space=space,
        manifest=manifest,
        parent=parent,
        frozen=frozen,
        query=query,
        geometry=geometry,
        groups=groups,
        safe=safe,
        unsafe=unsafe,
    )


def _measure_control_method_scores(inputs, method):
    space, frozen, query, geometry = (
        inputs.space,
        inputs.frozen,
        inputs.query,
        inputs.geometry,
    )
    name = f"{space}_{method}"
    calibration = frozen[name]
    safe_rule, unsafe_rule = calibration["safe_rule"], calibration["unsafe_rule"]
    safe_cutoff, unsafe_cutoff = (
        calibration["safe_cutoff"],
        calibration["unsafe_cutoff"],
    )
    s, u = (
        aggregate_anchor_distances(query["safe"], safe_rule),
        aggregate_anchor_distances(query["unsafe"], unsafe_rule),
    )
    original_s = aggregate_anchor_distances(geometry["safe__unsafe_test"], safe_rule)
    original_u = aggregate_anchor_distances(
        geometry["unsafe__unsafe_test"], unsafe_rule
    )
    return SimpleNamespace(
        name=name,
        calibration=calibration,
        safe_rule=safe_rule,
        unsafe_rule=unsafe_rule,
        safe_cutoff=safe_cutoff,
        unsafe_cutoff=unsafe_cutoff,
        s=s,
        u=u,
        original_s=original_s,
        original_u=original_u,
    )


def _score_control_mode(inputs, method, mode, scores):
    space, manifest, parent = inputs.space, inputs.manifest, inputs.parent
    groups, safe, unsafe = inputs.groups, inputs.safe, inputs.unsafe
    name, calibration = scores.name, scores.calibration
    safe_rule, unsafe_rule = scores.safe_rule, scores.unsafe_rule
    safe_cutoff, unsafe_cutoff = scores.safe_cutoff, scores.unsafe_cutoff
    s, u, original_s, original_u = (
        scores.s,
        scores.u,
        scores.original_s,
        scores.original_u,
    )
    cutoff = None if mode == "safe_attraction" else unsafe_cutoff
    flags, original_flags = s > safe_cutoff, original_s > safe_cutoff
    ranking, original_ranking = s / safe_cutoff, original_s / safe_cutoff
    if cutoff is not None:
        flags |= u <= cutoff
        original_flags |= original_u <= cutoff
        ranking, original_ranking = (
            np.maximum(ranking, 2 - u / cutoff),
            np.maximum(original_ranking, 2 - original_u / cutoff),
        )
    old = parent["methods"][name + "__" + mode]
    if float(original_flags.mean()) != old["unsafe_flagged_rate"]:
        raise ValueError("Frozen unsafe-test decisions changed")
    metrics = calculate_detection_metrics(
        np.r_[np.zeros(len(s)), np.ones(len(original_s))],
        np.r_[flags, original_flags],
        np.r_[ranking, original_ranking],
    )
    records, steering = steer_query_signatures(
        safe,
        unsafe,
        groups,
        safe_rule,
        unsafe_rule,
        safe_cutoff,
        cutoff,
        manifest["settings"]["steering"],
        query_splits=(SPLIT,),
    )
    for row, flag, rank in zip(records, flags, ranking, strict=True):
        row.update(
            flagged=bool(flag),
            ranking_score=float(rank),
            inside_safe_after=row["safe_score"] <= safe_cutoff,
            outside_unsafe_after=row["unsafe_score"] > unsafe_cutoff,
        )
        row["joint_conditions_after"] = (
            row["inside_safe_after"] and row["outside_unsafe_after"]
        )
    steering[SPLIT]["post_joint_stop_rate"] = float(
        np.mean([r["joint_conditions_after"] for r in records])
    )
    categories = sorted({r.get("category") for r in records if r.get("category")})
    category_rates = {
        c: float(np.mean([r["flagged"] for r in records if r["category"] == c]))
        for c in categories
    }
    result = {
        "space": space,
        "method": method,
        "mode": mode,
        **metrics,
        "original_safe_false_positive_rate": old["safe_false_positive_rate"],
        "additional_intended_benign_n": len(s),
        "intended_benign_flagged_rate": float(flags.mean()),
        "flagged_by_aligned_category": category_rates,
        "steering": steering,
        "calibration": calibration,
        "label_meaning": manifest["label_meaning"],
    }
    return result, records


def _persist_control_mode(inputs, scores, mode, result, records):
    target = inputs.output / "variants" / scores.name / mode
    write_csv(target / "decisions_and_steering.csv", records)
    write_json(target / "report.json", result)
    print(
        f"{scores.name}/{mode}: additional intended-benign flagged {result['intended_benign_flagged_rate']:.3f}; "
        f"AUROC {result['auroc_prompt_labels']:.4f}",
        flush=True,
    )


def score_control_signatures(output, space):
    frozen_inputs = read_frozen_control_inputs(output)
    inputs = _load_control_score_inputs(output, space, frozen_inputs)
    results = {}
    for method in VARIANTS[space]:
        scores = _measure_control_method_scores(inputs, method)
        for mode in MODES:
            result, records = _score_control_mode(inputs, method, mode, scores)
            _persist_control_mode(inputs, scores, mode, result, records)
            results[scores.name + "__" + mode] = result
    return results


def run_detection_scoring(
    output: Path,
    *,
    mode: str = "resident",
    source: Path | None = None,
    report_path: Path | None = None,
    options: dict | None = None,
) -> dict | list:
    """Score paired, resident, probe, source-slice, reused or control artifacts."""
    if mode not in {
        "paired",
        "resident",
        "probes",
        "source_slice",
        "reused",
        "controls",
        "comparison",
    }:
        raise ValueError("Unknown detection scoring mode")
    output = Path(output).resolve()
    options = dict(options or {})
    allowed = {"working_directory", "threads"} if mode == "comparison" else set()
    if set(options) - allowed:
        raise ValueError("Unknown scoring options")
    if mode == "resident":
        return _score_resident_detection(output)
    if mode == "paired":
        output, manifest, groups, tables = load_paired_detection_scores(output)
        calibration = freeze_validation_detection_thresholds(output, manifest, tables)
        return evaluate_held_out_paired_detection(output, groups, tables, calibration)
    if mode == "probes":
        inputs = load_probe_scoring_inputs(output)
        rules = fit_probe_training_rules(inputs)
        reports = evaluate_probe_detection_methods(inputs, rules)
        return complete_probe_detection_results(inputs, reports, report_path)
    if mode == "source_slice":
        inputs = load_source_scoring_inputs(output)
        rules = fit_source_training_rules(inputs)
        reports = evaluate_source_detection_rules(inputs, rules)
        return complete_source_comparison(inputs, reports)
    if mode == "comparison":
        if source is None:
            raise ValueError("Scoring comparison requires a frozen source")
        threads = options.get("threads", 4)
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        torch.set_num_threads(threads)
        inputs = load_scoring_comparison_inputs(source, output)
        methods = evaluate_reused_distance_methods(inputs)
        matrices = fit_training_adaptive_geometry(
            inputs, options.get("working_directory", "/var/tmp")
        )
        methods["anchor_adaptive"] = _evaluate_scoring_method(
            inputs, "anchor_adaptive", matrices
        )
        return complete_scoring_comparison(inputs, methods)
    results = {}
    for current_space in VARIANTS:
        results.update(
            score_control_signatures(output, current_space)
            if mode == "controls"
            else score_signature_space(output, current_space)
        )
    if mode == "controls":
        write_control_comparison_report(output, results)
    else:
        write_detection_comparison_report(output, results)
    write_json(
        output / "analysis_status.json",
        {
            "status": "complete",
            "phase": "additional_comparison_complete"
            if mode == "controls"
            else "comparison_complete",
        },
    )
    return results
