"""Detection CSV/JSON/Markdown publication and report provenance."""

from pathlib import Path

from backend.src.common.paths import PROJECT_ROOT
from backend.src.common.storage import (
    calculate_file_sha256,
    read_json,
    write_csv,
    write_json,
)
from backend.src.data.definitions import SPLIT
from backend.src.fields.definitions import VARIANTS


def _write_detection_comparison_tables(output, methods):
    manifest = read_json(output / "experiment.json")
    write_json(output / "comparison.json", {"experiment": manifest, "methods": methods})
    rows = [
        {
            "method": f"{r['space']}_{r['method']}",
            "mode": r["mode"],
            **{
                k: r[k]
                for k in (
                    "safe_test_n",
                    "unsafe_test_n",
                    "safe_false_positive_rate",
                    "unsafe_flagged_rate",
                    "auroc_prompt_labels",
                    "auprc_prompt_labels",
                )
            },
            **{
                f"unsafe_{category}_flagged_rate": value
                for category, value in r["unsafe_flagged_by_category"].items()
            },
            **{
                f"{split}_{k}": value
                for split, summary in r["steering"].items()
                for k, value in summary.items()
                if k != "n"
            },
        }
        for r in methods.values()
    ]
    write_csv(output / "comparison.csv", rows)
    return manifest, rows


def _write_detection_comparison_markdown(output, methods, manifest, rows):
    lines = [
        "# Detection comparison",
        "",
        f"Reuses {manifest['counts']['safe_reference']} frozen safe anchors, "
        f"{manifest['counts']['safe_validation']} safe calibration prompts, "
        f"{manifest['counts']['safe_test']} safe tests, "
        "their signatures, and both paired probe banks. Only new unsafe signatures require inference.",
        "",
        f"Unsafe sets: {manifest['counts']['unsafe_reference']} anchors, "
        f"{manifest['counts']['unsafe_validation']} independent calibration prompts, "
        f"{manifest['counts']['unsafe_test']} held-out tests.",
        "",
        ""
        "Categories come from the frozen dataset; multi-label prompts are assigned once "
        "for stratification, retain all original labels, and never cross partitions.",
        "",
        "| Method | Mode | Safe false alarms | Unsafe flagged | AUROC | AUPRC | Unsafe repairs / requested | Unsafe-test joint stop after steering |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['mode']} | {row['safe_false_positive_rate']:.1%} | "
            f"{row['unsafe_flagged_rate']:.1%} | {row['auroc_prompt_labels']:.4f} | "
            f"{row['auprc_prompt_labels']:.4f} | {row['unsafe_test_verified_repairs']} / {row['unsafe_test_repair_requested']} | "
            f"{row['unsafe_test_post_joint_stop_rate']:.1%} |"
        )
    lines += [
        "",
        "Safe attraction admits safe_score <= safe_cutoff. Joint mode also requires "
        "unsafe_score > unsafe_cutoff. Each method uses its own nearest, uniform, or query-adaptive score "
        "for both classes. Adaptive bandwidths are fitted independently on within-class anchor distances. "
        "Safe cutoffs use safe calibration only. Unsafe cutoffs use unsafe calibration and the existing "
        "safe-calibration 5% lower-order-statistic cap. No test prompt sets any cutoff or bandwidth.",
        "",
        "The cap limits observed safe-calibration exclusion; joint admission does not inherit the "
        "safe-only coverage guarantee. Rankings use max(safe_score/safe_cutoff, 2-unsafe_score/unsafe_cutoff), "
        "with the closed unsafe boundary rejected explicitly.",
        "",
        "Steering chooses the nearest approved safe anchor and searches the signature mixture segment. "
        "Weights are recomputed at every candidate point. A 257-point grid locates the first detected "
        "feasible interval, followed by 32 bisections and an explicit score check. Global minimality "
        "is not claimed for adaptive or nearest scores. Missing approved targets are unresolved.",
        "",
        "The post-steering joint-stop column applies the same safe-membership and unsafe-exclusion "
        "conditions to both modes. Safe-only repair success alone does not imply unsafe-region exclusion. "
        "Unresolved requests retain their original field signature in these measurements.",
        "",
        "Verified repairs satisfy only the frozen field-probe inequalities. These results do not measure "
        "generated-image safety, semantic preservation, or behavior along new sampling trajectories.",
        "",
    ]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _write_detection_comparison_receipt(output):
    artifacts = [
        p
        for p in output.rglob("*")
        if p.is_file()
        and (
            p.suffix in (".csv", ".md")
            or p.name in ("report.json", "calibration.json", "comparison.json")
        )
    ]
    write_json(
        output / "comparison.provenance.json",
        {
            "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
            "distance_receipts": {
                s: calculate_file_sha256(
                    output / s / "two_mode_distances.provenance.json"
                )
                for s in VARIANTS
            },
            "implementation": {
                p: calculate_file_sha256(PROJECT_ROOT / p)
                for p in (
                    "backend/src/detection/score.py",
                    "backend/src/common/paths.py",
                    "backend/src/common/storage.py",
                    "backend/src/data/prompts.py",
                    "backend/src/fields/signatures.py",
                    "backend/src/detection/calibration.py",
                    "backend/src/repair/steering.py",
                )
            },
            "artifacts": {
                p.relative_to(output).as_posix(): calculate_file_sha256(p)
                for p in artifacts
            },
        },
    )


def write_detection_comparison_report(output, methods):
    manifest, rows = _write_detection_comparison_tables(output, methods)
    _write_detection_comparison_markdown(output, methods, manifest, rows)
    _write_detection_comparison_receipt(output)


def complete_probe_detection_results(inputs, reports, report_path):
    keys = (
        "safe_test_n",
        "unsafe_test_n",
        "safe_false_positive_rate",
        "unsafe_flagged_rate",
        "auroc_prompt_labels",
        "auprc_prompt_labels",
        "accuracy",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    )
    rows = [
        {
            "detector": detector,
            "mode": mode,
            "probe_aggregation": inputs.aggregation,
            **{key: metrics[key] for key in keys},
        }
        for detector, report in reports.items()
        for mode, metrics in report["modes"].items()
    ]
    write_json(
        inputs.output / "comparison.json",
        {"probe_aggregation": inputs.aggregation, "methods": reports},
    )
    write_csv(inputs.output / "comparison.csv", rows)
    lines = [
        f"# Detection with {inputs.aggregation} probe deviation",
        "",
        "All five methods reuse frozen field signatures. The probe reduction is applied to each prompt–training-anchor distance before aggregation across anchors.",
        "",
        "| Detector | Detection rule | Safe false positives | Illegal detected | AUROC | AUPRC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {row['detector']} | {row['mode']} | {row['safe_false_positive_rate']:.2%} | {row['unsafe_flagged_rate']:.2%} | {row['auroc_prompt_labels']:.4f} | {row['auprc_prompt_labels']:.4f} |"
        for row in rows
    ]
    lines += [
        "",
        "Adaptive bandwidths are refitted from all within-class training pairs. Thresholds use validation only, with the existing legal-validation cap on the unsafe region. Test prompts are used for performance evaluation only. These results concern prompt labels; no images are generated.",
        "",
    ]
    text = "\n".join(lines)
    (inputs.output / "comparison.md").write_text(text, encoding="utf-8")
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(text, encoding="utf-8")
    artifacts = [
        p
        for p in inputs.output.rglob("*")
        if p.is_file()
        and p.name
        in (
            "calibration.json",
            "report.json",
            "test_decisions.csv",
            "comparison.json",
            "comparison.csv",
            "comparison.md",
            "experiment.json",
            "distances.provenance.json",
        )
    ]
    write_json(
        inputs.output / "comparison.provenance.json",
        {
            "artifacts": {
                p.relative_to(inputs.output).as_posix(): calculate_file_sha256(p)
                for p in artifacts
            }
        },
    )
    write_json(
        inputs.output / "status.json",
        {
            "status": "complete",
            "phase": "five_methods_evaluated",
            "probe_aggregation": inputs.aggregation,
            "methods": list(reports),
        },
    )
    return rows


def complete_source_comparison(inputs, reports):
    rows = [
        {"space": "source_slice", "method": method, "mode": mode, **metrics}
        for method, report in reports.items()
        for mode, metrics in report["modes"].items()
    ]
    write_json(inputs.output / "comparison.json", {"methods": reports})
    # Category tables are retained in JSON; CSV contains scalar aggregate metrics.
    write_csv(
        inputs.output / "comparison.csv",
        [
            {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
            for row in rows
        ],
    )
    write_json(
        inputs.output / "scoring.provenance.json",
        {
            "experiment_sha256": calculate_file_sha256(
                inputs.output / "experiment.json"
            ),
            "distance_receipt_sha256": calculate_file_sha256(
                inputs.output / "source_slice/distances.provenance.json"
            ),
            "bandwidth_policy": "Full within-class training upper triangle median, scale 1.0",
            "threshold_fit": "validation_only",
            "evaluation": "test_only",
            "artifacts": {
                p.relative_to(inputs.output).as_posix(): calculate_file_sha256(p)
                for p in (inputs.output / "scores").rglob("*")
                if p.is_file()
            },
        },
    )
    return reports


def complete_scoring_comparison(inputs, methods):
    original = read_json(inputs.source / "report.json")
    if methods["euclidean"]["modes"] != original["modes"]:
        raise ValueError("Reused Euclidean results differ from frozen baseline")
    rows = [
        {
            "method": method,
            "mode": mode,
            **{
                key: metrics[key]
                for key in (
                    "safe_test_n",
                    "unsafe_test_n",
                    "safe_false_positive_rate",
                    "unsafe_flagged_rate",
                    "auroc_prompt_labels",
                    "auprc_prompt_labels",
                    "accuracy",
                    "true_positive",
                    "false_positive",
                    "true_negative",
                    "false_negative",
                )
            },
        }
        for method, report in methods.items()
        for mode, metrics in report["modes"].items()
    ]
    write_json(
        inputs.output / "comparison.json",
        {"methods": methods, "euclidean_baseline_reproduced": True},
    )
    write_csv(inputs.output / "comparison.csv", rows)
    lines = [
        "# Trajectory scoring comparison",
        "",
        "Held-out test: 400 legal and 400 illegal prompts. All methods reuse the same frozen signatures and query distances.",
        "",
        "| Scorer | Detection rule | Safe false positives | Illegal detected | AUROC | AUPRC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['mode']} | {row['safe_false_positive_rate']:.2%} | {row['unsafe_flagged_rate']:.2%} | {row['auroc_prompt_labels']:.4f} | {row['auprc_prompt_labels']:.4f} |"
        )
    lines += [
        "",
        "Adaptive bandwidths use the complete within-class training-pair median with scale 1.0. Thresholds use validation only; test is evaluation only. Euclidean reproduces the frozen original results exactly.",
        "",
    ]
    (inputs.output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    artifacts = [
        p
        for p in inputs.output.rglob("*")
        if p.is_file() and p.name not in ("status.json", "comparison.provenance.json")
    ]
    write_json(
        inputs.output / "comparison.provenance.json",
        {
            "artifacts": {
                p.relative_to(inputs.output).as_posix(): calculate_file_sha256(p)
                for p in artifacts
            }
        },
    )
    write_json(
        inputs.output / "status.json",
        {"status": "complete", "phase": "scoring_comparison_complete"},
    )
    return rows


def _write_control_comparison_tables(output, results):
    manifest = read_json(output / "experiment.json")
    write_json(output / "comparison.json", {"experiment": manifest, "methods": results})
    rows = [
        {
            "method": f"{r['space']}_{r['method']}",
            "mode": r["mode"],
            **{
                k: r[k]
                for k in (
                    "additional_intended_benign_n",
                    "intended_benign_flagged_rate",
                    "original_safe_false_positive_rate",
                    "unsafe_test_n",
                    "unsafe_flagged_rate",
                    "auroc_prompt_labels",
                    "auprc_prompt_labels",
                )
            },
            **r["steering"][SPLIT],
        }
        for r in results.values()
    ]
    write_csv(output / "comparison.csv", rows)
    return manifest, rows


def _write_control_comparison_markdown(output, results, manifest, rows):
    input_name = manifest["input_file"].replace("\\", "/").rsplit("/", 1)[-1]
    lines = [
        "# Additional test: style-matched intended-benign controls",
        "",
        f"Input: `{input_name}`; {manifest['counts'][SPLIT]} new test prompts. "
        "Every prompt is disjoint from all original splits after canonical normalization.",
        "",
        "All five methods and both modes use the original probe banks, safe and unsafe anchors, bandwidths, "
        "calibration prompts, and cutoffs. No additional prompt enters fitting, calibration, or tuning. "
        "AUROC/AUPRC compare these new controls against the same 1,000 original unsafe test prompts; "
        "the original DiffusionDB safe-test results remain a separate evaluation.",
        "",
        "The supplied labels are **synthetic intended-benign labels**, without human safety or generated-image "
        "validation. Flag rates below are observed rejection rates for that intended population. "
        "The original conformal exchangeability assumption does not establish a 5% guarantee for this new distribution. "
        "These controls are matched by category and descriptive style, without one-to-one unsafe prompt pairing.",
        "",
        "| Method | Mode | Original safe flagged | New intended-benign flagged | Unsafe flagged | AUROC with new controls | AUPRC | New-control joint stop after steering |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['mode']} | {r['original_safe_false_positive_rate']:.1%} | "
            f"{r['intended_benign_flagged_rate']:.1%} | {r['unsafe_flagged_rate']:.1%} | "
            f"{r['auroc_prompt_labels']:.4f} | {r['auprc_prompt_labels']:.4f} | {r['post_joint_stop_rate']:.1%} |"
        )
    categories = sorted(
        {
            category
            for result in results.values()
            for category in result["flagged_by_aligned_category"]
        }
    )
    lines += [
        "",
        "Steering success refers to the frozen probe inequalities. It does not establish generated-image "
        "safety or semantic preservation. The capped unsafe cutoff is unchanged from the original run.",
        "",
        "Intended-benign flag rates by the supplied aligned category:",
        "",
        "| Method | Mode | " + " | ".join(categories) + " |",
        "|---|---|" + "---:|" * len(categories),
    ]
    for r in results.values():
        rates = r["flagged_by_aligned_category"]
        lines.append(
            f"| {r['space']}_{r['method']} | {r['mode']} | "
            + " | ".join(f"{rates[c]:.1%}" if c in rates else "N/A" for c in categories)
            + " |"
        )
    lines.append("")
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _write_control_comparison_receipt(output):
    artifacts = [
        p
        for p in output.rglob("*")
        if p.is_file()
        and (
            p.suffix in (".csv", ".md") or p.name in ("report.json", "comparison.json")
        )
    ]
    write_json(
        output / "comparison.provenance.json",
        {
            "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
            "implementation": {
                p: calculate_file_sha256(PROJECT_ROOT / p)
                for p in (
                    "backend/src/detection/score.py",
                    "backend/src/detection/score.py",
                    "backend/src/common/paths.py",
                    "backend/src/common/storage.py",
                    "backend/src/data/prompts.py",
                    "backend/src/fields/signatures.py",
                    "backend/src/detection/calibration.py",
                    "backend/src/repair/steering.py",
                )
            },
            "distance_receipts": {
                s: calculate_file_sha256(
                    output / s / "additional_distances.provenance.json"
                )
                for s in VARIANTS
            },
            "artifacts": {
                p.relative_to(output).as_posix(): calculate_file_sha256(p)
                for p in artifacts
            },
        },
    )


def write_control_comparison_report(output, results):
    manifest, rows = _write_control_comparison_tables(output, results)
    _write_control_comparison_markdown(output, results, manifest, rows)
    _write_control_comparison_receipt(output)
