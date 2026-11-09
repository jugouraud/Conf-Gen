"""H1: publish distinct field, image, preservation, and compute measurements."""

import csv
import json

from backend.src.common.storage import (
    calculate_file_sha256,
    calculate_source_fingerprints,
    read_json,
    write_json,
)
from backend.src.repair.contracts import RepairExperiment
from backend.src.repair.rendering import (
    build_manual_review_pairs,
    build_repair_figure_artifacts,
    build_repair_image_gallery,
    render_manual_review_html,
    summarize_repair_comparison,
)


def _read_execution_records(experiment, stage):
    records = []
    for path in sorted((experiment.output / stage / "records").glob("*.json")):
        receipt = read_json(path)
        if receipt["experiment_sha256"] != calculate_file_sha256(
            experiment.output / "experiment.json"
        ):
            raise ValueError("Report receipt belongs to a different frozen experiment")
        for relative, expected in receipt["artifacts"].items():
            if calculate_file_sha256(experiment.output / relative) != expected:
                raise ValueError("Report artifact checksum changed")
        records.extend(receipt["rows"])
    return records


def _estimate_full_run(experiment, records):
    options, steering = (
        experiment.settings["experiment"],
        experiment.settings["steering"],
    )
    splits = (
        ("unsafe_test",)
        if options.get("cohort_selection") == "unsafe_test_only"
        else ("safe_test", "unsafe_test", "additional_safe_test")
    )
    prompt_count = sum(len(experiment.groups[split]) for split in splits)
    methods = options["methods"]
    shapes = int("R1" in methods) + len(steering["window_widths"]) * (
        3 * int("R2" in methods) + int("R3" in methods) + 3 * int("R4" in methods)
    )
    per_pair = (
        1
        + shapes
        + int("R5" in methods)
        + (shapes + int("R5" in methods)) * len(steering["matched_budgets"])
    )
    records_upper = (
        prompt_count
        * len(options["modes"])
        * len(options["generation_seeds"])
        * per_pair
    )
    timings = [
        r["generation"]["inference_seconds"]
        for r in records
        if not r["generation"].get("reused_baseline")
    ]
    reference_bytes = (
        experiment.bank["states"].numel()
        * 4
        * sum(
            len(experiment.groups[split])
            for split in ("safe_reference", "unsafe_reference")
        )
    )
    return {
        "held_out_prompts": prompt_count,
        "held_out_splits": list(splits),
        "generation_records_upper_bound": records_upper,
        "sampling_field_calls_upper_bound": records_upper
        * options["inference_steps"]
        * 3,
        "raw_latent_bytes_upper_bound": records_upper
        * experiment.bank["states"][0].numel()
        * 4,
        "reference_signature_scan_bytes_per_flagged_prompt_mode": reference_bytes,
        "field_call_count_includes": "up to two steered conditions plus one perturbation diagnostic per Euler step",
        "sampling_seconds_rough_upper_estimate": records_upper * max(timings)
        if timings
        else None,
        "timing_excludes": "model load, prompt encoding, probe extraction/verification, streamed reference I/O, decoding, image evaluation",
        "reference_io": "One full safe+unsafe reference scan per selected anchor for geometry; direct verification loads possible nearest references using a residual/triangle-inequality bound. RAM holds one reference at a time.",
        "warning": "No GPU timing available"
        if not timings
        else "Pilot maximum per-image sampling time extrapolated; not a full wall-clock forecast",
    }


def _write_comparison_json(experiment, stage, records, summaries, estimate):
    status_path = experiment.output / stage / "status.json"
    status = read_json(status_path) if status_path.exists() else {"status": "prepared"}
    payload = {
        "cohort": stage,
        "status": status,
        "records": len(records),
        "methods": summaries,
        "full_run_estimate": estimate,
        "independent_probe": "not_evaluated",
        "image_evaluation": "not_evaluated"
        if not records
        or all(r["image_evaluation"]["status"] == "not_evaluated" for r in records)
        else "evaluated",
        "conclusion": "No empirical safety–preservation ranking without independently evaluated generated images.",
        "limitations": [
            "Frozen-probe feasibility does not establish image safety.",
            "Numerical search may miss narrow feasible intervals; R5 is a local optimizer.",
            "Pilot and full cohorts overlap; do not pool them for uncertainty estimates.",
            "Bootstrap intervals resample prompts within reported strata; seeds remain paired.",
            "Accepted prompts are unchanged. Pixel similarity is not semantic preservation.",
            "Independent probes have not been measured.",
        ],
    }
    write_json(experiment.output / stage / "comparison.json", payload)


def _write_comparison_csv(experiment, stage, summaries):
    if not summaries:
        return
    rows = []
    for summary in summaries:
        flat = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                flat[key] = value["mean"]
                flat[f"{key}_ci95"] = json.dumps(value["ci95"])
            else:
                flat[key] = value
        rows.append(flat)
    with (experiment.output / stage / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_markdown(experiment, stage, summaries, estimate):
    lines = [
        f"# Vector-field repair comparison: {stage}",
        "",
        (
            "Frozen field-probe admission, independent-probe behavior, and generated-image safety are separate outcomes. "
            "Passing the frozen constraints never proves image safety. Missing image assessments are unmeasured."
        ),
        "",
        (
            "All configured widths/statistics are reported separately. Minimum means smallest verified grid-detected intervention "
            "for that schedule, or the best feasible local R5 objective; no global optimality is claimed."
        ),
        "",
        "| Mode | Variant | Budget | Split | Category | Field repair | Unsafe images | Semantic preservation | Integrated strength | Field perturbation | Seconds |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        values = []
        for metric in (
            "field_repair_rate",
            "unsafe_image_rate",
            "semantic_preservation",
            "integrated_strength",
            "field_perturbation",
            "inference_seconds",
        ):
            value = row[metric]["mean"]
            values.append("unmeasured" if value is None else f"{value:.4g}")
        lines.append(
            f"| {row['mode']} | {row['variant']} | {row['budget']} | {row['split']} | {row['category']} | "
            + " | ".join(values)
            + " |"
        )
    if not summaries:
        lines.extend(
            [
                "",
                "No generations were completed for this cohort. See status.json for the execution state.",
            ]
        )
    lines.extend(
        [
            "",
            (
                "Confidence intervals and unresolved prompt counts are in comparison.json and comparison.csv. "
                "Intervals use a prompt-cluster bootstrap; sample sizes below two prompts have no interval. "
                "Inference time includes perturbation diagnostics; their model evaluations are recorded separately. "
                "Search time is shared by variants for a prompt/mode and must not be summed across rows."
            ),
            "",
            "[Paired images](paired_images.html) · [Temporal profiles](figures/deviations.svg)",
            "",
            "Full held-out run estimate:",
            "",
            "```json",
            json.dumps(estimate, indent=2),
            "```",
            "",
            (
                "No empirical safety–preservation ranking is established without independently evaluated images. "
                "Independent-probe results are unmeasured."
            ),
            "",
        ]
    )
    (experiment.output / stage / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_figure_artifacts(experiment, stage, records, summaries):
    folder = experiment.output / stage / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for name, contents in build_repair_figure_artifacts(records, summaries).items():
        (folder / name).write_text(contents, encoding="utf-8")


def _write_manual_review(experiment, stage, records):
    pairs = build_manual_review_pairs(records)
    folder = experiment.output / stage
    (folder / "manual_review.html").write_text(
        render_manual_review_html(pairs), encoding="utf-8"
    )
    if pairs:
        with (folder / "manual_review.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)


def _write_report_receipt(experiment, stage):
    folder = experiment.output / stage
    artifacts = [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and (
            path.name.startswith("comparison.")
            or path.name == "paired_images.html"
            or path.name.startswith("manual_review.")
            or path.parent.name == "figures"
        )
    ]
    write_json(
        folder / "report.provenance.json",
        {
            "experiment_sha256": calculate_file_sha256(
                experiment.output / "experiment.json"
            ),
            "implementation": calculate_source_fingerprints(
                (
                    "backend/src/repair/report.py",
                    "backend/src/repair/rendering.py",
                    "backend/src/common/storage.py",
                    "backend/src/common/paths.py",
                    "backend/src/repair/steering.py",
                    "backend/src/detection/calibration.py",
                    "backend/src/fields/signatures.py",
                    "backend/src/data/prompts.py",
                )
            ),
            "artifacts": {
                path.relative_to(folder).as_posix(): calculate_file_sha256(path)
                for path in artifacts
            },
        },
    )


def collect_repair_report_records(experiment, stage, records):
    if records is None:
        records = _read_execution_records(experiment, stage)
    summaries = summarize_repair_comparison(records)
    estimate = _estimate_full_run(experiment, records)
    return records, summaries, estimate


def write_repair_comparison_tables(experiment, stage, records, summaries, estimate):
    _write_comparison_json(experiment, stage, records, summaries, estimate)
    _write_comparison_csv(experiment, stage, summaries)
    _write_comparison_markdown(experiment, stage, summaries, estimate)


def write_repair_comparison_figures(experiment, stage, records, summaries):
    _write_figure_artifacts(experiment, stage, records, summaries)
    (experiment.output / stage / "paired_images.html").write_text(
        build_repair_image_gallery(records), encoding="utf-8"
    )
    _write_manual_review(experiment, stage, records)
    _write_report_receipt(experiment, stage)


def run_repair_reporting(
    experiment: "RepairExperiment", stage: str, records: list[dict] | None = None
) -> None:
    records, summaries, estimate = collect_repair_report_records(
        experiment, stage, records
    )
    write_repair_comparison_tables(experiment, stage, records, summaries, estimate)
    write_repair_comparison_figures(experiment, stage, records, summaries)
