"""Compare exactly five methods under safe attraction and joint unsafe avoidance."""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from backend.flow.common import ROOT, sha256, write_json
from backend.flow.t2i_prompts import SAFE_SPLITS, UNSAFE_SPLITS, read_json, read_jsonl
from backend.flow.two_mode_scoring import fit_distance_rule, reduce_distances, steer_segment
from backend.flow.two_sets import capped_cutoff, conformal_cutoff
from backend.flow.comparison import VARIANTS

MODES = ("safe_attraction", "safe_attraction_unsafe_avoidance")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def detection_metrics(labels, flags, scores):
    labels, flags = np.asarray(labels, bool), np.asarray(flags, bool)
    return {"safe_test_n": int((~labels).sum()), "unsafe_test_n": int(labels.sum()),
            "safe_false_positive_rate": float(flags[~labels].mean()),
            "unsafe_flagged_rate": float(flags[labels].mean()),
            "auroc_prompt_labels": float(roc_auc_score(labels, scores)),
            "auprc_prompt_labels": float(average_precision_score(labels, scores))}


def steer_queries(safe, unsafe, groups, safe_spec, unsafe_spec, safe_cutoff, unsafe_cutoff, settings,
                  query_splits=("safe_test", "unsafe_test")):
    eligible = reduce_distances(safe["safe_reference"], safe_spec) <= safe_cutoff
    if unsafe_cutoff is not None:
        eligible &= reduce_distances(unsafe["safe_reference"], unsafe_spec) > unsafe_cutoff
    candidates = np.flatnonzero(eligible)
    records = []
    for split in query_splits:
        for i, row in enumerate(groups[split]):
            qs, qu = safe[split][i], unsafe[split][i]
            ss, us = float(reduce_distances(qs, safe_spec)), float(reduce_distances(qu, unsafe_spec))
            admitted = ss <= safe_cutoff and (unsafe_cutoff is None or us > unsafe_cutoff)
            record = {"id": row["id"], "split": split, "category": row.get("category", "safe"),
                      "admitted_unchanged": admitted, "target_id": None, "kappa": 0. if admitted else None,
                      "initial_safe_score": ss, "initial_unsafe_score": us,
                      "safe_score": ss, "unsafe_score": us, "meets_stop": admitted}
            if not admitted and len(candidates):
                target = int(candidates[np.argmin(qs[candidates])])
                record["target_id"] = groups["safe_reference"][target]["id"]
                repaired = steer_segment(qs, safe["safe_reference"][target], qu, unsafe["safe_reference"][target],
                                         float(qs[target]), safe_spec, unsafe_spec, safe_cutoff, unsafe_cutoff, settings)
                if repaired is not None:
                    record.update(repaired)
            records.append(record)
    summaries = {}
    for split in query_splits:
        rows = [r for r in records if r["split"] == split]
        requested = [r for r in rows if not r["admitted_unchanged"]]
        solved = [r for r in requested if r["meets_stop"]]
        summaries[split] = {"n": len(rows), "admitted_unchanged": len(rows)-len(requested),
                            "repair_requested": len(requested), "verified_repairs": len(solved),
                            "unresolved": len(requested)-len(solved),
                            "mean_kappa_repaired": float(np.mean([r["kappa"] for r in solved])) if solved else None}
    return records, summaries


def score_space(output, space):
    manifest = read_json(output / "experiment.json")
    folder = output / space
    archive = folder / "two_mode_distances.npz"
    receipt_path = folder / "two_mode_distances.provenance.json"
    receipt = read_json(receipt_path)
    if (receipt["experiment_sha256"] != sha256(output / "experiment.json")
            or receipt["distances_sha256"] != sha256(archive)
            or receipt["split_ids"] != manifest["split_ids"]):
        raise ValueError("Distance provenance differs from the frozen experiment")
    with np.load(archive) as data:
        safe = {s: data[f"safe__{s}"].copy() for s in SAFE_SPLITS + UNSAFE_SPLITS}
        unsafe = {s: data[f"unsafe__{s}"].copy() for s in SAFE_SPLITS + UNSAFE_SPLITS}
    groups = {s: read_jsonl(folder / "splits" / f"{s}.jsonl") for s in SAFE_SPLITS + UNSAFE_SPLITS}
    if any([r["id"] for r in rows] != manifest["split_ids"][s] for s, rows in groups.items()):
        raise ValueError("Frozen split order changed")
    settings = manifest["settings"]["experiment"]
    results = {}
    for method in VARIANTS[space]:
        safe_spec = fit_distance_rule(method, safe["safe_reference"], manifest["bandwidth_scale"])
        unsafe_spec = fit_distance_rule(method, unsafe["unsafe_reference"], manifest["bandwidth_scale"])
        safe_cutoff = conformal_cutoff(reduce_distances(safe["safe_validation"], safe_spec),
                                      manifest["config"]["experiment"]["alpha"])
        unsafe_cutoff = conformal_cutoff(reduce_distances(unsafe["unsafe_validation"], unsafe_spec), settings["unsafe_alpha"])
        unsafe_cap, cap_rank = capped_cutoff(unsafe_cutoff, reduce_distances(unsafe["safe_validation"], unsafe_spec), settings["cap_alpha"])
        if not (np.isfinite([safe_cutoff, unsafe_cutoff, unsafe_cap]).all() and min(safe_cutoff, unsafe_cap) > 0):
            raise ValueError("Normalized joint scores require positive finite cutoffs")
        calibration = {"safe_cutoff": safe_cutoff, "unsafe_conformal_cutoff": unsafe_cutoff,
                       "unsafe_cutoff": unsafe_cap, "cap_rank": cap_rank,
                       "safe_rule": safe_spec, "unsafe_rule": unsafe_spec,
                       "safe_calibration_ids": manifest["split_ids"]["safe_validation"],
                       "unsafe_calibration_ids": manifest["split_ids"]["unsafe_validation"],
                       "safe_reference_ids": manifest["split_ids"]["safe_reference"],
                       "unsafe_reference_ids": manifest["split_ids"]["unsafe_reference"]}
        variant = output / "variants" / f"{space}_{method}"
        write_json(variant / "calibration.json", calibration)
        s = np.concatenate([reduce_distances(safe[split], safe_spec) for split in ("safe_test", "unsafe_test")])
        u = np.concatenate([reduce_distances(unsafe[split], unsafe_spec) for split in ("safe_test", "unsafe_test")])
        labels = np.r_[np.zeros(len(groups["safe_test"])), np.ones(len(groups["unsafe_test"]))]
        for mode in MODES:
            cutoff = None if mode == "safe_attraction" else unsafe_cap
            flags = s > safe_cutoff
            rank_scores = s / safe_cutoff
            if cutoff is not None:
                flags |= u <= cutoff
                rank_scores = np.maximum(rank_scores, 2-u/cutoff)
            detection = detection_metrics(labels, flags, rank_scores)
            records, steering = steer_queries(safe, unsafe, groups, safe_spec, unsafe_spec,
                                               safe_cutoff, cutoff, manifest["settings"]["steering"])
            for r, flag, rank_score in zip(records, flags, rank_scores, strict=True):
                r.update(flagged=bool(flag), ranking_score=float(rank_score))
                r["inside_safe_after"] = r["safe_score"] <= safe_cutoff
                r["outside_unsafe_after"] = r["unsafe_score"] > unsafe_cap
                r["joint_conditions_after"] = r["inside_safe_after"] and r["outside_unsafe_after"]
            for split, summary in steering.items():
                rr = [r for r in records if r["split"] == split]
                summary.update(post_safe_membership_rate=float(np.mean([r["inside_safe_after"] for r in rr])),
                               post_unsafe_exclusion_rate=float(np.mean([r["outside_unsafe_after"] for r in rr])),
                               post_joint_stop_rate=float(np.mean([r["joint_conditions_after"] for r in rr])))
            # Explicit per-category metrics preserve overlap labels in the split
            # files, while evaluation strata use each row's unique assignment.
            category_rates = {category: float(np.mean([r["flagged"] for r in records
                                                      if r["split"] == "unsafe_test" and r["category"] == category]))
                              for category in ("Violence", "Pornography", "Illegal_Activities")}
            result = {"space": space, "method": method, "mode": mode,
                      **detection, "unsafe_flagged_by_category": category_rates,
                      "steering": steering, "calibration": calibration}
            write_csv(variant / mode / "decisions_and_steering.csv", records)
            write_json(variant / mode / "report.json", result)
            results[f"{space}_{method}__{mode}"] = result
            print(f"{space}/{method}/{mode}: safe FAR {detection['safe_false_positive_rate']:.3f}, "
                  f"unsafe flagged {detection['unsafe_flagged_rate']:.3f}", flush=True)
    return results


def report(output, methods):
    manifest = read_json(output / "experiment.json")
    write_json(output / "comparison.json", {"experiment": manifest, "methods": methods})
    rows = [{"method": f"{r['space']}_{r['method']}", "mode": r["mode"],
             **{k: r[k] for k in ("safe_test_n", "unsafe_test_n", "safe_false_positive_rate", "unsafe_flagged_rate",
                                  "auroc_prompt_labels", "auprc_prompt_labels")},
             **{f"unsafe_{category}_flagged_rate": value for category, value in r["unsafe_flagged_by_category"].items()},
             **{f"{split}_{k}": value for split, summary in r["steering"].items()
                for k, value in summary.items() if k != "n"}}
            for r in methods.values()]
    write_csv(output / "comparison.csv", rows)
    lines = ["# T2I unsafe sets: five methods, two steering modes", "",
             "Reuses the frozen 1,000 safe anchors, 1,000 safe calibration prompts, 1,000 safe tests, "
             "their signatures, and both paired probe banks. Only new unsafe signatures require inference.", "",
             f"Unsafe sets: {manifest['counts']['unsafe_reference']} anchors, "
             f"{manifest['counts']['unsafe_validation']} independent calibration prompts, "
             f"{manifest['counts']['unsafe_test']} held-out tests. All three categories are represented in every split.", "",
             "[Dataset: T2I-RiskyPrompt](https://github.com/datar001/T2I-RiskyPrompt). "
             "Violence, Pornography, and Illegal_Activities labels only; multi-label prompts are assigned once "
             "for stratification, retain all original labels, and never cross partitions.", "",
             "| Method | Mode | Safe false alarms | Unsafe flagged | AUROC | AUPRC | Unsafe repairs / requested | Unsafe-test joint stop after steering |", 
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['mode']} | {row['safe_false_positive_rate']:.1%} | "
                     f"{row['unsafe_flagged_rate']:.1%} | {row['auroc_prompt_labels']:.4f} | "
                     f"{row['auprc_prompt_labels']:.4f} | {row['unsafe_test_verified_repairs']} / {row['unsafe_test_repair_requested']} | "
                     f"{row['unsafe_test_post_joint_stop_rate']:.1%} |")
    lines += ["", "Safe attraction admits safe_score <= safe_cutoff. Joint mode also requires "
              "unsafe_score > unsafe_cutoff. Each method uses its own nearest, uniform, or query-adaptive score "
              "for both classes. Adaptive bandwidths are fitted independently on within-class anchor distances. "
              "Safe cutoffs use safe calibration only. Unsafe cutoffs use unsafe calibration and the existing "
              "safe-calibration 5% lower-order-statistic cap. No test prompt sets any cutoff or bandwidth.", "",
              "The cap limits observed safe-calibration exclusion; joint admission does not inherit the "
              "safe-only coverage guarantee. Rankings use max(safe_score/safe_cutoff, 2-unsafe_score/unsafe_cutoff), "
              "with the closed unsafe boundary rejected explicitly.", "",
              "Steering chooses the nearest approved safe anchor and searches the signature mixture segment. "
              "Weights are recomputed at every candidate point. A 257-point grid locates the first detected "
              "feasible interval, followed by 32 bisections and an explicit score check. Global minimality "
              "is not claimed for adaptive or nearest scores. Missing approved targets are unresolved.", "",
              "The post-steering joint-stop column applies the same safe-membership and unsafe-exclusion "
              "conditions to both modes. Safe-only repair success alone does not imply unsafe-region exclusion. "
              "Unresolved requests retain their original field signature in these measurements.", "",
              "Verified repairs satisfy only the frozen field-probe inequalities. These results do not measure "
              "generated-image safety, semantic preservation, or behavior along new sampling trajectories.", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    artifacts = [p for p in output.rglob("*") if p.is_file() and
                 (p.suffix in (".csv", ".md") or p.name in ("report.json", "calibration.json", "comparison.json"))]
    write_json(output / "comparison.provenance.json", {
        "experiment_sha256": sha256(output / "experiment.json"),
        "distance_receipts": {s: sha256(output / s / "two_mode_distances.provenance.json") for s in VARIANTS},
        "implementation": {p: sha256(ROOT / p) for p in
                           ("scripts/score_t2i_comparison.py", "backend/flow/two_mode_scoring.py")},
        "artifacts": {p.relative_to(output).as_posix(): sha256(p) for p in artifacts}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/flow_t2i_two_mode_1000")
    args = parser.parse_args()
    output = args.output.resolve()
    results = {}
    for space in VARIANTS:
        results.update(score_space(output, space))
    report(output, results)
    write_json(output / "analysis_status.json", {"status": "complete", "phase": "comparison_complete"})


if __name__ == "__main__":
    main()
