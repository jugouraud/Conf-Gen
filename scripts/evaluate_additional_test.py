"""Evaluate additional intended-benign controls with exclusively frozen decision rules."""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from backend.flow.common import ROOT, sha256, write_json
from backend.flow.distance_matrix import euclidean_matrix
from backend.flow.feature_store import feature_file
from backend.flow.metrics import flat_feature
from backend.flow.probe_bank import bank_file, read_bank
from backend.flow.t2i_prompts import read_json, read_jsonl
from backend.flow.two_mode_scoring import reduce_distances
from backend.flow.comparison import VARIANTS, space_config
from scripts.prepare_additional_test import OUTPUT, SPLIT
from scripts.score_t2i_comparison import MODES, detection_metrics, steer_queries, write_csv


def frozen_inputs(output):
    manifest = read_json(output / "experiment.json")
    if sha256(output / "parent_comparison.json") != manifest["parent_comparison_sha256"]:
        raise ValueError("Frozen parent comparison changed")
    parent = read_json(output / "parent_comparison.json")
    frozen = read_json(output / "frozen_calibration.json")
    for key, calibration in frozen.items():
        if calibration != parent["methods"][key + "__safe_attraction"]["calibration"]:
            raise ValueError("Additional test attempted to change a frozen calibration rule")
    return manifest, parent, frozen


def measure(output, space, device="cuda"):
    cfg = space_config(output, space)
    folder = output / space
    manifest, _, _ = frozen_inputs(output)
    bank, bank_hash = read_bank(cfg), sha256(bank_file(cfg))
    reference_receipt = read_json(folder / "reference_features.provenance.json")
    query_receipt = read_json(folder / "additional_features.provenance.json")
    rows = read_jsonl(folder / "splits" / f"{SPLIT}.jsonl")
    if [r["id"] for r in rows] != manifest["split_ids"][SPLIT]:
        raise ValueError("Additional test ordering changed")
    if (reference_receipt["bank_sha256"] != bank_hash or query_receipt["bank_sha256"] != bank_hash
            or query_receipt["experiment_sha256"] != sha256(output / "experiment.json")):
        raise ValueError("Additional feature provenance changed")
    groups = {s: read_jsonl(folder / "splits" / f"{s}.jsonl") for s in ("safe_reference", "unsafe_reference")}
    for split, group in dict(groups, **{SPLIT: rows}).items():
        hashes = query_receipt["features"] if split == SPLIT else reference_receipt["features"]
        if split != SPLIT and [r["id"] for r in group] != manifest["reference_ids"][split]:
            raise ValueError("Frozen anchor ordering changed")
        for row in group:
            if sha256(feature_file(cfg, split, row["id"])) != hashes[f"{split}/{row['id']}.pt"]:
                raise ValueError("Additional-test input feature changed")
    target, receipt_path = folder / "additional_distances.npz", folder / "additional_distances.provenance.json"
    expected = {"experiment_sha256": sha256(output / "experiment.json"), "bank_sha256": bank_hash,
                "reference_receipt_sha256": sha256(folder / "reference_features.provenance.json"),
                "query_receipt_sha256": sha256(folder / "additional_features.provenance.json"),
                "query_ids": manifest["split_ids"][SPLIT], "implementation": "FP64 Euclidean Gram accumulation"}
    if target.exists() and receipt_path.exists():
        if read_json(receipt_path) != dict(expected, distances_sha256=sha256(target)):
            raise ValueError("Additional distance cache changed")
        return

    def load(split, row):
        return flat_feature(cfg, split, row, bank, bank_hash)

    tables = {}
    for split, anchors in groups.items():
        first = load(split, anchors[0])
        refs = torch.empty((len(anchors), first.numel()), dtype=torch.float64, device=device)
        refs[0] = first.to(device)
        for index, row in enumerate(anchors[1:], 1):
            refs[index] = load(split, row).to(device)
        norms = torch.linalg.vector_norm(refs, dim=1).square()
        values = []
        for start in range(0, len(rows), 16):
            queries = torch.stack([load(SPLIT, row) for row in rows[start:start+16]]).to(device=device, dtype=torch.float64)
            values.append(euclidean_matrix(queries, refs, reference_squared_norms=norms).cpu().numpy())
            write_json(output / "analysis_status.json", {"status": "running", "phase": "additional_distances",
                       "space": space, "references": split, "completed": min(start+16, len(rows)), "total": len(rows)})
        tables["safe" if split == "safe_reference" else "unsafe"] = np.concatenate(values)
        del refs, norms, queries
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    np.savez_compressed(target, **tables)
    write_json(receipt_path, dict(expected, distances_sha256=sha256(target)))


def score(output, space):
    manifest, parent, frozen = frozen_inputs(output)
    folder = output / space
    if sha256(folder / "frozen_geometry.npz") != manifest["frozen_geometry_sha256"][space]:
        raise ValueError("Parent geometry changed")
    receipt = read_json(folder / "additional_distances.provenance.json")
    if (receipt["experiment_sha256"] != sha256(output / "experiment.json")
            or receipt["distances_sha256"] != sha256(folder / "additional_distances.npz")):
        raise ValueError("Additional distance provenance changed")
    with np.load(folder / "additional_distances.npz") as data:
        query = {key: data[key].copy() for key in ("safe", "unsafe")}
    with np.load(folder / "frozen_geometry.npz") as data:
        geometry = {key: data[key].copy() for key in data.files}
    groups = {s: read_jsonl(folder / "splits" / f"{s}.jsonl") for s in ("safe_reference", "unsafe_test", SPLIT)}
    if [r["id"] for r in groups[SPLIT]] != manifest["split_ids"][SPLIT]:
        raise ValueError("Additional test ordering changed")
    safe = {"safe_reference": geometry["safe__safe_reference"], SPLIT: query["safe"]}
    unsafe = {"safe_reference": geometry["unsafe__safe_reference"], SPLIT: query["unsafe"]}
    results = {}
    for method in VARIANTS[space]:
        name = f"{space}_{method}"
        calibration = frozen[name]
        safe_rule, unsafe_rule = calibration["safe_rule"], calibration["unsafe_rule"]
        safe_cutoff, unsafe_cutoff = calibration["safe_cutoff"], calibration["unsafe_cutoff"]
        s, u = reduce_distances(query["safe"], safe_rule), reduce_distances(query["unsafe"], unsafe_rule)
        original_s = reduce_distances(geometry["safe__unsafe_test"], safe_rule)
        original_u = reduce_distances(geometry["unsafe__unsafe_test"], unsafe_rule)
        for mode in MODES:
            cutoff = None if mode == "safe_attraction" else unsafe_cutoff
            flags, original_flags = s > safe_cutoff, original_s > safe_cutoff
            ranking, original_ranking = s/safe_cutoff, original_s/safe_cutoff
            if cutoff is not None:
                flags |= u <= cutoff
                original_flags |= original_u <= cutoff
                ranking, original_ranking = np.maximum(ranking, 2-u/cutoff), np.maximum(original_ranking, 2-original_u/cutoff)
            old = parent["methods"][name + "__" + mode]
            if float(original_flags.mean()) != old["unsafe_flagged_rate"]:
                raise ValueError("Frozen unsafe-test decisions changed")
            metrics = detection_metrics(np.r_[np.zeros(len(s)), np.ones(len(original_s))],
                                        np.r_[flags, original_flags], np.r_[ranking, original_ranking])
            records, steering = steer_queries(safe, unsafe, groups, safe_rule, unsafe_rule, safe_cutoff, cutoff,
                                               manifest["settings"]["steering"], query_splits=(SPLIT,))
            for row, flag, rank in zip(records, flags, ranking, strict=True):
                row.update(flagged=bool(flag), ranking_score=float(rank),
                           inside_safe_after=row["safe_score"] <= safe_cutoff,
                           outside_unsafe_after=row["unsafe_score"] > unsafe_cutoff)
                row["joint_conditions_after"] = row["inside_safe_after"] and row["outside_unsafe_after"]
            steering[SPLIT]["post_joint_stop_rate"] = float(np.mean([r["joint_conditions_after"] for r in records]))
            categories = sorted({r.get("category") for r in records if r.get("category")})
            category_rates = {c: float(np.mean([r["flagged"] for r in records if r["category"] == c])) for c in categories}
            result = {"space": space, "method": method, "mode": mode, **metrics,
                      "original_safe_false_positive_rate": old["safe_false_positive_rate"],
                      "additional_intended_benign_n": len(s), "intended_benign_flagged_rate": float(flags.mean()),
                      "flagged_by_aligned_category": category_rates, "steering": steering,
                      "calibration": calibration, "label_meaning": manifest["label_meaning"]}
            target = output / "variants" / name / mode
            write_csv(target / "decisions_and_steering.csv", records)
            write_json(target / "report.json", result)
            results[name + "__" + mode] = result
            print(f"{name}/{mode}: additional intended-benign flagged {float(flags.mean()):.3f}; AUROC {metrics['auroc_prompt_labels']:.4f}", flush=True)
    return results


def report(output, results):
    manifest = read_json(output / "experiment.json")
    input_name = manifest["input_file"].replace("\\", "/").rsplit("/", 1)[-1]
    write_json(output / "comparison.json", {"experiment": manifest, "methods": results})
    rows = [{"method": f"{r['space']}_{r['method']}", "mode": r["mode"],
             **{k: r[k] for k in ("additional_intended_benign_n", "intended_benign_flagged_rate", "original_safe_false_positive_rate",
                                  "unsafe_test_n", "unsafe_flagged_rate", "auroc_prompt_labels", "auprc_prompt_labels")},
             **r["steering"][SPLIT]} for r in results.values()]
    write_csv(output / "comparison.csv", rows)
    lines = ["# Additional test: style-matched intended-benign controls", "",
             f"Input: `{input_name}`; {manifest['counts'][SPLIT]} new test prompts. "
             "Every prompt is disjoint from all original splits after canonical normalization.", "",
             "All five methods and both modes use the original probe banks, safe and unsafe anchors, bandwidths, "
             "calibration prompts, and cutoffs. No additional prompt enters fitting, calibration, or tuning. "
             "AUROC/AUPRC compare these new controls against the same 1,000 original unsafe test prompts; "
             "the original DiffusionDB safe-test results remain a separate evaluation.", "",
             "The supplied labels are **synthetic intended-benign labels**, without human safety or generated-image "
             "validation. Flag rates below are observed rejection rates for that intended population. "
             "The original conformal exchangeability assumption does not establish a 5% guarantee for this new distribution. "
             "These controls are matched by category and descriptive style, without one-to-one unsafe prompt pairing.", "",
             "| Method | Mode | Original safe flagged | New intended-benign flagged | Unsafe flagged | AUROC with new controls | AUPRC | New-control joint stop after steering |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['mode']} | {r['original_safe_false_positive_rate']:.1%} | "
                     f"{r['intended_benign_flagged_rate']:.1%} | {r['unsafe_flagged_rate']:.1%} | "
                     f"{r['auroc_prompt_labels']:.4f} | {r['auprc_prompt_labels']:.4f} | {r['post_joint_stop_rate']:.1%} |")
    lines += ["", "Steering success refers to the frozen probe inequalities. It does not establish generated-image "
              "safety or semantic preservation. The capped unsafe cutoff is unchanged from the original run.", "",
              "Intended-benign flag rates by the supplied aligned category:", "",
              "| Method | Mode | Violence | Pornography | Illegal activities |",
              "|---|---|---:|---:|---:|"]
    for r in results.values():
        rates = r["flagged_by_aligned_category"]
        lines.append(f"| {r['space']}_{r['method']} | {r['mode']} | "
                     + " | ".join(f"{rates[c]:.1%}" for c in ("Violence", "Pornography", "Illegal_Activities")) + " |")
    lines.append("")
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    artifacts = [p for p in output.rglob("*") if p.is_file() and
                 (p.suffix in (".csv", ".md") or p.name in ("report.json", "comparison.json"))]
    write_json(output / "comparison.provenance.json", {"experiment_sha256": sha256(output / "experiment.json"),
                "implementation": {p: sha256(ROOT / p) for p in ("scripts/evaluate_additional_test.py", "scripts/score_t2i_comparison.py", "backend/flow/two_mode_scoring.py")},
                "distance_receipts": {s: sha256(output / s / "additional_distances.provenance.json") for s in VARIANTS},
                "artifacts": {p.relative_to(output).as_posix(): sha256(p) for p in artifacts}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    torch.set_num_threads(4)
    results = {}
    for space in VARIANTS:
        measure(output, space, args.device)
        results.update(score(output, space))
    report(output, results)
    write_json(output / "analysis_status.json", {"status": "complete", "phase": "additional_comparison_complete"})


if __name__ == "__main__":
    main()
