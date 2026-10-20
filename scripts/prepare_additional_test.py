"""Freeze a supplied intended-benign test set alongside the completed T2I run."""

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from backend.flow.common import ROOT, sha256, write_json
from backend.flow.prompt_records import canonical
from backend.flow.t2i_prompts import read_json, read_jsonl
from backend.flow.comparison import VARIANTS, link_file

PARENT = ROOT / "data/flow_t2i_two_mode_1000"
OUTPUT = PARENT / "additional_tests/style_matched_safe_prompts_1000"
SPLIT = "additional_safe_test"


def prepare(source, parent=PARENT, output=OUTPUT):
    source, parent, output = source.resolve(), parent.resolve(), output.resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    inputs = payload["prompts"] if isinstance(payload, dict) else payload
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    original = read_json(parent / "experiment.json")
    receipt = read_json(parent / "comparison.provenance.json")
    for relative, digest in receipt["artifacts"].items():
        if sha256(parent / relative) != digest:
            raise ValueError(f"Parent result changed: {relative}")
    excluded = {canonical(r["prompt"]) for path in (parent / "trajectory/splits").glob("*.jsonl")
                for r in read_jsonl(path)}
    seen, rows = set(), []
    for item in inputs:
        prompt = item.get("prompt", "").strip()
        key = canonical(prompt)
        if not key or key in seen or key in excluded:
            raise ValueError("Additional tests must be nonempty, canonically unique, and disjoint from every parent split")
        if item.get("label") != "benign_intended":
            raise ValueError("This evaluation expects explicitly intended-benign controls")
        seen.add(key)
        rows.append(dict(item, source_id=item.get("id"), id=hashlib.sha256(key.encode()).hexdigest(),
                         source="synthetic_style_matched_controls", split=SPLIT,
                         eligible_for_region=False, evaluation_only=True,
                         category=item.get("aligned_unsafe_category")))
    if not rows:
        raise ValueError("No additional test prompts")
    config = copy.deepcopy(original["config"])
    config["paths"]["output"] = str(output)
    frozen = {f"{space}_{method}": read_json(parent / "variants" / f"{space}_{method}" / "calibration.json")
              for space, methods in VARIANTS.items() for method in methods}
    manifest = {"config": config, "counts": {SPLIT: len(rows)}, "split_ids": {SPLIT: [r["id"] for r in rows]},
                "parent": str(parent), "parent_experiment_sha256": sha256(parent / "experiment.json"),
                "parent_comparison_sha256": sha256(parent / "comparison.json"),
                "input_file": str(source), "input_sha256": sha256(source), "input_metadata": metadata,
                "category_counts": dict(Counter(r["category"] for r in rows)),
                "model_revision": original["model_revision"], "settings": original["settings"],
                "reference_ids": {s: original["split_ids"][s] for s in ("safe_reference", "unsafe_reference", "unsafe_test")},
                "policy": "Additional test only; preserve banks, anchors, calibration, bandwidths, cutoffs, and original test results",
                "label_meaning": "Synthetic intended-benign controls; not human safety-validated or generated-image-validated"}
    if ((output / "experiment.json").exists() and
            {k: v for k, v in read_json(output / "experiment.json").items() if k != "frozen_geometry_sha256"} != manifest):
        raise ValueError("Additional test inputs changed; use a fresh output folder")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / "input.json")
    write_json(output / "frozen_calibration.json", frozen)
    shutil.copy2(parent / "comparison.json", output / "parent_comparison.json")
    for space in VARIANTS:
        folder = output / space
        for split in ("safe_reference", "unsafe_reference", "unsafe_test"):
            link_file(parent / space / "splits" / f"{split}.jsonl", folder / "splits" / f"{split}.jsonl")
        path = folder / "splits" / f"{SPLIT}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
        link_file(parent / space / "probes/probe_bank.pt", folder / "probes/probe_bank.pt")
        hashes = {}
        for split in ("safe_reference", "unsafe_reference"):
            parent_receipt = read_json(parent / space / ("safe_distances.provenance.json" if split == "safe_reference"
                                                         else "unsafe_features.provenance.json"))
            for identifier in original["split_ids"][split]:
                relative = f"features/{split}/{identifier}.pt"
                link_file(parent / space / relative, folder / relative)
                hashes[f"{split}/{identifier}.pt"] = parent_receipt["features"][f"{split}/{identifier}.pt"]
        write_json(folder / "reference_features.provenance.json", {"features": hashes,
                   "bank_sha256": sha256(folder / "probes/probe_bank.pt")})
        with np.load(parent / space / "two_mode_distances.npz") as cache:
            np.savez_compressed(folder / "frozen_geometry.npz", **{key: cache[key] for key in
                ("safe__safe_reference", "unsafe__safe_reference", "safe__unsafe_test", "unsafe__unsafe_test")})
        manifest.setdefault("frozen_geometry_sha256", {})[space] = sha256(folder / "frozen_geometry.npz")
    # Geometry hashes are derived from immutable parent matrices; compare after
    # preparation so the complete manifest also protects this frozen state.
    existing = output / "experiment.json"
    if existing.exists() and read_json(existing) != manifest:
        raise ValueError("Frozen additional-test manifest changed")
    write_json(existing, manifest)
    if not (output / "status.json").exists():
        write_json(output / "status.json", {"status": "prepared", "phase": "prepared"})
    print(json.dumps({"output": str(output), "count": len(rows), "categories": manifest["category_counts"]}, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    source = args.input or max((Path.home() / "Downloads").glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
    prepare(source)


if __name__ == "__main__":
    main()
