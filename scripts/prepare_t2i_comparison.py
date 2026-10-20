"""Freeze new unsafe partitions and reuse the existing safe experiment artifacts."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from backend.flow.common import ROOT, load_config, sha256, write_json
from backend.flow.t2i_prompts import SAFE_SPLITS, read_json, read_jsonl, select_unsafe_prompts
from backend.flow.comparison import VARIANTS, link_file, safe_source_path


def prepare(settings):
    paths = {k: (ROOT / v).resolve() for k, v in settings["paths"].items()}
    source, output = paths["source"], paths["output"]
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use an independent output folder")
    source = safe_source_path(source)
    original = read_json(source / "experiment.json")
    safe = {s: read_jsonl(source / "trajectory/splits" / f"{s}.jsonl") for s in SAFE_SPLITS}
    for split in SAFE_SPLITS:
        if [r["id"] for r in safe[split]] != original["split_ids"][split]:
            raise ValueError("Safe splits differ from their frozen manifest")
    unsafe = select_unsafe_prompts(paths["dataset"], safe, settings["experiment"])
    groups = dict(safe, **unsafe)
    config = copy.deepcopy(original["config"])
    config["paths"]["output"] = str(output)
    manifest = {"config": config, "settings": settings,
                "source": str(paths["source"]), "source_experiment_sha256": sha256(source / "experiment.json"),
                "dataset_sha256": sha256(paths["dataset"]),
                "dataset_url": "https://github.com/datar001/T2I-RiskyPrompt",
                "model_revision": original["model_revision"], "bandwidth_scale": original["bandwidth_scale"],
                "counts": {s: len(r) for s, r in groups.items()},
                "category_counts": {s: dict(Counter(r["category"] for r in rows)) for s, rows in unsafe.items()},
                "split_ids": {s: [r["id"] for r in rows] for s, rows in groups.items()},
                "variants": {s: list(v) for s, v in VARIANTS.items()},
                "modes": ["safe_attraction", "safe_attraction_unsafe_avoidance"],
                "unsafe_cutoff_rule": "95% unsafe calibration conformal radius, capped at safe calibration 5% lower order statistic",
                "feature_policy": "Reuse all safe splits, features, distances and paired frozen banks; extract only new T2I unsafe signatures",
                "meaning": "Prompt-label detection and frozen-probe field steering; no image generation or image safety measurement."}
    existing = output / "experiment.json"
    if existing.exists() and read_json(existing) != manifest:
        raise ValueError("Existing T2I manifest differs; use a fresh output folder")
    for space in VARIANTS:
        for split in SAFE_SPLITS:
            link_file(source / space / "splits" / f"{split}.jsonl", output / space / "splits" / f"{split}.jsonl")
            for row in safe[split]:
                relative = Path("features") / split / f"{row['id']}.pt"
                link_file(source / space / relative, output / space / relative)
        for split, rows in unsafe.items():
            target = output / space / "splits" / f"{split}.jsonl"
            content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            if target.exists() and target.read_text(encoding="utf-8") != content:
                raise ValueError("Frozen unsafe split changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        link_file(source / space / "probes/probe_bank.pt", output / space / "probes/probe_bank.pt")
        link_file(source / space / "distances.npz", output / space / "safe_distances.npz")
        link_file(source / space / "distances.provenance.json", output / space / "safe_distances.provenance.json")
    write_json(existing, manifest)
    write_json(output / "status.json", {"status": "prepared", "phase": "prepared", "counts": manifest["counts"]})
    print(json.dumps({"output": str(output), "counts": manifest["counts"], "categories": manifest["category_counts"]}, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/t2i_two_mode_comparison.toml")
    args = parser.parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
