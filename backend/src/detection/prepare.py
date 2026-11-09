"""H1: prepare detection inputs from paired, reused, source-slice or control data."""

import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.src.common.storage import (
    calculate_file_sha256,
    calculate_source_fingerprints,
    link_or_copy_frozen_file,
    read_json,
    read_jsonl,
    read_toml_config,
    relocate_path,
    write_json,
)
from backend.src.data.definitions import SAFE_SPLITS
from backend.src.data.paired import read_paired_detection_prompts
from backend.src.data.prompts import normalize_prompt_text, select_unsafe_prompts
from backend.src.detection.contracts import IMPLEMENTATION_FILES
from backend.src.fields.definitions import VARIANTS
from backend.src.fields.signatures import (
    build_source_slice_bank,
    load_signature_space_config,
    read_probe_bank,
    resolve_probe_bank_path,
)
from backend.src.inference.encoding import save_atomic_tensor

SPLIT = "additional_safe_test"


def _read_safe_source(settings):
    """Input: explicit source paths. Output: validated safe source and split records."""
    paths = {k: relocate_path(v) for k, v in settings["paths"].items()}
    source, output = paths["source"], paths["output"]
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use an independent output folder")
    original = read_json(source / "experiment.json")
    safe = {
        s: read_jsonl(source / "trajectory/splits" / f"{s}.jsonl") for s in SAFE_SPLITS
    }
    for split in SAFE_SPLITS:
        if [r["id"] for r in safe[split]] != original["split_ids"][split]:
            raise ValueError("Safe splits differ from their frozen manifest")
    return SimpleNamespace(
        paths=paths, source=source, output=output, original=original, safe=safe
    )


def _select_unsafe_partitions(inputs, settings):
    """Input: raw prompt source and safe records. Output: disjoint unsafe partitions."""
    inputs.unsafe = select_unsafe_prompts(
        inputs.paths["dataset"], inputs.safe, settings["experiment"]
    )
    inputs.groups = dict(inputs.safe, **inputs.unsafe)
    return inputs


def _build_detection_manifest(inputs, settings):
    """Input: selected partitions and settings. Output: provenance-bound manifest."""
    paths, source, output = inputs.paths, inputs.source, inputs.output
    original, unsafe, groups = inputs.original, inputs.unsafe, inputs.groups
    config = copy.deepcopy(original["config"])
    config["paths"]["output"] = str(output)
    manifest = {
        "config": config,
        "settings": settings,
        "source": str(paths["source"]),
        "source_experiment_sha256": calculate_file_sha256(source / "experiment.json"),
        "dataset_sha256": calculate_file_sha256(paths["dataset"]),
        "dataset_url": settings.get("dataset", {}).get("url", ""),
        "model_revision": original["model_revision"],
        "bandwidth_scale": original["bandwidth_scale"],
        "counts": {s: len(r) for s, r in groups.items()},
        "category_counts": {
            s: dict(Counter(r["category"] for r in rows)) for s, rows in unsafe.items()
        },
        "split_ids": {s: [r["id"] for r in rows] for s, rows in groups.items()},
        "variants": {s: list(v) for s, v in VARIANTS.items()},
        "modes": ["safe_attraction", "safe_attraction_unsafe_avoidance"],
        "unsafe_cutoff_rule": "95% unsafe calibration conformal radius, capped at safe calibration 5% lower order statistic",
        "feature_policy": "Reuse all safe splits, features, distances and paired frozen banks; extract only new unsafe signatures",
        "meaning": "Prompt-label detection and frozen-probe field steering; no image generation or image safety measurement.",
    }
    return manifest


def _validate_existing_manifest(inputs, manifest):
    """Input: requested manifest. Output: rejection if a frozen run would change."""
    output = inputs.output
    existing = output / "experiment.json"
    if existing.exists() and read_json(existing) != manifest:
        raise ValueError(
            "Existing detection manifest differs; use a fresh output folder"
        )


def _copy_frozen_artifacts(inputs):
    """Input: validated source and partitions. Output: linked banks, features and splits."""
    source, output, safe, unsafe = (
        inputs.source,
        inputs.output,
        inputs.safe,
        inputs.unsafe,
    )
    for space in VARIANTS:
        for split in SAFE_SPLITS:
            link_or_copy_frozen_file(
                source / space / "splits" / f"{split}.jsonl",
                output / space / "splits" / f"{split}.jsonl",
            )
            for row in safe[split]:
                relative = Path("features") / split / f"{row['id']}.pt"
                link_or_copy_frozen_file(
                    source / space / relative, output / space / relative
                )
        for split, rows in unsafe.items():
            target = output / space / "splits" / f"{split}.jsonl"
            content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            if target.exists() and target.read_text(encoding="utf-8") != content:
                raise ValueError("Frozen unsafe split changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        link_or_copy_frozen_file(
            source / space / "probes/probe_bank.pt",
            output / space / "probes/probe_bank.pt",
        )
        link_or_copy_frozen_file(
            source / space / "distances.npz", output / space / "safe_distances.npz"
        )
        link_or_copy_frozen_file(
            source / space / "distances.provenance.json",
            output / space / "safe_distances.provenance.json",
        )


def _write_preparation_receipt(inputs, manifest):
    """Input: frozen manifest. Output: prepared status and manifest under experiment output."""
    output = inputs.output
    existing = output / "experiment.json"
    write_json(existing, manifest)
    write_json(
        output / "status.json",
        {"status": "prepared", "phase": "prepared", "counts": manifest["counts"]},
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": manifest["counts"],
                "categories": manifest["category_counts"],
            },
            indent=2,
        )
    )
    return output


def load_detection_inputs(settings):
    """Input: settings. Output: validated safe inputs and selected unsafe partitions."""
    inputs = _read_safe_source(settings)
    return _select_unsafe_partitions(inputs, settings)


def freeze_detection_manifest(inputs, settings):
    """Input: partitions and settings. Output: manifest compatible with existing artifacts."""
    manifest = _build_detection_manifest(inputs, settings)
    _validate_existing_manifest(inputs, manifest)
    return manifest


def persist_detection_inputs(inputs, manifest):
    """Input: validated manifest and source. Output: prepared experiment directory."""
    _copy_frozen_artifacts(inputs)
    return _write_preparation_receipt(inputs, manifest)


def freeze_paired_detection_inputs(settings):
    dataset = relocate_path(settings["paths"]["dataset"])
    output = Path(settings["paths"]["output"])
    if (output / "experiment.json").exists():
        raise ValueError("Fresh preparation requires a new output directory")
    columns = settings.get("dataset", {}).get("prompt_columns")
    groups, audit = read_paired_detection_prompts(dataset, prompt_columns=columns)
    manifest = {
        "config": settings,
        "implementation": calculate_source_fingerprints(IMPLEMENTATION_FILES),
        "dataset_sha256": calculate_file_sha256(dataset),
        "counts": {s: len(rows) for s, rows in groups.items()},
        "dataset_audit": audit,
        "split_ids": {s: [row["id"] for row in rows] for s, rows in groups.items()},
        "split_sha256": {},
        "variants": {"trajectory": ["euclidean"]},
        "reference_policy": (
            f"training columns: {columns}"
            if columns
            else "legal train=safe; illegal train=unsafe"
        ),
        "calibration_policy": "validation only; no test-dependent tuning",
        "probe_policy": "fresh trajectories from safe training prompts only",
        "scope": "prompt-label detection only; no steering, repair, or image generation",
    }
    for split, rows in groups.items():
        path = output / "trajectory/splits" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest["split_sha256"][split] = calculate_file_sha256(path)
    write_json(output / "experiment.json", manifest)
    write_json(
        output / "status.json", {"status": "prepared", "counts": manifest["counts"]}
    )
    return output


def load_paired_source_inputs(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output in source.parents:
        raise ValueError("Use a separate source-method output")
    manifest = read_json(source / "experiment.json")
    cfg = load_signature_space_config(source, "trajectory")
    bank = read_probe_bank(cfg)
    for split, ids in manifest["split_ids"].items():
        path = source / "trajectory/splits" / f"{split}.jsonl"
        if calculate_file_sha256(path) != manifest["split_sha256"][split]:
            raise ValueError("Frozen parent split changed")
        if [row["id"] for row in read_jsonl(path)] != ids:
            raise ValueError("Frozen parent split order changed")
    return SimpleNamespace(
        source=source,
        output=output,
        original=manifest,
        bank=bank,
        parent_bank=resolve_probe_bank_path(cfg),
    )


def _freeze_source_manifest(inputs):
    manifest = copy.deepcopy(inputs.original)
    manifest["config"]["paths"]["output"] = str(inputs.output)
    manifest["config"]["probes"]["time_points"] = 1
    manifest["config"]["scoring"] = {"method": "anchor_static"}
    manifest["probe_policy"] = (
        "Reconstruct paired Gaussian source states from the frozen training-only trajectory seeds at t=0"
    )
    manifest.update(
        variants={"source_slice": ["anchor_static", "anchor_adaptive"]},
        scope="detection_only",
        source_lineage={
            "experiment_sha256": calculate_file_sha256(
                inputs.source / "experiment.json"
            ),
            "trajectory_bank_sha256": calculate_file_sha256(inputs.parent_bank),
            "geometry": f"{inputs.bank['trajectories']} paired Gaussian source states at flow time zero",
            "feature_reuse": "Trajectory signatures do not contain time-zero fields",
        },
    )
    path = inputs.output / "experiment.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Existing source manifest differs; use a fresh output")
    write_json(path, manifest)


def _copy_source_splits(inputs):
    for path in (inputs.source / "trajectory/splits").glob("*.jsonl"):
        link_or_copy_frozen_file(
            path, inputs.output / "source_slice/splits" / path.name
        )


def _persist_source_bank(inputs):
    bank = build_source_slice_bank(inputs.bank)
    path = inputs.output / "source_slice/probes/probe_bank.pt"
    if path.exists():
        existing = read_json(inputs.output / "source_slice/probes/provenance.json")
        if existing["bank_sha256"] != calculate_file_sha256(path):
            raise ValueError("Source bank bytes changed")
    else:
        save_atomic_tensor(path, bank)
        write_json(
            inputs.output / "source_slice/probes/provenance.json",
            {
                "bank_sha256": calculate_file_sha256(path),
                "parent_trajectory_bank_sha256": calculate_file_sha256(
                    inputs.parent_bank
                ),
                "construction": "CPU torch.randn from each paired trajectory seed; t=0; uniform weights",
                "training_fit_sha256": bank["fit_sha256"],
                "provenance": bank["provenance"],
            },
        )
    read_probe_bank(load_signature_space_config(inputs.output, "source_slice"))
    return inputs.output


def freeze_paired_source_inputs(inputs):
    _freeze_source_manifest(inputs)
    _copy_source_splits(inputs)
    return _persist_source_bank(inputs)


def _read_control_preparation_inputs(source, parent, output):
    source, parent, output = source.resolve(), parent.resolve(), output.resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    inputs = payload["prompts"] if isinstance(payload, dict) else payload
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    original = read_json(parent / "experiment.json")
    receipt = read_json(parent / "comparison.provenance.json")
    for relative, digest in receipt["artifacts"].items():
        if calculate_file_sha256(parent / relative) != digest:
            raise ValueError(f"Parent result changed: {relative}")
    excluded = {
        normalize_prompt_text(r["prompt"])
        for path in (parent / "trajectory/splits").glob("*.jsonl")
        for r in read_jsonl(path)
    }
    seen, rows = set(), []
    for item in inputs:
        prompt = item.get("prompt", "").strip()
        key = normalize_prompt_text(prompt)
        if not key or key in seen or key in excluded:
            raise ValueError(
                "Additional tests must be nonempty, canonically unique, and disjoint from every parent split"
            )
        if item.get("label") != "benign_intended":
            raise ValueError(
                "This evaluation expects explicitly intended-benign controls"
            )
        seen.add(key)
        rows.append(
            dict(
                item,
                source_id=item.get("id"),
                id=hashlib.sha256(key.encode()).hexdigest(),
                source="synthetic_style_matched_controls",
                split=SPLIT,
                eligible_for_region=False,
                evaluation_only=True,
                category=item.get("aligned_unsafe_category"),
            )
        )
    if not rows:
        raise ValueError("No additional test prompts")
    return SimpleNamespace(
        source=source,
        parent=parent,
        output=output,
        original=original,
        rows=rows,
        metadata=metadata,
    )


def _build_control_manifest(inputs):
    source, parent, output = inputs.source, inputs.parent, inputs.output
    original, rows, metadata = inputs.original, inputs.rows, inputs.metadata
    config = copy.deepcopy(original["config"])
    config["paths"]["output"] = str(output)
    frozen = {
        f"{space}_{method}": read_json(
            parent / "variants" / f"{space}_{method}" / "calibration.json"
        )
        for space, methods in VARIANTS.items()
        for method in methods
    }
    manifest = {
        "config": config,
        "counts": {SPLIT: len(rows)},
        "split_ids": {SPLIT: [r["id"] for r in rows]},
        "parent": str(parent),
        "parent_experiment_sha256": calculate_file_sha256(parent / "experiment.json"),
        "parent_comparison_sha256": calculate_file_sha256(parent / "comparison.json"),
        "input_file": str(source),
        "input_sha256": calculate_file_sha256(source),
        "input_metadata": metadata,
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "model_revision": original["model_revision"],
        "settings": original["settings"],
        "reference_ids": {
            s: original["split_ids"][s]
            for s in ("safe_reference", "unsafe_reference", "unsafe_test")
        },
        "policy": "Additional test only; preserve banks, anchors, calibration, bandwidths, cutoffs, and original test results",
        "label_meaning": "Synthetic intended-benign controls; not human safety-validated or generated-image-validated",
    }
    if (output / "experiment.json").exists() and {
        k: v
        for k, v in read_json(output / "experiment.json").items()
        if k != "frozen_geometry_sha256"
    } != manifest:
        raise ValueError("Additional test inputs changed; use a fresh output folder")
    return manifest, frozen


def _copy_control_artifacts(inputs, manifest, frozen):
    source, parent, output = inputs.source, inputs.parent, inputs.output
    original, rows = inputs.original, inputs.rows
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / "input.json")
    write_json(output / "frozen_calibration.json", frozen)
    shutil.copy2(parent / "comparison.json", output / "parent_comparison.json")
    for space in VARIANTS:
        folder = output / space
        for split in ("safe_reference", "unsafe_reference", "unsafe_test"):
            link_or_copy_frozen_file(
                parent / space / "splits" / f"{split}.jsonl",
                folder / "splits" / f"{split}.jsonl",
            )
        path = folder / "splits" / f"{SPLIT}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        link_or_copy_frozen_file(
            parent / space / "probes/probe_bank.pt", folder / "probes/probe_bank.pt"
        )
        hashes = {}
        for split in ("safe_reference", "unsafe_reference"):
            parent_receipt = read_json(
                parent
                / space
                / (
                    "safe_distances.provenance.json"
                    if split == "safe_reference"
                    else "unsafe_features.provenance.json"
                )
            )
            for identifier in original["split_ids"][split]:
                relative = f"features/{split}/{identifier}.pt"
                link_or_copy_frozen_file(parent / space / relative, folder / relative)
                hashes[f"{split}/{identifier}.pt"] = parent_receipt["features"][
                    f"{split}/{identifier}.pt"
                ]
        write_json(
            folder / "reference_features.provenance.json",
            {
                "features": hashes,
                "bank_sha256": calculate_file_sha256(folder / "probes/probe_bank.pt"),
            },
        )
        with np.load(parent / space / "two_mode_distances.npz") as cache:
            np.savez_compressed(
                folder / "frozen_geometry.npz",
                **{
                    key: cache[key]
                    for key in (
                        "safe__safe_reference",
                        "unsafe__safe_reference",
                        "safe__unsafe_test",
                        "unsafe__unsafe_test",
                    )
                },
            )
        manifest.setdefault("frozen_geometry_sha256", {})[space] = (
            calculate_file_sha256(folder / "frozen_geometry.npz")
        )


def _persist_control_manifest(inputs, manifest):
    output, rows = inputs.output, inputs.rows
    # Geometry hashes are derived from immutable parent matrices; compare after
    # preparation so the complete manifest also protects this frozen state.
    existing = output / "experiment.json"
    if existing.exists() and read_json(existing) != manifest:
        raise ValueError("Frozen additional-test manifest changed")
    write_json(existing, manifest)
    if not (output / "status.json").exists():
        write_json(output / "status.json", {"status": "prepared", "phase": "prepared"})
    print(
        json.dumps(
            {
                "output": str(output),
                "count": len(rows),
                "categories": manifest["category_counts"],
            },
            indent=2,
        )
    )
    return output


def freeze_control_inputs(source, parent, output):
    inputs = _read_control_preparation_inputs(source, parent, output)
    manifest, frozen = _build_control_manifest(inputs)
    _copy_control_artifacts(inputs, manifest, frozen)
    return _persist_control_manifest(inputs, manifest)


def run_detection_preparation(
    config: Path | dict, output: Path, *, mode: str = "paired"
) -> Path:
    """Freeze inputs in the selected artifact mode; return the prepared directory."""
    if mode not in {"paired", "reused", "source_slice", "controls"}:
        raise ValueError("Unknown detection preparation mode")
    settings = (
        read_toml_config(config)
        if not isinstance(config, dict)
        else copy.deepcopy(config)
    )
    output = Path(output).resolve()
    settings.setdefault("paths", {})["output"] = str(output)
    if mode == "paired":
        freeze_paired_detection_inputs(settings)
        return output
    if mode == "reused":
        inputs = load_detection_inputs(settings)
        manifest = freeze_detection_manifest(inputs, settings)
        return persist_detection_inputs(inputs, manifest)
    source = relocate_path(settings["paths"]["source"])
    if mode == "source_slice":
        inputs = load_paired_source_inputs(source, output)
        return freeze_paired_source_inputs(inputs)
    dataset = relocate_path(settings["paths"]["dataset"])
    return freeze_control_inputs(dataset, source, output)
