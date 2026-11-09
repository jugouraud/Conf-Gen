"""H1: verify frozen signatures and measure distances in explicit artifact modes."""

import hashlib
import io
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from backend.src.common.storage import (
    calculate_file_sha256,
    calculate_source_fingerprints,
    read_json,
    read_jsonl,
    write_json,
)
from backend.src.data.definitions import SAFE_SPLITS, SPLIT, UNSAFE_SPLITS
from backend.src.detection.contracts import read_frozen_control_inputs
from backend.src.fields.definitions import FEATURE_KIND, VARIANTS
from backend.src.fields.distances import (
    calculate_blocked_reference_distance_blocks,
    calculate_reference_distance_blocks,
    calculate_streamed_probe_distances,
    validate_probe_aggregation,
)
from backend.src.fields.signatures import (
    load_signature_space_config,
    load_weighted_signature_vector,
    read_probe_bank,
    resolve_probe_bank_path,
    resolve_signature_path,
)


def _load_distance_inputs(output, space):
    cfg = load_signature_space_config(output, space)
    root = output / space
    bank, bank_hash = (
        read_probe_bank(cfg),
        calculate_file_sha256(resolve_probe_bank_path(cfg)),
    )
    groups = {
        s: read_jsonl(root / "splits" / f"{s}.jsonl")
        for s in SAFE_SPLITS + UNSAFE_SPLITS
    }
    manifest = read_json(output / "experiment.json")
    if any(
        [r["id"] for r in rows] != manifest["split_ids"][s]
        for s, rows in groups.items()
    ):
        raise ValueError("Frozen split IDs changed")
    safe_receipt = read_json(root / "safe_distances.provenance.json")
    unsafe_receipt = read_json(root / "unsafe_features.provenance.json")
    if (
        safe_receipt["bank_sha256"] != bank_hash
        or unsafe_receipt["bank_sha256"] != bank_hash
        or safe_receipt["distances_sha256"]
        != calculate_file_sha256(root / "safe_distances.npz")
        or unsafe_receipt["experiment_sha256"]
        != calculate_file_sha256(output / "experiment.json")
    ):
        raise ValueError("Feature geometry or source cache changed")
    if any(
        safe_receipt["split_ids"][s] != manifest["split_ids"][s] for s in SAFE_SPLITS
    ):
        raise ValueError(
            "Reused safe distance rows differ from the frozen safe split order"
        )
    expected = {
        "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
        "bank_sha256": bank_hash,
        "safe_receipt_sha256": calculate_file_sha256(
            root / "safe_distances.provenance.json"
        ),
        "unsafe_receipt_sha256": calculate_file_sha256(
            root / "unsafe_features.provenance.json"
        ),
        "split_ids": manifest["split_ids"],
        "implementation": "FP64 Euclidean Gram accumulation",
    }
    target, receipt = (
        root / "two_mode_distances.npz",
        root / "two_mode_distances.provenance.json",
    )
    # Validate each feature against its original extraction receipt before any
    # matrix is computed. The new unsafe test never enters a fitted cutoff.
    for split, rows in groups.items():
        hashes = (
            safe_receipt["features"]
            if split in SAFE_SPLITS
            else unsafe_receipt["features"]
        )
        for row in rows:
            if (
                calculate_file_sha256(resolve_signature_path(cfg, split, row["id"]))
                != hashes[f"{split}/{row['id']}.pt"]
            ):
                raise ValueError(
                    f"Feature checksum changed: {space}/{split}/{row['id']}"
                )
    return SimpleNamespace(
        output=output,
        space=space,
        cfg=cfg,
        root=root,
        bank=bank,
        bank_hash=bank_hash,
        groups=groups,
        expected=expected,
        target=target,
        receipt=receipt,
    )


def _validate_distance_cache(inputs):
    if not inputs.target.exists() or not inputs.receipt.exists():
        return False
    if read_json(inputs.receipt) != dict(
        inputs.expected, distances_sha256=calculate_file_sha256(inputs.target)
    ):
        raise ValueError("Two-mode distance cache changed")
    return True


def _compute_distance_tables(inputs, device, batch_size):
    def load_signature_vector(split, row):
        return load_weighted_signature_vector(
            inputs.cfg, split, row, inputs.bank, inputs.bank_hash
        )

    def report_progress(split, completed, total):
        write_json(
            inputs.output / "analysis_status.json",
            {
                "status": "running",
                "phase": "distances",
                "space": inputs.space,
                "references": reference_split,
                "split": split,
                "completed": completed,
                "total": total,
            },
        )

    reference_split = "unsafe_reference"
    unsafe = calculate_reference_distance_blocks(
        inputs.groups,
        reference_split,
        SAFE_SPLITS + UNSAFE_SPLITS,
        load_signature_vector,
        device=device,
        batch_size=batch_size,
        progress=report_progress,
    )
    np.fill_diagonal(unsafe["unsafe_reference"], 0.0)
    with np.load(inputs.root / "safe_distances.npz") as cached:
        safe = {s: cached[s].copy() for s in SAFE_SPLITS}
    safe["unsafe_reference"] = unsafe["safe_reference"].T.copy()
    reference_split = "safe_reference"
    safe.update(
        calculate_reference_distance_blocks(
            inputs.groups,
            reference_split,
            ("unsafe_validation", "unsafe_test"),
            load_signature_vector,
            device=device,
            batch_size=batch_size,
            progress=report_progress,
        )
    )
    return {
        **{f"safe__{s}": v for s, v in safe.items()},
        **{f"unsafe__{s}": v for s, v in unsafe.items()},
    }


def _persist_distance_tables(inputs, tables):
    np.savez_compressed(inputs.target, **tables)
    write_json(
        inputs.receipt,
        dict(inputs.expected, distances_sha256=calculate_file_sha256(inputs.target)),
    )


def measure_signature_space_distances(output, space, device="cuda", batch_size=16):
    inputs = _load_distance_inputs(output, space)
    if _validate_distance_cache(inputs):
        return
    tables = _compute_distance_tables(inputs, device, batch_size)
    _persist_distance_tables(inputs, tables)


def load_paired_distance_inputs(output, space="trajectory"):
    output = Path(output).resolve()
    manifest = read_json(output / "experiment.json")
    cfg = load_signature_space_config(output, space)
    bank = read_probe_bank(cfg)
    bank_hash = calculate_file_sha256(resolve_probe_bank_path(cfg))
    receipt = read_json(output / space / "features.provenance.json")
    if (
        receipt["experiment_sha256"]
        != calculate_file_sha256(output / "experiment.json")
        or receipt["bank_sha256"] != bank_hash
    ):
        raise ValueError("Extraction provenance changed")
    groups = {}
    for split in manifest["counts"]:
        path = output / space / "splits" / f"{split}.jsonl"
        if calculate_file_sha256(path) != manifest["split_sha256"][split]:
            raise ValueError("Split bytes changed")
        groups[split] = read_jsonl(path)
        for row in groups[split]:
            feature = resolve_signature_path(cfg, split, row["id"])
            if (
                calculate_file_sha256(feature)
                != receipt["features"][f"{split}/{row['id']}.pt"]
            ):
                raise ValueError("Signature bytes changed")
    return output, cfg, bank, bank_hash, groups


def measure_paired_reference_distances(
    cfg, bank, bank_hash, groups, device, output, include_training=False
):
    def load_vector(split, row):
        return load_weighted_signature_vector(cfg, split, row, bank, bank_hash)

    def progress(split, completed, total):
        write_json(
            output / "analysis_status.json",
            {
                "status": "running",
                "phase": f"measure/{split}",
                "completed": completed,
                "total": total,
            },
        )

    # Only validation/test queries are needed; train rows define the regions.
    queries = ("safe_validation", "unsafe_validation", "safe_test", "unsafe_test")
    tables = {}
    for label in ("safe", "unsafe"):
        measure = (
            calculate_blocked_reference_distance_blocks
            if str(device).startswith("cuda") and bank["time_points"] > 1
            else calculate_reference_distance_blocks
        )
        blocks = measure(
            groups,
            f"{label}_reference",
            (*queries, f"{label}_reference") if include_training else queries,
            load_vector,
            device=device,
            batch_size=16,
            progress=progress,
        )
        tables.update({f"{label}__{split}": values for split, values in blocks.items()})
    return tables


def persist_paired_distances(output, tables, bank_hash, space="trajectory"):
    target = output / space / "distances.npz"
    np.savez_compressed(target, **tables)
    write_json(
        output / space / "distances.provenance.json",
        {
            "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
            "distances_sha256": calculate_file_sha256(target),
            "bank_sha256": bank_hash,
            "features_receipt_sha256": calculate_file_sha256(
                output / space / "features.provenance.json"
            ),
            "implementation": "FP64 Euclidean; uniform probe quadrature; train-only references; trajectory GPU reference blocks capped at 128",
            "reference_splits": ["safe_reference", "unsafe_reference"],
        },
    )


def load_probe_measurement_inputs(source, output, space, aggregation):
    validate_probe_aggregation(aggregation)
    if space not in VARIANTS:
        raise ValueError("Unknown signature space")
    source, output = Path(source).resolve(), Path(output).resolve()
    if (
        output == source
        or output in source.parents
        or output.is_relative_to(source / space)
    ):
        raise ValueError("Use a separate distance output")
    manifest = read_json(source / "experiment.json")
    root = source / space
    features = read_json(root / "features.provenance.json")
    bank_hash = calculate_file_sha256(root / "probes/probe_bank.pt")
    if (
        features["experiment_sha256"]
        != calculate_file_sha256(source / "experiment.json")
        or features["bank_sha256"] != bank_hash
    ):
        raise ValueError("Signature provenance changed")
    cfg = load_signature_space_config(source, space)
    bank = read_probe_bank(cfg)
    groups = {}
    for split, ids in manifest["split_ids"].items():
        path = root / "splits" / f"{split}.jsonl"
        if calculate_file_sha256(path) != manifest["split_sha256"][split]:
            raise ValueError("Frozen split changed")
        groups[split] = read_jsonl(path)
        if (
            len(groups[split]) != manifest["counts"][split]
            or [r["id"] for r in groups[split]] != ids
        ):
            raise ValueError("Frozen split IDs/counts changed")
    expected = {
        f"{split}/{row['id']}.pt" for split, rows in groups.items() for row in rows
    }
    if set(features["features"]) != expected:
        raise ValueError("Incomplete source signatures")
    provenance = {
        "source": str(source),
        "space": space,
        "probe_aggregation": aggregation,
        "source_experiment_sha256": calculate_file_sha256(source / "experiment.json"),
        "features_receipt_sha256": calculate_file_sha256(
            root / "features.provenance.json"
        ),
        "bank_sha256": bank_hash,
        "counts": manifest["counts"],
        "metric": "max_j ||F_p(j)-F_a(j)||_2"
        if aggregation == "max"
        else "sqrt(sum_j w_j ||F_p(j)-F_a(j)||_2^2)",
        "implementation": calculate_source_fingerprints(
            [
                "backend/src/fields/distances.py",
                "backend/src/detection/measure.py",
                "backend/src/detection/calibration.py",
            ]
        ),
    }
    target = output / "experiment.json"
    if target.exists() and read_json(target) != provenance:
        raise ValueError("Existing aggregation provenance differs; use a fresh output")
    write_json(target, provenance)
    return SimpleNamespace(
        source=source,
        output=output,
        space=space,
        cfg=cfg,
        bank=bank,
        bank_hash=bank_hash,
        manifest=manifest,
        groups=groups,
        features=features,
        aggregation=aggregation,
    )


def _read_measurement_checkpoint(inputs):
    path = inputs.output / "checkpoint.npz"
    receipt = inputs.output / "checkpoint.json"
    if not path.exists() or not receipt.exists():
        return None
    saved = read_json(receipt)
    if saved["experiment_sha256"] != calculate_file_sha256(
        inputs.output / "experiment.json"
    ) or saved["sha256"] != calculate_file_sha256(path):
        raise ValueError("Probe-distance checkpoint changed")
    with np.load(path) as archive:
        return saved["next_probe"], {key: archive[key].copy() for key in archive.files}


def await_frozen_signature_source(source, output, space, wait):
    path = Path(source) / space / "features.provenance.json"
    while wait and not path.exists():
        write_json(
            Path(output) / "status.json",
            {
                "status": "waiting",
                "phase": "waiting_for_source_signatures",
                "space": space,
            },
        )
        collection = Path(source) / "gpu/collection_status.json"
        if collection.exists() and read_json(collection).get("status") == "failed":
            raise RuntimeError(
                "Source signature collection failed; resume its collector before rescoring"
            )
        time.sleep(30)


def _measure_probe_tables(inputs, working_directory, device, resumed):
    target = inputs.output / "distances.npz"
    receipt = inputs.output / "distances.provenance.json"
    if target.exists() and receipt.exists():
        saved = read_json(receipt)
        if saved["experiment_sha256"] != calculate_file_sha256(
            inputs.output / "experiment.json"
        ) or saved["distances_sha256"] != calculate_file_sha256(target):
            raise ValueError("Saved probe distances changed")
        return target

    def load_values(split, row):
        path = resolve_signature_path(inputs.cfg, split, row["id"])
        data = path.read_bytes()
        if (
            hashlib.sha256(data).hexdigest()
            != inputs.features["features"][f"{split}/{row['id']}.pt"]
        ):
            raise ValueError(f"Signature bytes changed: {split}/{row['id']}")
        feature = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if (
            feature["bank_sha256"] != inputs.bank_hash
            or feature["kind"] != FEATURE_KIND
            or feature["id"] != row["id"]
            or feature["values"].shape != inputs.bank["states"].shape
            or feature["adapter_signature"] != inputs.features["adapter_signature"]
        ):
            raise ValueError("Incompatible signature geometry or model")
        return feature["values"]

    def progress(phase, completed, total):
        temporary = inputs.output / "status.tmp"
        write_json(
            temporary,
            {
                "status": "complete" if phase == "distances_complete" else "running",
                "phase": phase,
                "completed": completed,
                "total": total,
                "space": inputs.space,
                "probe_aggregation": inputs.aggregation,
            },
        )
        temporary.replace(inputs.output / "status.json")

    def checkpoint(next_probe, tables):
        path = inputs.output / "checkpoint.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **tables)
        temporary.replace(path)
        write_json(
            inputs.output / "checkpoint.json",
            {
                "experiment_sha256": calculate_file_sha256(
                    inputs.output / "experiment.json"
                ),
                "next_probe": next_probe,
                "sha256": calculate_file_sha256(path),
            },
        )

    tables = calculate_streamed_probe_distances(
        inputs.groups,
        load_values,
        inputs.bank["weights"],
        probe_aggregation=inputs.aggregation,
        working_directory=working_directory,
        device=device,
        progress=progress,
        checkpoint=checkpoint,
        resumed=resumed,
    )
    with target.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, **tables)
    target.with_suffix(".tmp").replace(target)
    write_json(
        receipt,
        {
            "experiment_sha256": calculate_file_sha256(
                inputs.output / "experiment.json"
            ),
            "distances_sha256": calculate_file_sha256(target),
            "bank_sha256": inputs.bank_hash,
            "probe_aggregation": inputs.aggregation,
            "reference_splits": ["safe_reference", "unsafe_reference"],
            "training_pairs": "Full within-class reference matrices for bandwidth fitting; diagonal excluded by scorer",
            "precision": "FP64 pairwise Euclidean products at each probe, before probe reduction",
        },
    )
    progress(
        "distances_complete", len(inputs.bank["weights"]), len(inputs.bank["weights"])
    )
    return target


def _load_control_distance_inputs(output, space, frozen_inputs):
    cfg = load_signature_space_config(output, space)
    folder = output / space
    manifest, _, _ = frozen_inputs
    bank, bank_hash = (
        read_probe_bank(cfg),
        calculate_file_sha256(resolve_probe_bank_path(cfg)),
    )
    reference_receipt = read_json(folder / "reference_features.provenance.json")
    query_receipt = read_json(folder / "additional_features.provenance.json")
    rows = read_jsonl(folder / "splits" / f"{SPLIT}.jsonl")
    if [r["id"] for r in rows] != manifest["split_ids"][SPLIT]:
        raise ValueError("Additional test ordering changed")
    if (
        reference_receipt["bank_sha256"] != bank_hash
        or query_receipt["bank_sha256"] != bank_hash
        or query_receipt["experiment_sha256"]
        != calculate_file_sha256(output / "experiment.json")
    ):
        raise ValueError("Additional feature provenance changed")
    groups = {
        s: read_jsonl(folder / "splits" / f"{s}.jsonl")
        for s in ("safe_reference", "unsafe_reference")
    }
    for split, group in dict(groups, **{SPLIT: rows}).items():
        hashes = (
            query_receipt["features"]
            if split == SPLIT
            else reference_receipt["features"]
        )
        if (
            split != SPLIT
            and [r["id"] for r in group] != manifest["reference_ids"][split]
        ):
            raise ValueError("Frozen anchor ordering changed")
        for row in group:
            if (
                calculate_file_sha256(resolve_signature_path(cfg, split, row["id"]))
                != hashes[f"{split}/{row['id']}.pt"]
            ):
                raise ValueError("Additional-test input feature changed")
    target, receipt_path = (
        folder / "additional_distances.npz",
        folder / "additional_distances.provenance.json",
    )
    expected = {
        "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
        "bank_sha256": bank_hash,
        "reference_receipt_sha256": calculate_file_sha256(
            folder / "reference_features.provenance.json"
        ),
        "query_receipt_sha256": calculate_file_sha256(
            folder / "additional_features.provenance.json"
        ),
        "query_ids": manifest["split_ids"][SPLIT],
        "implementation": "FP64 Euclidean Gram accumulation",
    }
    return SimpleNamespace(
        output=output,
        space=space,
        cfg=cfg,
        bank=bank,
        bank_hash=bank_hash,
        groups=dict(groups, **{SPLIT: rows}),
        expected=expected,
        target=target,
        receipt=receipt_path,
    )


def _validate_control_distance_cache(inputs):
    if not inputs.target.exists() or not inputs.receipt.exists():
        return False
    if read_json(inputs.receipt) != dict(
        inputs.expected, distances_sha256=calculate_file_sha256(inputs.target)
    ):
        raise ValueError("Additional distance cache changed")
    return True


def _compute_control_distance_tables(inputs, device):
    def load_signature_vector(split, row):
        return load_weighted_signature_vector(
            inputs.cfg, split, row, inputs.bank, inputs.bank_hash
        )

    def report_progress(split, completed, total):
        write_json(
            inputs.output / "analysis_status.json",
            {
                "status": "running",
                "phase": "additional_distances",
                "space": inputs.space,
                "references": reference_split,
                "completed": completed,
                "total": total,
            },
        )

    tables = {}
    for label in ("safe", "unsafe"):
        reference_split = f"{label}_reference"
        blocks = calculate_reference_distance_blocks(
            inputs.groups,
            reference_split,
            (SPLIT,),
            load_signature_vector,
            device=device,
            progress=report_progress,
        )
        tables[label] = blocks[SPLIT]
    return tables


def _persist_control_distance_tables(inputs, tables):
    np.savez_compressed(inputs.target, **tables)
    write_json(
        inputs.receipt,
        dict(inputs.expected, distances_sha256=calculate_file_sha256(inputs.target)),
    )


def measure_control_distances(output, space, device="cuda"):
    frozen_inputs = read_frozen_control_inputs(output)
    inputs = _load_control_distance_inputs(output, space, frozen_inputs)
    if _validate_control_distance_cache(inputs):
        return
    tables = _compute_control_distance_tables(inputs, device)
    _persist_control_distance_tables(inputs, tables)


def run_detection_measurement(
    output: Path,
    *,
    mode: str = "paired",
    source: Path | None = None,
    space: str = "trajectory",
    device: str = "cuda",
    options: dict | None = None,
) -> Path:
    """Measure paired, rescored-probe, reused-reference or control distances."""
    if mode not in {"paired", "probes", "reused", "controls"}:
        raise ValueError("Unknown detection measurement mode")
    output = Path(output).resolve()
    options = dict(options or {})
    allowed = {
        "paired": {"include_training"},
        "probes": {"probe_aggregation", "working_directory", "wait_for_source"},
        "reused": {"batch_size"},
        "controls": set(),
    }
    if set(options) - allowed[mode]:
        raise ValueError("Unknown measurement options")
    if space not in VARIANTS:
        raise ValueError("Unknown signature space")
    if mode == "paired":
        output, config, bank, bank_hash, groups = load_paired_distance_inputs(
            output, space
        )
        tables = measure_paired_reference_distances(
            config,
            bank,
            bank_hash,
            groups,
            device,
            output,
            options.get("include_training", False),
        )
        persist_paired_distances(output, tables, bank_hash, space)
    elif mode == "probes":
        if source is None:
            raise ValueError("Probe remeasurement requires a frozen source")
        aggregation = options.get("probe_aggregation", "rms")
        await_frozen_signature_source(
            source, output, space, options.get("wait_for_source", False)
        )
        inputs = load_probe_measurement_inputs(source, output, space, aggregation)
        resumed = _read_measurement_checkpoint(inputs)
        _measure_probe_tables(
            inputs, options.get("working_directory", "/var/tmp"), device, resumed
        )
    elif mode == "controls":
        for current_space in VARIANTS:
            measure_control_distances(output, current_space, device)
    else:
        for current_space in VARIANTS:
            measure_signature_space_distances(
                output, current_space, device, options.get("batch_size", 16)
            )
        write_json(
            output / "analysis_status.json",
            {"status": "complete", "phase": "distances_complete"},
        )
    return output
