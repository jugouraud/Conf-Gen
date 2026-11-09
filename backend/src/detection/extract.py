"""H1: extract frozen fields with native, staged, resident or cached execution."""

import gc
import os
import time
import traceback
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from backend.src.common.storage import (
    calculate_file_sha256,
    read_json,
    read_jsonl,
    write_json,
)
from backend.src.data.definitions import UNSAFE_SPLITS
from backend.src.detection.contracts import QUERY_SPLITS, read_resident_detection_inputs
from backend.src.fields.definitions import VARIANTS
from backend.src.fields.distances import (
    ResidentFieldDistances,
    calculate_resident_training_distances,
)
from backend.src.fields.resident import (
    create_resident_probe_banks,
    encode_resident_prompts,
    extract_resident_field,
    load_native_baseline,
    read_resident_field,
    record_resident_status,
    verify_resident_compatibility,
)
from backend.src.fields.signatures import (
    SignatureExtractionTracker,
    build_probe_bank,
    evaluate_probe_field,
    load_signature_space_config,
    read_probe_bank,
    read_probe_fit_rows,
    resolve_probe_bank_path,
    resolve_signature_path,
    sample_safe_reference_trajectories,
    save_field_signature,
    validate_cached_signature,
)
from backend.src.inference.encoding import (
    ResidentProbeEvaluator,
    StagedPromptEncoder,
    compare_signature_values,
    save_atomic_tensor,
    synchronize_model_device,
)
from backend.src.inference.flow import FlowModelAdapter


def _load_signature_extraction_inputs(output, adapter, splits, precomputed):
    checkpoint = SignatureExtractionTracker(output)
    manifest = read_json(output / "experiment.json")
    checkpoint.state["total_features"] = len(VARIANTS) * sum(
        manifest["counts"][s] for s in splits
    )
    checkpoint.write_status()
    configs = {s: load_signature_space_config(output, s) for s in VARIANTS}
    if adapter is None:
        adapter = FlowModelAdapter(configs["trajectory"])
    banks = {s: read_probe_bank(configs[s], adapter) for s in VARIANTS}
    hashes = {
        s: calculate_file_sha256(resolve_probe_bank_path(configs[s])) for s in VARIANTS
    }
    cached = {} if precomputed is None else precomputed["features"]
    if precomputed is not None and precomputed[
        "experiment_sha256"
    ] != calculate_file_sha256(output / "experiment.json"):
        raise ValueError("Restored features belong to a different experiment")
    return SimpleNamespace(
        output=output,
        checkpoint=checkpoint,
        configs=configs,
        adapter=adapter,
        banks=banks,
        hashes=hashes,
        cached=cached,
    )


def _read_extraction_split(inputs, split):
    inputs.checkpoint.write_status(phase=f"extract/{split}")
    return read_jsonl(inputs.output / "trajectory/splits" / f"{split}.jsonl")


def _extract_prompt_signatures(inputs, split, row):
    checkpoint, configs, adapter = inputs.checkpoint, inputs.configs, inputs.adapter
    banks, hashes, cached = inputs.banks, inputs.hashes, inputs.cached
    relative_paths = {s: f"{s}/features/{split}/{row['id']}.pt" for s in VARIANTS}
    missing = [
        s
        for s in VARIANTS
        if not resolve_signature_path(configs[s], split, row["id"]).exists()
        and f"{s}/features/{split}/{row['id']}.pt" not in cached
    ]
    embeddings = adapter.encode_prompt(row["prompt"]) if missing else None
    for space in VARIANTS:
        path = resolve_signature_path(configs[space], split, row["id"])
        if not path.exists() and relative_paths[space] in cached:
            # Verified local checkpoints are transferred separately;
            # measurement checks their bytes before using any vector.
            checkpoint.hashes[space][f"{split}/{row['id']}.pt"] = cached[
                relative_paths[space]
            ]
            checkpoint.state["completed_features"] += 1
            key = f"{space}/{split}"
            checkpoint.state["counts"][key] = checkpoint.state["counts"].get(key, 0) + 1
            continue
        if path.exists():
            validate_cached_signature(path, hashes[space], adapter.signature)
        else:
            field = evaluate_probe_field(
                configs[space], adapter, banks[space], embeddings
            )
            if not torch.isfinite(field).all():
                raise ValueError("Non-finite unsafe signature")
            save_field_signature(
                path, row["id"], field, hashes[space], adapter.signature
            )
        checkpoint.register_artifact(path, space, split, row["id"])
    checkpoint.write_status()


def _write_extraction_receipts(inputs, provenance_name):
    output, checkpoint, hashes, adapter = (
        inputs.output,
        inputs.checkpoint,
        inputs.hashes,
        inputs.adapter,
    )
    for space in VARIANTS:
        path = output / space / provenance_name
        write_json(
            path,
            {
                "bank_sha256": hashes[space],
                "features": checkpoint.hashes[space],
                "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
                "adapter_signature": adapter.signature,
            },
        )
        checkpoint.register_artifact(path)
    checkpoint.flush_transfer_bundle(force=True)
    checkpoint.write_status(status="complete", phase="extraction_complete")


def _record_extraction_failure(inputs):
    inputs.checkpoint.flush_transfer_bundle(force=True)
    inputs.checkpoint.write_status(status="failed", error=traceback.format_exc())


def _extract_cached_signatures(
    output,
    adapter=None,
    *,
    splits=UNSAFE_SPLITS,
    provenance_name="unsafe_features.provenance.json",
    precomputed=None,
):
    inputs = _load_signature_extraction_inputs(output, adapter, splits, precomputed)
    try:
        for split in splits:
            rows = _read_extraction_split(inputs, split)
            for row in rows:
                _extract_prompt_signatures(inputs, split, row)
            inputs.checkpoint.flush_transfer_bundle(force=True)
        _write_extraction_receipts(inputs, provenance_name)
    except BaseException:
        _record_extraction_failure(inputs)
        raise


def initialize_paired_signature_extraction(output, adapter):
    output = Path(output).resolve()
    manifest = read_json(output / "experiment.json")
    tracker = SignatureExtractionTracker(output)
    tracker.state["total_features"] = sum(manifest["counts"].values())
    tracker.write_status(phase="loading_model")
    cfg = load_signature_space_config(output, "trajectory")
    if adapter is None:
        adapter = FlowModelAdapter(cfg)
    groups = {}
    for split in manifest["counts"]:
        path = output / "trajectory/splits" / f"{split}.jsonl"
        if calculate_file_sha256(path) != manifest["split_sha256"][split]:
            raise ValueError("Frozen dataset split changed")
        groups[split] = read_jsonl(path)
    return output, cfg, adapter, groups, tracker


def freeze_training_trajectory_bank(cfg, adapter, groups, tracker):
    path = resolve_probe_bank_path(cfg)
    if not path.exists():
        tracker.write_status(phase="building_train_only_trajectory_bank")
        _, fit = read_probe_fit_rows(cfg)
        states, provenance, midpoints = sample_safe_reference_trajectories(
            cfg, adapter, fit
        )
        bank = build_probe_bank(cfg, adapter, fit, states, provenance, midpoints)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(bank, temporary)
        os.replace(temporary, path)
    bank = read_probe_bank(cfg, adapter)
    tracker.register_artifact(path)
    tracker.flush_transfer_bundle(force=True)
    return bank, calculate_file_sha256(path)


def _extract_paired_prompt(cfg, adapter, bank, bank_hash, split, row, tracker):
    path = resolve_signature_path(cfg, split, row["id"])
    if path.exists():
        validate_cached_signature(path, bank_hash, adapter.signature)
    else:
        embeddings = adapter.encode_prompt(row["prompt"])
        field = evaluate_probe_field(cfg, adapter, bank, embeddings)
        if not torch.isfinite(field).all():
            raise ValueError("Nonfinite paired signature")
        save_field_signature(path, row["id"], field, bank_hash, adapter.signature)
    tracker.register_artifact(path, "trajectory", split, row["id"])
    tracker.write_status(phase=f"extract/{split}", current_id=row["id"])


def persist_paired_extraction_receipt(output, adapter, bank_hash, tracker):
    path = output / "trajectory/features.provenance.json"
    write_json(
        path,
        {
            "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
            "bank_sha256": bank_hash,
            "adapter_signature": adapter.signature,
            "features": tracker.hashes["trajectory"],
            "model_metadata": adapter.metadata,
        },
    )
    tracker.register_artifact(path)
    tracker.flush_transfer_bundle(force=True)
    tracker.write_status(status="complete", phase="extraction_complete")


def extract_all_paired_signatures(cfg, adapter, groups, tracker, bank, bank_hash):
    for split, rows in groups.items():
        for row in rows:
            _extract_paired_prompt(cfg, adapter, bank, bank_hash, split, row, tracker)


def _extract_native_signatures(output, adapter=None):
    output, cfg, adapter, groups, tracker = initialize_paired_signature_extraction(
        output, adapter
    )
    bank, bank_hash = freeze_training_trajectory_bank(cfg, adapter, groups, tracker)
    extract_all_paired_signatures(cfg, adapter, groups, tracker, bank, bank_hash)
    persist_paired_extraction_receipt(output, adapter, bank_hash, tracker)


def initialize_optimized_extraction(output, adapter, batch_size, space="trajectory"):
    output = Path(output).resolve()
    manifest = read_json(output / "experiment.json")
    tracker = SignatureExtractionTracker(output, resume=True)
    tracker.state["total_features"] = sum(manifest["counts"].values())
    tracker.write_status(phase="optimization/loading_model")
    cfg = load_signature_space_config(output, space)
    if adapter is None:
        adapter = FlowModelAdapter(cfg)
    groups = {}
    for split in manifest["counts"]:
        path = output / space / "splits" / f"{split}.jsonl"
        if calculate_file_sha256(path) != manifest["split_sha256"][split]:
            raise ValueError("Frozen dataset split changed")
        groups[split] = read_jsonl(path)
    bank = read_probe_bank(cfg, adapter)
    bank_hash = calculate_file_sha256(resolve_probe_bank_path(cfg))
    pending = []
    reused = 0
    for split, rows in groups.items():
        for row in rows:
            path = resolve_signature_path(cfg, split, row["id"])
            if path.exists():
                validate_cached_signature(path, bank_hash, adapter.signature)
                tracker.register_artifact(path, space, split, row["id"])
                reused += 1
            else:
                pending.append((split, row))
    tracker.flush_transfer_bundle(force=True)
    tracker.write_status(phase="optimization/compatibility", reused_features=reused)
    return SimpleNamespace(
        output=output,
        space=space,
        cfg=cfg,
        adapter=adapter,
        groups=groups,
        bank=bank,
        bank_hash=bank_hash,
        tracker=tracker,
        pending=pending,
        encoder=StagedPromptEncoder(adapter, output / "optimization/embeddings"),
        batch_size=batch_size,
    )


def _record_baseline_compatibility(inputs):
    path = inputs.output / "optimization/baseline.pt"
    if path.exists():
        baseline = torch.load(path, map_location="cpu", weights_only=True)
        if (
            baseline["bank_sha256"] != inputs.bank_hash
            or baseline["adapter_signature"] != inputs.adapter.signature
        ):
            raise ValueError("Stale compatibility baseline")
        return baseline
    # Only training prompts are used to validate execution equivalence.
    rows = [inputs.groups["safe_reference"][0], inputs.groups["unsafe_reference"][0]]
    records = []
    for row in rows:
        started = synchronize_model_device(inputs.adapter)
        embeddings = inputs.adapter.encode_prompt(row["prompt"])
        field = evaluate_probe_field(
            inputs.cfg, inputs.adapter, inputs.bank, embeddings
        )
        seconds = synchronize_model_device(inputs.adapter) - started
        records.append(
            {
                "row": row,
                "embeddings": [value.cpu() for value in embeddings],
                "field": field,
                "baseline_seconds": seconds,
            }
        )
    baseline = {
        "bank_sha256": inputs.bank_hash,
        "adapter_signature": inputs.adapter.signature,
        "records": records,
    }
    save_atomic_tensor(path, baseline)
    return baseline


def _encode_component_pass(inputs, component, rows):
    inputs.encoder.prepare_component(component)
    started = synchronize_model_device(inputs.adapter)
    for index, row in enumerate(rows, 1):
        inputs.encoder.encode_component(component, row)
        if index % 10 == 0 or index == len(rows):
            inputs.tracker.write_status(
                phase=f"encode/{component}",
                encoded_prompts=index,
                encoding_total=len(rows),
            )
    return synchronize_model_device(inputs.adapter) - started


def _create_resident_evaluator(inputs):
    inputs.encoder.prepare_transformer()
    evaluator = ResidentProbeEvaluator(
        inputs.adapter,
        inputs.bank,
        inputs.bank_hash,
        inputs.output / "optimization/null_field.pt",
        inputs.batch_size,
        pad_final_batch=inputs.space == "source_slice",
    )
    evaluator.prepare_null_cache()
    return evaluator


def _verify_resident_compatibility(inputs, baseline, evaluator, encoding_seconds):
    records = []
    for record in baseline["records"]:
        embeddings = inputs.encoder.read_embeddings(record["row"])
        embeddings_equal = all(
            torch.equal(a, b)
            for a, b in zip(embeddings, record["embeddings"], strict=True)
        )
        started = synchronize_model_device(inputs.adapter)
        field = evaluator.evaluate_field(embeddings)
        seconds = synchronize_model_device(inputs.adapter) - started
        records.append(
            {
                "id": record["row"]["id"],
                "embedding_bitwise_equal": embeddings_equal,
                **compare_signature_values(record["field"], field),
                "baseline_seconds": record["baseline_seconds"],
                "optimized_field_seconds": seconds,
            }
        )
        previous = resolve_signature_path(
            inputs.cfg, f"{record['row']['label']}_reference", record["row"]["id"]
        )
        if previous.exists():
            saved = torch.load(previous, map_location="cpu", weights_only=True)[
                "values"
            ]
            records[-1]["saved_checkpoint_bitwise_equal"] = bool(
                torch.equal(saved, field)
            )
    passed = all(
        row["bitwise_equal"]
        and row["embedding_bitwise_equal"]
        and row.get("saved_checkpoint_bitwise_equal", True)
        for row in records
    )
    report = {
        "passed": passed,
        "acceptance": "bitwise equal native embeddings and full FP32 signature tensors",
        "bank_sha256": inputs.bank_hash,
        "adapter_signature": inputs.adapter.signature,
        "batch_size": inputs.batch_size,
        "records": records,
        "encoding_seconds": encoding_seconds,
    }
    write_json(inputs.output / "optimization/compatibility.json", report)
    if not passed:
        raise ValueError(
            "Optimized signatures failed bitwise compatibility; saved checkpoints are untouched"
        )
    return report


def _extract_resident_prompt(inputs, evaluator, split, row):
    started = synchronize_model_device(inputs.adapter)
    field = evaluator.evaluate_field(inputs.encoder.read_embeddings(row))
    if not torch.isfinite(field).all():
        raise ValueError("Nonfinite optimized signature")
    path = resolve_signature_path(inputs.cfg, split, row["id"])
    save_field_signature(
        path, row["id"], field, inputs.bank_hash, inputs.adapter.signature
    )
    inputs.tracker.register_artifact(path, inputs.space, split, row["id"])
    inputs.tracker.write_status(
        phase=f"optimized_extract/{split}",
        current_id=row["id"],
        last_prompt_seconds=synchronize_model_device(inputs.adapter) - started,
    )


def complete_optimized_extraction(inputs, compatibility):
    path = inputs.output / inputs.space / "features.provenance.json"
    write_json(
        path,
        {
            "experiment_sha256": calculate_file_sha256(
                inputs.output / "experiment.json"
            ),
            "bank_sha256": inputs.bank_hash,
            "adapter_signature": inputs.adapter.signature,
            "features": inputs.tracker.hashes[inputs.space],
            "model_metadata": inputs.adapter.metadata,
            "execution": "staged_native_encoders_resident_positive_field_v1",
            "compatibility_sha256": calculate_file_sha256(
                inputs.output / "optimization/compatibility.json"
            ),
        },
    )
    inputs.tracker.register_artifact(path)
    inputs.tracker.register_artifact(inputs.output / "optimization/compatibility.json")
    inputs.tracker.flush_transfer_bundle(force=True)
    inputs.tracker.write_status(status="complete", phase="extraction_complete")


def _read_accepted_compatibility(inputs):
    existing = inputs.output / "optimization/compatibility.json"
    if not existing.exists():
        return None
    report = read_json(existing)
    if not (
        report["passed"]
        and report["bank_sha256"] == inputs.bank_hash
        and report["adapter_signature"] == inputs.adapter.signature
        and report["batch_size"] == inputs.batch_size
    ):
        return None
    baseline = torch.load(
        inputs.output / "optimization/baseline.pt",
        map_location="cpu",
        weights_only=True,
    )
    for reference, record in zip(baseline["records"], report["records"], strict=True):
        row = reference["row"]
        previous = resolve_signature_path(
            inputs.cfg, f"{row['label']}_reference", row["id"]
        )
        if previous.exists():
            values = torch.load(previous, map_location="cpu", weights_only=True)[
                "values"
            ]
            if not torch.equal(values, reference["field"]):
                raise ValueError(
                    "Saved checkpoint differs from the verified native baseline"
                )
            record["saved_checkpoint_bitwise_equal"] = True
    write_json(existing, report)
    return report


def benchmark_optimized_extraction(inputs):
    accepted = _read_accepted_compatibility(inputs)
    if accepted is not None:
        return accepted
    baseline = _record_baseline_compatibility(inputs)
    encoding_seconds = {}
    for component in inputs.encoder.components:
        encoding_seconds[component] = _encode_component_pass(
            inputs, component, [r["row"] for r in baseline["records"]]
        )
    evaluator = _create_resident_evaluator(inputs)
    return _verify_resident_compatibility(inputs, baseline, evaluator, encoding_seconds)


def encode_remaining_prompt_components(inputs):
    for component in inputs.encoder.components:
        _encode_component_pass(inputs, component, [row for _, row in inputs.pending])


def extract_remaining_resident_signatures(inputs):
    evaluator = _create_resident_evaluator(inputs)
    for split, row in inputs.pending:
        _extract_resident_prompt(inputs, evaluator, split, row)


def extract_pending_prompt_chunks(inputs, chunk_size):
    if chunk_size < 1:
        raise ValueError("prompt_chunk_size must be positive")
    for start in range(0, len(inputs.pending), chunk_size):
        chunk = SimpleNamespace(**vars(inputs))
        chunk.pending = inputs.pending[start : start + chunk_size]
        for component in chunk.encoder.components:
            _encode_component_pass(chunk, component, [row for _, row in chunk.pending])
        evaluator = _create_resident_evaluator(chunk)
        for split, row in chunk.pending:
            _extract_resident_prompt(chunk, evaluator, split, row)
        del evaluator


def _extract_staged_signatures(
    output,
    adapter=None,
    *,
    benchmark_only=False,
    batch_size=16,
    space="trajectory",
    prompt_chunk_size=None,
):
    inputs = initialize_optimized_extraction(output, adapter, batch_size, space)
    compatibility = benchmark_optimized_extraction(inputs)
    if benchmark_only:
        return compatibility
    if prompt_chunk_size is None:
        encode_remaining_prompt_components(inputs)
        extract_remaining_resident_signatures(inputs)
    else:
        extract_pending_prompt_chunks(inputs, prompt_chunk_size)
    complete_optimized_extraction(inputs, compatibility)
    return compatibility


def _measure_query_distances(
    output, space, groups, bank, load_field, execution, device
):
    query_rows = [(s, r) for s in QUERY_SPLITS for r in groups[s]]
    cache_bytes = len(query_rows) * bank["states"].numel() * 4
    free, _ = torch.cuda.mem_get_info(device)
    required = cache_bytes + int(execution["workspace_gib"] * 1024**3)
    if free < required:
        raise RuntimeError(
            f"Resident query cache plus workspace needs {required / 1024**3:.2f} GiB free; "
            f"available {free / 1024**3:.2f} GiB. Use a larger GPU or a fresh smaller-batch config."
        )
    cache = ResidentFieldDistances(
        len(query_rows),
        bank["states"].shape,
        bank["weights"],
        device=device,
        query_batch_size=execution["query_batch_size"],
    )
    for index, (split, row) in enumerate(query_rows, 1):
        cache.append(load_field(split, row))
        record_resident_status(
            output, f"{space}/queries", completed=index, total=len(query_rows)
        )
    tables = {}
    for label in ("safe", "unsafe"):
        rows = groups[f"{label}_reference"]
        distances = torch.empty(
            (len(query_rows), len(rows)),
            device=device,
            dtype=torch.float64,
        )
        batch_size = execution["reference_batch_size"]
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            fields = [load_field(f"{label}_reference", row) for row in batch]
            distances[:, start : start + len(batch)] = cache.measure(fields)
            del fields
            record_resident_status(
                output,
                f"{space}/{label}_references",
                completed=start + len(batch),
                total=len(rows),
            )
        values, offset = distances.cpu().numpy(), 0
        for split in QUERY_SPLITS:
            tables[f"{label}__{split}"] = values[
                offset : offset + len(groups[split])
            ].copy()
            offset += len(groups[split])
        del distances
    del cache
    torch.cuda.empty_cache()
    return tables


def _extract_checkpointed_fields(output, space, groups, evaluator, encoder):
    for split, rows in groups.items():
        for index, row in enumerate(rows, 1):
            field = extract_resident_field(
                output, space, split, row, evaluator, encoder
            )
            del field
            record_resident_status(
                output, f"{space}/extract/{split}", completed=index, total=len(rows)
            )


def _training_geometry(
    output, space, groups, bank, bank_hash, adapter_signature, execution, device
):
    matrices = {}
    for label in ("safe", "unsafe"):
        rows = groups[f"{label}_reference"]
        cache = ResidentFieldDistances(
            len(rows),
            bank["states"].shape,
            bank["weights"],
            device=device,
            query_batch_size=execution["query_batch_size"],
        )
        for index, row in enumerate(rows, 1):
            path = (
                output / space / "features" / f"{label}_reference" / f"{row['id']}.pt"
            )
            cache.append(
                read_resident_field(
                    path, row, bank_hash, adapter_signature, bank["states"].shape
                )
            )
            if index % 25 == 0:
                record_resident_status(
                    output,
                    f"{space}/training_geometry/{label}",
                    completed=index,
                    total=len(rows),
                )
        matrices[label] = (
            calculate_resident_training_distances(
                cache.values.flatten(1),
                feature_batch_size=execution["feature_batch_size"],
            )
            .cpu()
            .numpy()
        )
        del cache
        torch.cuda.empty_cache()
    return matrices


def initialize_resident_paired_evaluation(output):
    output = Path(output).resolve()
    manifest, groups = read_resident_detection_inputs(output)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Resident paired evaluation requires CUDA; preparation is CPU-only"
        )
    torch.set_num_threads(manifest["config"]["experiment"].get("threads", 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    record_resident_status(output, "loading_model")
    adapter = FlowModelAdapter(load_signature_space_config(output, "trajectory"))
    return output, manifest, groups, adapter


def extract_resident_distances(output, manifest, groups, adapter):
    started = time.perf_counter()
    banks, hashes = create_resident_probe_banks(output, adapter)
    baseline = load_native_baseline(output, adapter, groups, banks, hashes)
    encoder = StagedPromptEncoder(adapter, output / "optimization/embeddings")
    encode_resident_prompts(output, encoder, groups)
    execution = manifest["config"]["execution"]
    all_tables = {}
    for space, bank in banks.items():
        evaluator = ResidentProbeEvaluator(
            adapter,
            bank,
            hashes[space],
            output / "optimization" / f"{space}.null.pt",
            execution["probe_batch_size"],
            pad_final_batch=space == "source_slice",
        )
        evaluator.prepare_null_cache()
        verify_resident_compatibility(output, space, evaluator, encoder, baseline)
        if execution.get("mode", "fused") == "checkpointed":
            _extract_checkpointed_fields(output, space, groups, evaluator, encoder)
        else:
            all_tables[space] = _measure_query_distances(
                output,
                space,
                groups,
                bank,
                partial(
                    extract_resident_field,
                    output,
                    space,
                    evaluator=evaluator,
                    encoder=encoder,
                ),
                execution,
                adapter.device,
            )
        del evaluator
        torch.cuda.empty_cache()
        # Publish the existing checkpoint hashes so alternative probe reductions
        # reuse these host-local fields without extracting or transferring again.
        write_json(
            output / space / "features.provenance.json",
            {
                "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
                "bank_sha256": hashes[space],
                "adapter_signature": adapter.signature,
                "features": {
                    f"{split}/{row['id']}.pt": read_json(
                        output / space / "features" / split / f"{row['id']}.sha256.json"
                    )["sha256"]
                    for split, rows in groups.items()
                    for row in rows
                },
            },
        )
    # Free every model component before training geometry; only one reference
    # class and a bounded FP64 coordinate tile then occupy VRAM.
    adapter.pipe.to("cpu")
    adapter.null_embeddings = tuple(v.cpu() for v in adapter.null_embeddings)
    gc.collect()
    torch.cuda.empty_cache()
    for space, bank in banks.items():
        if execution.get("mode", "fused") == "checkpointed":
            all_tables[space] = _measure_query_distances(
                output,
                space,
                groups,
                bank,
                lambda split, row: read_resident_field(
                    output / space / "features" / split / f"{row['id']}.pt",
                    row,
                    hashes[space],
                    adapter.signature,
                    bank["states"].shape,
                ),
                execution,
                adapter.device,
            )
        geometry = _training_geometry(
            output,
            space,
            groups,
            bank,
            hashes[space],
            adapter.signature,
            execution,
            adapter.device,
        )
        all_tables[space].update(
            {
                f"{label}__{label}_reference": values
                for label, values in geometry.items()
            }
        )
        path = output / space / "distances.npz"
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **all_tables[space])
        temporary.replace(path)
        write_json(
            path.with_suffix(".provenance.json"),
            {
                "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
                "bank_sha256": hashes[space],
                "distances_sha256": calculate_file_sha256(path),
                "adapter_signature": adapter.signature,
                "implementation": "resident FP32 weighted fields; FP64 GPU products; RMS probe aggregation",
                "reference_ids": {
                    label: manifest["split_ids"][f"{label}_reference"]
                    for label in ("safe", "unsafe")
                },
            },
        )
    write_json(
        output / "optimization/execution.json",
        {
            "elapsed_seconds": time.perf_counter() - started,
            "device": str(adapter.device),
            "torch_version": str(torch.__version__),
            "gpu_name": torch.cuda.get_device_name()
            if str(adapter.device).startswith("cuda")
            else None,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3
            if str(adapter.device).startswith("cuda")
            else 0,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3
            if str(adapter.device).startswith("cuda")
            else 0,
            "execution": execution,
            "raw_field_network_transfers": 0,
            "checkpoint_policy": "one CPU checkpoint copy per field; retained on GPU host; no transfer bundles",
        },
    )


def _extract_resident_signatures(output: Path) -> Path:
    """Create verified frozen fields and FP64 query/reference RMS distance tables."""
    output, manifest, groups, adapter = initialize_resident_paired_evaluation(output)
    extract_resident_distances(output, manifest, groups, adapter)
    return output


def run_detection_extraction(
    output: Path, *, mode: str = "resident", adapter=None, options: dict | None = None
) -> Path:
    """Own extraction lifecycle and checkpoints; return the output directory.

    Native creates a training-only trajectory bank. Resident evaluates both
    spaces and fuses distance products while fields are on device. Staged uses
    encoder passes; cached extracts selected splits against existing banks.
    """
    allowed = {
        "native": set(),
        "resident": set(),
        "staged": {"batch_size", "space", "prompt_chunk_size", "benchmark_only"},
        "cached": {"splits", "provenance_name", "precomputed"},
    }
    if mode not in allowed:
        raise ValueError("Unknown detection extraction mode")
    options = dict(options or {})
    threads = options.pop("threads", 4)
    if type(threads) is not int or threads < 1 or set(options) - allowed[mode]:
        raise ValueError("Invalid extraction options")
    output = Path(output).resolve()
    torch.set_num_threads(threads)
    if mode == "native":
        _extract_native_signatures(output, adapter)
    elif mode == "staged":
        _extract_staged_signatures(output, adapter, **options)
    elif mode == "cached":
        _extract_cached_signatures(output, adapter, **options)
    elif adapter is None:
        _extract_resident_signatures(output)
    else:
        manifest, groups = read_resident_detection_inputs(output)
        extract_resident_distances(output, manifest, groups, adapter)
    return output
