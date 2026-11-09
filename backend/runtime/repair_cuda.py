"""Unsafe-only repair with benchmarked GPU inference and cached CUDA geometry."""

import gc
import json
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from backend.runtime.repair_cpu import recover_native_baseline_fields
from backend.runtime.repair_fields import (
    GPUReferenceStore,
    GPURepairAdapter,
    benchmark_probe_batches,
)
from backend.src.common.storage import (
    calculate_file_sha256,
    read_json,
    read_jsonl,
    write_json,
)
from backend.src.fields.resident import extract_resident_field
from backend.src.fields.signatures import load_signature_space_config
from backend.src.inference.encoding import compare_signature_values
from backend.src.repair.artifacts import compare_reconstruction, export_files, update
from backend.src.repair.contracts import RepairExperiment
from backend.src.repair.execute import run_repair_execution
from backend.src.repair.normalization import fit_safe_deviation_normalization
from backend.src.repair.report import run_repair_reporting
from backend.src.repair.score import run_repair_scoring


class GPUTelemetry:
    def __init__(self, output):
        self.output = output
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        with (self.output / "gpu_telemetry.jsonl").open("a") as stream:
            while not self.stop.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    phase = (
                        read_json(self.output / "status.json").get("phase")
                        if (self.output / "status.json").exists()
                        else "starting"
                    )
                    stream.write(
                        json.dumps(
                            {
                                "time": time.time(),
                                "phase": phase,
                                "gpu_util_memory_power": result.stdout.strip(),
                            }
                        )
                        + "\n"
                    )
                    stream.flush()
                except Exception:
                    pass
                self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=7)


def weighted_distance_tables(
    references, queries, weights, safe_count, device, report, tile_size=16384
):
    """FP64 GPU products with bounded coordinate tiles and FP32 native weighting."""
    shape = references.shape
    refs = references.reshape(len(references), -1)
    query = queries.to(device).reshape(len(queries), -1)
    scale = weights.to(device).float().sqrt().repeat_interleave(int(np.prod(shape[2:])))
    qref = torch.zeros((len(query), len(refs)), device=device, dtype=torch.float64)
    cross = torch.zeros(
        (safe_count, len(refs) - safe_count), device=device, dtype=torch.float64
    )
    qnorm = torch.zeros(len(query), device=device, dtype=torch.float64)
    rnorm = torch.zeros(len(refs), device=device, dtype=torch.float64)
    for start in range(0, refs.shape[1], tile_size):
        end = min(start + tile_size, refs.shape[1])
        block = (refs[:, start:end].to(device).float() * scale[start:end]).double()
        qblock = (query[:, start:end].float() * scale[start:end]).double()
        qref.addmm_(qblock, block.T)
        cross.addmm_(block[:safe_count], block[safe_count:].T)
        qnorm += qblock.square().sum(1)
        rnorm += block.square().sum(1)
        if start // tile_size % 16 == 0:
            report(completed_coordinates=end, total_coordinates=refs.shape[1])
    distances = (qnorm[:, None] + rnorm[None, :] - 2 * qref).clamp_min_(0).sqrt_()
    cross = (
        (rnorm[:safe_count, None] + rnorm[None, safe_count:] - 2 * cross)
        .clamp_min_(0)
        .sqrt_()
    )
    return distances.cpu().numpy(), cross.cpu().numpy()


def check_distances(output, store, groups, bank, distances, selected, settings):
    queries = [
        ("safe_validation", i, row)
        for i, row in enumerate(
            groups["safe_validation"][: settings["experiment"]["normalization_limit"]]
        )
    ]
    queries += [(row["split"], row["source_index"], row) for row in selected]
    values = torch.stack(
        [store.load_signature(split, row) for split, _, row in queries]
    )
    measured, cross = weighted_distance_tables(
        store.values,
        values,
        bank["weights"],
        len(groups["safe_reference"]),
        "cuda",
        lambda **details: update(output, "verify_reconstructed_distances", **details),
    )
    del values
    checks = {}
    safe_count = len(groups["safe_reference"])
    for label, matrix in (
        ("safe", measured[:, :safe_count]),
        ("unsafe", measured[:, safe_count:]),
    ):
        expected = np.stack(
            [distances[f"{label}__{split}"][index] for split, index, _ in queries]
        )
        checks[label] = compare_reconstruction(
            matrix,
            expected,
            settings["execution"]["reconstruction_rtol"],
            settings["execution"]["reconstruction_atol"],
        )
    passed = all(record["passed"] for record in checks.values())
    write_json(
        output / "reconstruction.json",
        {
            "passed": passed,
            "checks": checks,
            "query_count": len(queries),
            "thresholds_refitted": False,
            "reference_cache": store.location,
        },
    )
    if not passed:
        raise ValueError(
            "Reconstructed GPU distances differ from the retained detector; images not generated"
        )
    distances["unsafe__safe_reference"] = cross
    np.savez_compressed(
        output / "reference_cross_distances.npz", unsafe__safe_reference=cross
    )


def generate(output):
    manifest = read_json(output / "experiment.json")
    settings = manifest["fingerprint"]["settings"]
    selected = manifest["selected_cohort"]
    if any(
        row["split"] != "unsafe_test" or row["label"] != "unsafe" for row in selected
    ):
        raise ValueError("Only unsafe test prompts may enter this repair run")
    source = output / "source"
    for name, digest in manifest["fingerprint"]["inputs"].items():
        if calculate_file_sha256(source / name) != digest:
            raise ValueError(f"Frozen detector input changed: {name}")
    for name, digest in manifest["fingerprint"]["implementation"].items():
        if calculate_file_sha256(Path.cwd() / name) != digest:
            raise ValueError(f"Frozen GPU implementation changed: {name}")
    torch.set_num_threads(settings["experiment"]["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = read_json(source / "experiment.json")
    groups = {
        split: read_jsonl(source / "trajectory/splits" / f"{split}.jsonl")
        for split in parent["split_ids"]
    }
    bank_path = source / "trajectory/probes/probe_bank.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    bank_hash = calculate_file_sha256(bank_path)
    if bank_hash != manifest["reuse"]["bank_sha256"]:
        raise ValueError("Recovered probe bank changed")
    recover_native_baseline_fields(source, bank, bank_hash, groups)
    needed = {split: groups[split] for split in ("safe_reference", "unsafe_reference")}
    needed["safe_validation"] = groups["safe_validation"][
        : settings["experiment"]["normalization_limit"]
    ]
    needed["unsafe_test"] = selected
    update(output, "loading_resident_gpu_model")
    adapter = GPURepairAdapter(source, bank, needed)
    config = load_signature_space_config(source, "trajectory")
    config["model"].update(device="cuda", cpu_offload=False)
    config["experiment"]["probe_batch_size"] = 8
    update(output, "benchmark_probe_batches")
    evaluator = benchmark_probe_batches(
        output,
        source,
        adapter,
        bank,
        bank_hash,
        [groups["safe_reference"][0], groups["unsafe_reference"][0]],
        config,
        settings["execution"],
    )
    adapter.resident = evaluator
    baseline = torch.load(
        source / "optimization/baseline.pt", map_location="cpu", weights_only=True
    )
    original_checks = []
    for record in baseline["records"]:
        expected = record["fields"]["trajectory"].cpu()
        actual = evaluator.evaluate_field(
            adapter.encode_prompt(record["row"]["prompt"])
        ).cpu()
        original_checks.append(
            {
                "id": record["row"]["id"],
                **compare_signature_values(expected, actual),
                "passed": bool(torch.allclose(expected, actual, rtol=1e-3, atol=1e-3)),
            }
        )
    write_json(
        output / "original_gpu_field_compatibility.json",
        {
            "checks": original_checks,
            "passed": all(record["passed"] for record in original_checks),
        },
    )
    if not all(record["passed"] for record in original_checks):
        raise ValueError("Fresh GPU fields differ from the original detector baseline")
    adapter.frozen_probe_forward_calls = int(
        np.ceil(len(bank["states"]) / evaluator.batch_size)
    )
    adapter.check_packed_sampling(
        output,
        bank,
        [
            groups["safe_reference"][0]["prompt"],
            groups["unsafe_reference"][0]["prompt"],
        ],
    )
    total, completed, reused = sum(map(len, needed.values())), 0, 0
    futures = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as exporter:
        for split, rows in needed.items():
            for row in rows:
                path = source / "trajectory/features" / split / f"{row['id']}.pt"
                existed = path.exists() and path.with_suffix(".sha256.json").exists()
                field = extract_resident_field(
                    source, "trajectory", split, row, evaluator, adapter.encoder
                )
                del field
                reused += int(existed)
                if not existed and settings["execution"].get(
                    "persist_reference_checkpoints"
                ):
                    futures.append(
                        exporter.submit(
                            export_files,
                            output,
                            f"field_{split}_{row['id']}",
                            [path, path.with_suffix(".sha256.json")],
                        )
                    )
                completed += 1
                update(
                    output,
                    "extract_fields_gpu",
                    completed=completed,
                    total=total,
                    reused_fields=reused,
                    gpu_batch_size=evaluator.batch_size,
                    elapsed_seconds=time.perf_counter() - started,
                )
        extraction_seconds = time.perf_counter() - started
        update(output, "cache_training_references")
        store = GPUReferenceStore(
            source,
            bank,
            bank_hash,
            adapter.signature,
            groups,
            settings["execution"]["reference_workspace_gib"],
            settings["execution"]["geometry_batch_size"],
        )
        with np.load(source / "trajectory/distances.npz") as archive:
            distances = {name: archive[name].copy() for name in archive.files}
        check_distances(output, store, groups, bank, distances, selected, settings)
        groups["additional_safe_test"] = []
        experiment = RepairExperiment(
            settings,
            output,
            source,
            config,
            bank,
            manifest["calibration"],
            groups,
            distances,
            store,
            manifest["fingerprint"],
        )
        experiment.normalization = fit_safe_deviation_normalization(experiment)
        write_json(output / "normalization.json", experiment.normalization)
        write_json(
            output / "gpu_execution.json",
            {
                "gpu": torch.cuda.get_device_name(),
                "batch_size": evaluator.batch_size,
                "reference_cache": store.location,
                "reused_fields": reused,
                "extracted_fields": total - reused,
                "extraction_seconds": extraction_seconds,
                "geometry_device": "cuda",
                "safe_test_prompts_generated": 0,
                "model_dtype": "float16",
                "reference_arithmetic": "float64",
                "sampling": "native current-latent null/original/approved evaluations batched together; no cached velocity substitution",
            },
        )
        adapter.load_decoder()
        experiment.adapter = adapter

        def completed_case(case, _records, completed, total):
            row = case[2]
            receipt = output / "pilot/records" / f"{row['split']}_{row['id']}.json"
            export_files(
                output,
                f"image_{row['id']}",
                [receipt, *[output / name for name in read_json(receipt)["artifacts"]]],
            )
            update(
                output,
                "image_generation",
                completed=completed,
                total=total,
                prompt_id=row["id"],
            )

        cases = [(row["split"], row["source_index"], row) for row in selected]
        records = run_repair_execution(
            experiment, "pilot", cases=cases, on_case_completed=completed_case
        )
        if len(records) != manifest["expected_generation_records"]:
            raise ValueError("Missing repair comparison records")
        run_repair_reporting(experiment, "pilot", records)
        for future in futures:
            future.result()


def run_worker(output):
    try:
        with GPUTelemetry(output):
            generate(output)
            gc.collect()
            torch.cuda.empty_cache()
            settings = read_json(output / "experiment.json")["fingerprint"]["settings"][
                "image_scoring"
            ]
            update(output, "independent_image_scoring")
            run_repair_scoring(
                [
                    "--output",
                    str(output),
                    "--cohort",
                    "pilot",
                    "--device",
                    "cuda",
                    "--model-id",
                    settings["model_id"],
                    "--revision",
                    settings["revision"],
                    "--threshold",
                    str(settings["threshold"]),
                ]
            )
        paths = [
            path
            for path in output.rglob("*")
            if path.is_file()
            and path.parts[len(output.parts)] not in {"source", "exports"}
            and "generations" not in path.parts
            and path.name != "gpu_worker.log"
        ]
        export_files(output, "final", paths)
        write_json(output / "gpu_worker_status.json", {"status": "complete"})
    except BaseException:
        write_json(
            output / "gpu_worker_status.json",
            {"status": "failed", "error": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run_worker(parser.parse_args().output.resolve())
