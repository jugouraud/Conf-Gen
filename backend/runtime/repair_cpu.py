"""Resume a recovered paired-repair experiment on a memory-bounded local CPU."""

import argparse
import copy
import gc
import os
import time
import traceback
from datetime import datetime, timezone
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
from backend.src.fields.resident import RebuiltSignatureStore
from backend.src.fields.signatures import (
    calculate_model_signature,
    load_signature_space_config,
    save_field_signature,
)
from backend.src.inference.encoding import StagedPromptEncoder
from backend.src.inference.flow import FlowModelAdapter, decode_latents_to_image
from backend.src.inference.scorers import ShieldGemmaImageScorer
from backend.src.repair.artifacts import compare_reconstruction
from backend.src.repair.contracts import RepairExperiment
from backend.src.repair.execute import run_repair_execution
from backend.src.repair.normalization import fit_safe_deviation_normalization
from backend.src.repair.report import run_repair_reporting
from backend.src.repair.score import run_repair_scoring


class CPUProgress:
    def __init__(self, output, threads):
        self.output, self.threads = output, threads
        self.state = {}
        manifest_path = output / "experiment.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        self.expected = {
            key: manifest[key]
            for key in (
                "expected_pairs",
                "expected_prompts",
                "expected_generation_records",
            )
            if key in manifest
        }

    def __call__(self, phase=None, **details):
        if phase is not None:
            self.state = {"stage": phase}
        self.state.update(details)
        write_json(
            self.output / "progress.json",
            {
                "status": "running",
                "device": "cpu",
                "worker_pid": os.getpid(),
                "threads": self.threads,
                **self.expected,
                "error": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **self.state,
            },
        )


class CachedCPUAdapter(FlowModelAdapter):
    """Native SD3 velocity equations with cached embeddings and no text encoders."""

    def __init__(self, output, bank, rows, progress, source=None):
        self.output, self.progress = output, progress
        self.source = source if source is not None else output / "source"
        self.device = "cpu"
        self.metadata = copy.deepcopy(bank["metadata"])
        self.signature = calculate_model_signature(self.metadata)
        if self.signature != bank["adapter_signature"]:
            raise ValueError("Recovered bank model identity changed")
        self.shape = tuple(self.metadata["latent_shape"])
        self.guidance_scale = self.metadata["guidance_scale"]
        self.max_sequence_length = self.metadata["max_sequence_length"]
        self.num_train_timesteps = self.metadata["scheduler"]["num_train_timesteps"]
        self.pipe = SimpleNamespace(transformer=None, vae=None)
        null = torch.load(
            self.source / "optimization/null_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )
        if null["adapter_signature"] != bank["adapter_signature"]:
            raise ValueError(
                "Recovered null embeddings have a different model identity"
            )
        self.null_embeddings = tuple(value.to("cpu") for value in null["embeddings"])
        frozen_null = torch.load(
            self.source / "optimization/trajectory.null.pt",
            map_location="cpu",
            weights_only=True,
        )
        if (
            frozen_null["bank_sha256"]
            != calculate_file_sha256(self.source / "trajectory/probes/probe_bank.pt")
            or frozen_null["adapter_signature"] != self.signature
            or frozen_null["values"].shape != bank["states"].shape
        ):
            raise ValueError("Recovered frozen null velocities do not match the bank")
        self.frozen_null_velocities = frozen_null["values"].float()
        self.encoder = StagedPromptEncoder(
            self, self.source / "optimization/embeddings"
        )
        self.rows = {row["prompt"]: row for group in rows.values() for row in group}

    def load_transformer(self):
        if self.pipe.transformer is not None:
            return
        from diffusers import SD3Transformer2DModel

        self.progress("loading_cpu_transformer")
        self.pipe.transformer = (
            SD3Transformer2DModel.from_pretrained(
                self.metadata["model_id"],
                revision=self.metadata["revision_sha"],
                subfolder="transformer",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )

    def unload_transformer(self):
        self.pipe.transformer = None
        gc.collect()

    def load_decoder(self):
        from diffusers import AutoencoderKL
        from diffusers.image_processor import VaeImageProcessor

        self.pipe.vae = (
            AutoencoderKL.from_pretrained(
                self.metadata["model_id"],
                revision=self.metadata["revision_sha"],
                subfolder="vae",
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )
        self.pipe.image_processor = VaeImageProcessor(vae_scale_factor=8)

    def encode_prompt(self, prompt):
        return self.encoder.read_embeddings(self.rows[prompt])

    def decode_latents_to_image(self, latents):
        return decode_latents_to_image(self, latents)

    def evaluate_recovered_probe_effect(self, state, flow_time, embeddings, index):
        """Reuse the original null velocity only at its exact frozen probe state."""
        positive, pooled = embeddings
        timestep = torch.full(
            (1,), (1 - flow_time) * self.num_train_timesteps, dtype=torch.float32
        )
        prediction = self.pipe.transformer(
            hidden_states=state.to(dtype=self.pipe.transformer.dtype),
            timestep=timestep,
            encoder_hidden_states=positive,
            pooled_projections=pooled,
            return_dict=False,
        )[0]
        return self.guidance_scale * (
            -prediction.float() - self.frozen_null_velocities[index].unsqueeze(0)
        )


def compute_field(adapter, bank, row, progress, completed, total):
    embeddings = adapter.encode_prompt(row["prompt"])
    values = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, (state, flow_time) in enumerate(
            zip(bank["states"], bank["times"], strict=True), 1
        ):
            recovered_evaluator = getattr(
                adapter, "evaluate_recovered_probe_effect", None
            )
            field = (
                recovered_evaluator(
                    state.unsqueeze(0), float(flow_time), embeddings, index - 1
                )
                if callable(recovered_evaluator)
                else adapter.evaluate_prompt_effect(
                    state.unsqueeze(0), float(flow_time), embeddings
                )
            )
            if not torch.isfinite(field).all():
                raise ValueError("Nonfinite CPU field")
            values.append(field[0].cpu())
            progress(
                completed_fields=completed,
                total_fields=total,
                prompt_id=row["id"],
                completed_probes=index,
                total_probes=len(bank["states"]),
                field_elapsed_seconds=round(time.perf_counter() - started, 2),
            )
    return torch.stack(values)


def build_weighted_cache(path, cases, store, bank, progress):
    """Keep large reference vectors on disk, allocating one field at a time."""
    shape = (len(cases), int(bank["states"][0].numel() * len(bank["states"])))
    metadata_path = path.with_suffix(".json")
    identity = {
        "shape": list(shape),
        "bank_sha256": store.bank_hash,
        "ids": [row["id"] for _, row in cases],
        "field_sha256": {
            f"{split}/{row['id']}": read_json(
                store.root / "trajectory/features" / split / f"{row['id']}.sha256.json"
            )["sha256"]
            for split, row in cases
        },
    }
    if (
        path.exists()
        and metadata_path.exists()
        and read_json(metadata_path) == identity
    ):
        return np.load(path, mmap_mode="r")
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    scale = bank["weights"].float().sqrt()[:, None, None, None]
    for index, (split, row) in enumerate(cases):
        matrix[index] = (
            (store.load_signature(split, row).float() * scale).flatten().numpy()
        )
        if index % 25 == 0:
            progress(completed_vectors=index + 1, total_vectors=len(cases))
    matrix.flush()
    del matrix
    write_json(metadata_path, identity)
    return np.load(path, mmap_mode="r")


def tiled_distance_products(references, queries, safe_count, progress, tile_size=16384):
    """Exact FP64 Gram sums, bounded by a coordinate tile rather than all references."""
    unsafe_count = len(references) - safe_count
    qs = np.zeros((len(queries), safe_count), dtype=np.float64)
    qu = np.zeros((len(queries), unsafe_count), dtype=np.float64)
    su = np.zeros((safe_count, unsafe_count), dtype=np.float64)
    norms = [
        np.zeros(count, dtype=np.float64)
        for count in (len(queries), safe_count, unsafe_count)
    ]
    for start in range(0, references.shape[1], tile_size):
        end = min(start + tile_size, references.shape[1])
        safe = np.array(references[:safe_count, start:end], dtype=np.float64)
        unsafe = np.array(references[safe_count:, start:end], dtype=np.float64)
        query = np.array(queries[:, start:end], dtype=np.float64)
        for norm, array in zip(norms, (query, safe, unsafe), strict=True):
            norm += np.einsum("ij,ij->i", array, array)
        qs += query @ safe.T
        qu += query @ unsafe.T
        su += safe @ unsafe.T
        progress(completed_coordinates=end, total_coordinates=references.shape[1])

    def distance(gram, left, right):
        return np.sqrt(np.maximum(left[:, None] + right[None, :] - 2 * gram, 0))

    return (
        distance(qs, norms[0], norms[1]),
        distance(qu, norms[0], norms[2]),
        distance(su, norms[1], norms[2]),
    )


def reconstruct_distances(
    output, store, bank, groups, selected, distances, settings, progress
):
    queries = [
        ("safe_validation", i, row)
        for i, row in enumerate(
            groups["safe_validation"][: settings["experiment"]["normalization_limit"]]
        )
    ]
    queries += [(row["split"], row["source_index"], row) for row in selected]
    folder = store.root.parent / "cpu_work"
    folder.mkdir(exist_ok=True)
    reference_cases = [
        (split, row)
        for split in ("safe_reference", "unsafe_reference")
        for row in groups[split]
    ]
    progress("preparing_disk_backed_reference_vectors")
    references = build_weighted_cache(
        folder / "references.npy", reference_cases, store, bank, progress
    )
    progress("preparing_disk_backed_query_vectors")
    query_matrix = build_weighted_cache(
        folder / "queries.npy",
        [(split, row) for split, _, row in queries],
        store,
        bank,
        progress,
    )
    progress("verifying_reconstructed_distances_cpu")
    safe, unsafe, cross = tiled_distance_products(
        references, query_matrix, len(groups["safe_reference"]), progress
    )
    checks = {}
    for label, observed in (("safe", safe), ("unsafe", unsafe)):
        expected = np.stack(
            [distances[f"{label}__{split}"][index] for split, index, _ in queries]
        )
        checks[label] = compare_reconstruction(
            observed,
            expected,
            settings["execution"]["reconstruction_rtol"],
            settings["execution"]["reconstruction_atol"],
        )
    passed = all(check["passed"] for check in checks.values())
    write_json(
        output / "reconstruction_cpu.json",
        {
            "passed": passed,
            "checks": checks,
            "device": "cpu",
            "thresholds_refitted": False,
        },
    )
    if not passed:
        raise ValueError(
            "CPU reconstruction differs from retained detector distances; image generation stopped"
        )
    distances["unsafe__safe_reference"] = cross
    np.savez_compressed(
        output / "reference_cross_distances_cpu.npz", unsafe__safe_reference=cross
    )


class DiskOffloadedCPUScorer(ShieldGemmaImageScorer):
    """Use the pinned classifier with CPU RAM bounded and excess weights on disk."""

    def __init__(self, settings, output):
        super().__init__(
            settings["model_id"], settings["revision"], device="cpu", dtype="bfloat16"
        )
        self.offload = (
            Path(read_json(output / "cpu_execution.json")["work_directory"])
            / "shieldgemma_offload"
        )
        self.offload.mkdir(parents=True, exist_ok=True)
        self.metadata.update(cpu_weight_budget_mib=3500, excess_weights="disk_offload")

    def score_image(self, path):
        if self.model is None:
            from transformers import ShieldGemma2ForImageClassification

            self.model = (
                ShieldGemma2ForImageClassification.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    torch_dtype=self.dtype,
                    attn_implementation="eager",
                    device_map="auto",
                    max_memory={"cpu": "3500MiB"},
                    offload_folder=str(self.offload),
                    offload_state_dict=True,
                    offload_buffers=True,
                    low_cpu_mem_usage=True,
                )
                .eval()
                .requires_grad_(False)
            )
        return super().score_image(path)


def recover_native_baseline_fields(source, bank, bank_hash, groups):
    """Recover additional fields already verified bitwise by the original GPU run."""
    baseline_path = source / "optimization/baseline.pt"
    compatibility_path = source / "optimization/trajectory.compatibility.json"
    if not baseline_path.exists() or not compatibility_path.exists():
        return 0
    compatibility = read_json(compatibility_path)
    if not compatibility["passed"] or compatibility["bank_sha256"] != bank_hash:
        raise ValueError("Native baseline compatibility is not verified")
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if (
        baseline["bank_hashes"]["trajectory"] != bank_hash
        or baseline["adapter_signature"] != bank["adapter_signature"]
    ):
        raise ValueError("Native baseline model or bank changed")
    verified = {
        record["id"]
        for record in compatibility["records"]
        if record["bitwise_equal"] and record["embedding_bitwise_equal"]
    }
    lookup = {
        row["id"]: split
        for split in ("safe_reference", "unsafe_reference")
        for row in groups[split]
    }
    recovered = 0
    for record in baseline["records"]:
        row = record["row"]
        if row["id"] not in verified or row["id"] not in lookup:
            raise ValueError("Native baseline row is not a verified training reference")
        path = source / "trajectory/features" / lookup[row["id"]] / f"{row['id']}.pt"
        if path.exists() and path.with_suffix(".sha256.json").exists():
            continue
        field = record["fields"]["trajectory"]
        if field.shape != bank["states"].shape or not torch.isfinite(field).all():
            raise ValueError("Invalid recovered native baseline field")
        save_field_signature(
            path, row["id"], field, bank_hash, bank["adapter_signature"]
        )
        write_json(
            path.with_suffix(".sha256.json"),
            {
                "sha256": calculate_file_sha256(path),
                "origin": "recovered_verified_native_baseline",
            },
        )
        recovered += 1
    return recovered


def generate_cpu(output, threads, progress):
    manifest = read_json(output / "experiment.json")
    settings = manifest["fingerprint"]["settings"]
    migration = output / "cpu_migration/recovery.json"
    source = (
        Path(read_json(migration)["source_directory"])
        if migration.exists()
        else output / "source"
    )
    for name, digest in manifest["fingerprint"]["inputs"].items():
        if calculate_file_sha256(source / name) != digest:
            raise ValueError(f"Frozen parent input changed: {name}")
    for name, digest in manifest["fingerprint"]["implementation"].items():
        if calculate_file_sha256(Path.cwd() / name) != digest:
            raise ValueError(f"Original repair implementation changed: {name}")
    parent = read_json(source / "experiment.json")
    groups = {
        split: read_jsonl(source / "trajectory/splits" / f"{split}.jsonl")
        for split in parent["split_ids"]
    }
    selected = manifest["selected_cohort"]
    bank_path = source / "trajectory/probes/probe_bank.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    bank_hash = calculate_file_sha256(bank_path)
    baseline_recovered = recover_native_baseline_fields(source, bank, bank_hash, groups)
    rebuilt = {split: groups[split] for split in ("safe_reference", "unsafe_reference")}
    rebuilt["safe_validation"] = groups["safe_validation"][
        : settings["experiment"]["normalization_limit"]
    ]
    for split in ("safe_test", "unsafe_test"):
        rebuilt[split] = [row for row in selected if row["split"] == split]
    adapter = CachedCPUAdapter(output, bank, rebuilt, progress, source=source)
    store = RebuiltSignatureStore(source, bank, bank_hash, adapter.signature)
    cases = [(split, row) for split, rows in rebuilt.items() for row in rows]
    # Validate caches before allocating the model or doing any new inference.
    cached = set()
    for split, row in cases:
        path = source / "trajectory/features" / split / f"{row['id']}.pt"
        if path.exists() and path.with_suffix(".sha256.json").exists():
            store.load_signature(split, row)
            cached.add(row["id"])
    progress(
        "resuming_recovered_fields", reused_fields=len(cached), total_fields=len(cases)
    )
    execution_path = output / "cpu_execution.json"
    execution = {
        "device": "cpu",
        "source_directory": str(source),
        "work_directory": str(source.parent / "cpu_work"),
        "threads": threads,
        "dtype": "float16",
        "probe_batch_size": 1,
        "original_experiment_sha256": calculate_file_sha256(output / "experiment.json"),
        "implementation": {
            "backend/runtime/repair_cpu.py": calculate_file_sha256(Path(__file__))
        },
        "reused_fields_at_launch": len(cached),
        "additional_native_baseline_fields_recovered": baseline_recovered,
        "frozen_probe_null_velocities": "reused original verified cache; actual native negative velocity is evaluated along generated trajectories",
        "cached_embedding_files": len(
            list((source / "optimization/embeddings").glob("*/*.pt"))
        ),
        "bank_sha256": bank_hash,
        "thresholds_refitted": False,
        "execution_change": "CPU kernels with one probe per call; no claim of GPU bitwise equivalence; retained detector distance tolerances still enforced",
    }
    if execution_path.exists():
        previous = read_json(execution_path)
        for key in (
            "implementation",
            "bank_sha256",
            "original_experiment_sha256",
            "dtype",
            "probe_batch_size",
        ):
            if previous[key] != execution[key]:
                raise ValueError(f"Frozen CPU execution changed: {key}")
    else:
        write_json(execution_path, execution)
    if len(cached) < len(cases):
        adapter.load_transformer()
    completed = len(cached)
    for split, row in cases:
        if row["id"] in cached:
            continue
        progress(
            "rebuild_reference_fields_cpu",
            completed_fields=completed,
            total_fields=len(cases),
            reused_fields=len(cached),
            prompt_id=row["id"],
        )
        field = compute_field(adapter, bank, row, progress, completed, len(cases))
        path = source / "trajectory/features" / split / f"{row['id']}.pt"
        save_field_signature(path, row["id"], field, bank_hash, adapter.signature)
        write_json(
            path.with_suffix(".sha256.json"),
            {
                "sha256": calculate_file_sha256(path),
                "execution_device": "cpu",
                "probe_batch_size": 1,
            },
        )
        del field
        completed += 1
        progress(completed_fields=completed)
    adapter.unload_transformer()
    with np.load(source / "trajectory/distances.npz") as archive:
        distances = {name: archive[name].copy() for name in archive.files}
    reconstruct_distances(
        output, store, bank, groups, selected, distances, settings, progress
    )
    config = load_signature_space_config(source, "trajectory")
    config["model"].update(device="cpu", cpu_offload=False)
    config["experiment"].update(probe_batch_size=1, threads=threads)
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
    write_json(output / "normalization_cpu.json", experiment.normalization)
    adapter.load_transformer()
    adapter.load_decoder()
    experiment.adapter = adapter

    def completed_case(case, _records, completed, total):
        progress(
            "image_generation_cpu",
            completed_prompts=completed,
            total_prompts=total,
            prompt_id=case[2]["id"],
        )

    cases = [(row["split"], row["source_index"], row) for row in selected]
    records = run_repair_execution(
        experiment, "pilot", cases=cases, on_case_completed=completed_case
    )
    if len(records) != manifest["expected_generation_records"]:
        raise ValueError("Missing repair comparison records")
    run_repair_reporting(experiment, "pilot", records)


def run(output, threads=4):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    progress = CPUProgress(output, threads)
    try:
        generate_cpu(output, threads, progress)
        gc.collect()
        progress("independent_image_scoring_cpu")
        settings = read_json(output / "experiment.json")["fingerprint"]["settings"][
            "image_scoring"
        ]
        scorer = DiskOffloadedCPUScorer(settings, output)
        run_repair_scoring(
            [
                "--output",
                str(output),
                "--cohort",
                "pilot",
                "--device",
                "cpu",
                "--dtype",
                "bfloat16",
                "--score-name",
                "shieldgemma2_cpu",
                "--model-id",
                settings["model_id"],
                "--revision",
                settings["revision"],
                "--threshold",
                str(settings["threshold"]),
            ],
            scorer=scorer,
        )
        progress(
            "completed",
            status="completed",
            completed_prompts=progress.expected["expected_prompts"],
            generation_records=progress.expected["expected_generation_records"],
        )
    except BaseException:
        progress("failed", status="error", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.output.resolve(), args.threads)
