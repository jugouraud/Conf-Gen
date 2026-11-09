"""Rebuild paired-detector geometry, execute frozen repairs, and export each case."""

import argparse
import gc
import traceback
from pathlib import Path

import numpy as np
import torch

from backend.src.common.storage import (
    calculate_file_sha256,
    read_json,
    read_jsonl,
    write_json,
)
from backend.src.fields.distances import ResidentFieldDistances
from backend.src.fields.resident import (
    RebuiltSignatureStore,
    create_resident_probe_banks,
    encode_resident_prompts,
    extract_resident_field,
    load_native_baseline,
    verify_resident_compatibility,
)
from backend.src.fields.signatures import load_signature_space_config
from backend.src.inference.encoding import ResidentProbeEvaluator, StagedPromptEncoder
from backend.src.inference.flow import FlowModelAdapter, decode_latents_to_image
from backend.src.repair.artifacts import compare_reconstruction, export_files, update
from backend.src.repair.contracts import RepairExperiment
from backend.src.repair.execute import run_repair_execution
from backend.src.repair.normalization import fit_safe_deviation_normalization
from backend.src.repair.report import run_repair_reporting
from backend.src.repair.score import run_repair_scoring


class CachedRepairAdapter:
    """Reuse native cached embeddings while evaluating actual velocity at each ODE state."""

    def __init__(self, adapter, encoder, groups):
        self.base, self.encoder = adapter, encoder
        self.rows = {row["prompt"]: row for rows in groups.values() for row in rows}

    def __getattr__(self, name):
        return getattr(self.base, name)

    def encode_prompt(self, prompt):
        return tuple(
            value.to(self.base.device)
            for value in self.encoder.read_embeddings(self.rows[prompt])
        )

    def decode_latents_to_image(self, latents):
        self.pipe.vae.to(self.device)
        try:
            return decode_latents_to_image(self.base, latents)
        finally:
            self.pipe.vae.to("cpu")


def measure_rebuilt_distances(
    output, store, groups, bank, distances, selected, settings
):
    checks = []
    queries = [
        ("safe_validation", i, row)
        for i, row in enumerate(
            groups["safe_validation"][: settings["experiment"]["normalization_limit"]]
        )
    ]
    queries += [(row["split"], row["source_index"], row) for row in selected]
    cache = ResidentFieldDistances(
        len(queries),
        bank["states"].shape,
        bank["weights"],
        device="cuda",
        query_batch_size=4,
    )
    for split, _, row in queries:
        cache.append(store.load_signature(split, row))
    options = settings["execution"]
    for label in ("safe", "unsafe"):
        references = groups[f"{label}_reference"]
        for start in range(0, len(references), 8):
            rows = references[start : start + 8]
            measured = (
                cache.measure(
                    [store.load_signature(f"{label}_reference", row) for row in rows]
                )
                .cpu()
                .numpy()
            )
            expected = np.stack(
                [
                    distances[f"{label}__{split}"][index, start : start + len(rows)]
                    for split, index, _ in queries
                ]
            )
            check = compare_reconstruction(
                measured,
                expected,
                options["reconstruction_rtol"],
                options["reconstruction_atol"],
            )
            checks.append(check)
            update(
                output,
                f"verify_reconstructed_distances/{label}",
                completed=start + len(rows),
                total=len(references),
            )
            if not check["passed"]:
                write_json(
                    output / "reconstruction.json", {"passed": False, "checks": checks}
                )
                raise ValueError(
                    "Rebuilt fields disagree with the retained detector; no images generated"
                )
    del cache
    torch.cuda.empty_cache()
    # Paired detection retained within-class geometry. Repair additionally needs
    # unsafe-to-safe reference distances to exclude unsafe-adjacent safe anchors.
    safe = groups["safe_reference"]
    unsafe = groups["unsafe_reference"]
    cache = ResidentFieldDistances(
        len(safe),
        bank["states"].shape,
        bank["weights"],
        device="cuda",
        query_batch_size=4,
    )
    for row in safe:
        cache.append(store.load_signature("safe_reference", row))
    cross = np.empty((len(safe), len(unsafe)), dtype=np.float64)
    for start in range(0, len(unsafe), 8):
        rows = unsafe[start : start + 8]
        cross[:, start : start + len(rows)] = (
            cache.measure(
                [store.load_signature("unsafe_reference", row) for row in rows]
            )
            .cpu()
            .numpy()
        )
        update(
            output,
            "reconstruct_reference_cross_distances",
            completed=start + len(rows),
            total=len(unsafe),
        )
    distances["unsafe__safe_reference"] = cross
    np.savez_compressed(
        output / "reference_cross_distances.npz", unsafe__safe_reference=cross
    )
    write_json(
        output / "reconstruction.json",
        {
            "passed": True,
            "checks": checks,
            "query_count": len(queries),
            "thresholds_refitted": False,
            "bank_policy": "regenerated from frozen training inputs, seeds, model and settings; serialized identity may differ",
        },
    )
    del cache
    torch.cuda.empty_cache()


def generate(output):
    manifest = read_json(output / "experiment.json")
    settings = manifest["fingerprint"]["settings"]
    source = output / "source"
    for name, digest in manifest["fingerprint"]["inputs"].items():
        if calculate_file_sha256(source / name) != digest:
            raise ValueError(f"Frozen parent input changed: {name}")
    for name, digest in manifest["fingerprint"]["implementation"].items():
        if calculate_file_sha256(Path.cwd() / name) != digest:
            raise ValueError(f"Frozen repair code changed: {name}")
    torch.set_num_threads(settings["experiment"]["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = read_json(source / "experiment.json")
    groups = {
        split: read_jsonl(source / "trajectory/splits" / f"{split}.jsonl")
        for split in parent["split_ids"]
    }
    selected = manifest["selected_cohort"]
    rebuild = {
        split: list(rows)
        for split, rows in groups.items()
        if split.endswith("reference")
    }
    rebuild["safe_validation"] = groups["safe_validation"][
        : settings["experiment"]["normalization_limit"]
    ]
    for split in ("safe_test", "unsafe_test"):
        rebuild[split] = [row for row in selected if row["split"] == split]
    config = load_signature_space_config(source, "trajectory")
    update(output, "loading_generation_model")
    adapter = FlowModelAdapter(config)
    update(output, "rebuild_probe_bank")
    banks, hashes = create_resident_probe_banks(source, adapter)
    bank, bank_hash = banks["trajectory"], hashes["trajectory"]
    baseline = load_native_baseline(source, adapter, groups, banks, hashes)
    encoder = StagedPromptEncoder(adapter, source / "optimization/embeddings")
    encode_resident_prompts(source, encoder, rebuild)
    evaluator = ResidentProbeEvaluator(
        adapter,
        bank,
        bank_hash,
        source / "optimization/trajectory.null.pt",
        parent["config"]["execution"]["probe_batch_size"],
    )
    evaluator.prepare_null_cache()
    verify_resident_compatibility(source, "trajectory", evaluator, encoder, baseline)
    total, completed = sum(map(len, rebuild.values())), 0
    for split, rows in rebuild.items():
        for row in rows:
            field = extract_resident_field(
                source, "trajectory", split, row, evaluator, encoder
            )
            del field
            completed += 1
            update(
                output,
                "rebuild_reference_fields",
                completed=completed,
                total=total,
                split=split,
            )
    del evaluator, baseline, banks
    adapter.pipe.transformer.to("cpu")
    torch.cuda.empty_cache()
    store = RebuiltSignatureStore(source, bank, bank_hash, adapter.signature)
    with np.load(source / "trajectory/distances.npz") as archive:
        distances = {name: archive[name].copy() for name in archive.files}
    measure_rebuilt_distances(
        output, store, groups, bank, distances, selected, settings
    )
    config["experiment"]["probe_batch_size"] = settings["execution"]["probe_batch_size"]
    adapter.pipe.transformer.to(adapter.device)
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
    experiment.adapter = CachedRepairAdapter(adapter, encoder, rebuild)
    normalization = fit_safe_deviation_normalization(experiment)
    path = output / "normalization.json"
    if path.exists() and read_json(path) != normalization:
        raise ValueError("Normalization changed on resume")
    write_json(path, normalization)
    experiment.normalization = normalization
    write_json(
        output / "rebuilt_bank.json",
        {
            "sha256": bank_hash,
            "parent_sha256": manifest["parent_bank_sha256"],
            "adapter_signature": adapter.signature,
        },
    )

    def completed_case(case, _records, completed, total):
        row = case[2]
        receipt = output / "pilot/records" / f"{row['split']}_{row['id']}.json"
        export_files(
            output,
            row["id"],
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


def run_worker(output):
    try:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run_worker(parser.parse_args().output.resolve())
