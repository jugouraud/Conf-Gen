"""Public resident field artifacts and native compatibility gates shared by detection and repair."""

import shutil
from pathlib import Path

import torch

from backend.src.common.storage import calculate_file_sha256, read_json, write_json
from backend.src.fields.distances import ResidentFieldDistances
from backend.src.fields.signatures import (
    build_probe_bank,
    build_source_slice_bank,
    evaluate_probe_field,
    load_signature_space_config,
    read_probe_bank,
    read_probe_fit_rows,
    resolve_probe_bank_path,
    sample_safe_reference_trajectories,
    save_field_signature,
    validate_cached_signature,
)
from backend.src.inference.encoding import (
    ResidentProbeEvaluator,
    StagedPromptEncoder,
    compare_signature_values,
    save_atomic_tensor,
)
from backend.src.inference.flow import FlowModelAdapter


def record_resident_status(output: Path, phase: str, **values) -> None:
    path = output / "status.json"
    temporary = path.with_suffix(".tmp")
    write_json(temporary, {"status": "running", "phase": phase, **values})
    temporary.replace(path)


def create_resident_probe_banks(
    output: Path, adapter: FlowModelAdapter
) -> tuple[dict, dict[str, str]]:
    """Publish frozen trajectory/source banks with identical model and split provenance."""
    cfg = load_signature_space_config(output, "trajectory")
    path = resolve_probe_bank_path(cfg)
    if not path.exists():
        record_resident_status(output, "build/train_only_trajectory_bank")
        _, fit = read_probe_fit_rows(cfg)
        states, provenance, midpoints = sample_safe_reference_trajectories(
            cfg, adapter, fit
        )
        save_atomic_tensor(
            path, build_probe_bank(cfg, adapter, fit, states, provenance, midpoints)
        )
    trajectory = read_probe_bank(cfg, adapter)
    source_cfg = load_signature_space_config(output, "source_slice")
    for split in read_json(output / "experiment.json")["counts"]:
        target = output / "source_slice/splits" / f"{split}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "trajectory/splits" / target.name, target)
    source_path = resolve_probe_bank_path(source_cfg)
    if not source_path.exists():
        save_atomic_tensor(source_path, build_source_slice_bank(trajectory))
    banks = {
        "trajectory": trajectory,
        "source_slice": read_probe_bank(source_cfg, adapter),
    }
    hashes = {
        space: calculate_file_sha256(
            resolve_probe_bank_path(load_signature_space_config(output, space))
        )
        for space in banks
    }
    receipt = {
        "banks": hashes,
        "adapter_signature": adapter.signature,
        "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
    }
    receipt_path = output / "banks.provenance.json"
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError("Frozen bank identity changed")
    write_json(receipt_path, receipt)
    return banks, hashes


def load_native_baseline(
    output: Path, adapter: FlowModelAdapter, groups: dict, banks: dict, hashes: dict
) -> dict:
    """Read or create native training-only embeddings and field compatibility evidence."""
    target = output / "optimization/baseline.pt"
    identity = {"bank_hashes": hashes, "adapter_signature": adapter.signature}
    if target.exists():
        baseline = torch.load(target, map_location="cpu", weights_only=True)
        if any(baseline[k] != v for k, v in identity.items()):
            raise ValueError("Stale native compatibility baseline")
        return baseline
    records = []
    for row in (groups["safe_reference"][0], groups["unsafe_reference"][0]):
        embeddings = adapter.encode_prompt(row["prompt"])
        records.append(
            {
                "row": row,
                "embeddings": [v.cpu() for v in embeddings],
                "fields": {
                    s: evaluate_probe_field(
                        load_signature_space_config(output, s),
                        adapter,
                        bank,
                        embeddings,
                    )
                    for s, bank in banks.items()
                },
            }
        )
    baseline = {**identity, "records": records}
    save_atomic_tensor(target, baseline)
    return baseline


def encode_resident_prompts(
    output: Path, encoder: StagedPromptEncoder, groups: dict
) -> None:
    rows = [row for records in groups.values() for row in records]
    for component in encoder.components:
        encoder.prepare_component(component)
        for index, row in enumerate(rows, 1):
            encoder.encode_component(component, row)
            if index % 25 == 0 or index == len(rows):
                record_resident_status(
                    output, f"encode/{component}", completed=index, total=len(rows)
                )
    encoder.prepare_transformer()


def verify_resident_compatibility(
    output: Path,
    space: str,
    evaluator: ResidentProbeEvaluator,
    encoder: StagedPromptEncoder,
    baseline: dict,
) -> None:
    """Require bitwise native fields and FP64 distance tolerances before using caches."""
    checks = []
    for record in baseline["records"]:
        embeddings = encoder.read_embeddings(record["row"])
        equal = all(
            torch.equal(a, b)
            for a, b in zip(embeddings, record["embeddings"], strict=True)
        )
        checks.append(
            {
                "id": record["row"]["id"],
                "embedding_bitwise_equal": equal,
                **compare_signature_values(
                    record["fields"][space], evaluator.evaluate_field(embeddings)
                ),
            }
        )
    report = {
        "passed": all(
            r["embedding_bitwise_equal"] and r["bitwise_equal"] for r in checks
        ),
        "records": checks,
        "bank_sha256": evaluator.bank_hash,
        "batch_size": evaluator.batch_size,
        "adapter_signature": evaluator.adapter.signature,
    }
    write_json(output / "optimization" / f"{space}.compatibility.json", report)
    if not report["passed"]:
        raise ValueError(
            f"{space} failed native bitwise compatibility on training prompts"
        )
    # Validate CUDA distance arithmetic against direct CPU FP64 subtraction,
    # using only the same two fixed training examples.
    fields = [r["fields"][space] for r in baseline["records"]]
    scale = torch.tensor(1 / len(evaluator.states), dtype=torch.float32).sqrt()
    weighted = [field.float() * scale for field in fields]
    expected = torch.linalg.vector_norm(weighted[0].double() - weighted[1].double())
    cache = ResidentFieldDistances(
        1,
        fields[0].shape,
        torch.full((len(evaluator.states),), 1 / len(evaluator.states)),
        device=evaluator.adapter.device,
    )
    cache.append(fields[0])
    actual = cache.measure([fields[1]])[0, 0].cpu()
    passed = torch.isclose(actual, expected, rtol=1e-10, atol=1e-8).item()
    write_json(
        output / "optimization" / f"{space}.distance_compatibility.json",
        {
            "passed": passed,
            "expected_cpu_fp64": expected.item(),
            "actual": actual.item(),
            "absolute_error": abs(actual.item() - expected.item()),
            "rtol": 1e-10,
            "atol": 1e-8,
            "prompt_ids": [r["row"]["id"] for r in baseline["records"]],
        },
    )
    if not passed:
        raise ValueError("GPU distances failed direct CPU FP64 compatibility")


def read_resident_field(
    path: Path,
    row: dict,
    bank_hash: str,
    adapter_signature: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Read FP32 probe-order effects only after checksum, identity and shape validation."""
    sidecar = path.with_suffix(".sha256.json")
    if not sidecar.exists() or read_json(sidecar)["sha256"] != calculate_file_sha256(
        path
    ):
        raise ValueError(f"Incomplete or changed field checkpoint: {path}")
    validate_cached_signature(path, bank_hash, adapter_signature)
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (
        record["id"] != row["id"]
        or tuple(record["values"].shape) != tuple(shape)
        or record["values"].dtype != torch.float32
        or not torch.isfinite(record["values"]).all()
    ):
        raise ValueError("Field checkpoint identity or shape changed")
    return record["values"]


def extract_resident_field(
    output: Path,
    space: str,
    split: str,
    row: dict,
    evaluator: ResidentProbeEvaluator,
    encoder: StagedPromptEncoder,
) -> torch.Tensor:
    """Return the model-device field, committing its CPU checkpoint and hash sidecar."""
    path = output / space / "features" / split / f"{row['id']}.pt"
    if path.exists() and path.with_suffix(".sha256.json").exists():
        return read_resident_field(
            path,
            row,
            evaluator.bank_hash,
            evaluator.adapter.signature,
            evaluator.states.shape,
        ).to(evaluator.adapter.device)
    field = evaluator.evaluate_field(
        encoder.read_embeddings(row), output_device=evaluator.adapter.device
    )
    if not torch.isfinite(field).all():
        raise ValueError("Nonfinite resident field")
    # A single D2H checkpoint per prompt, retained remotely. The scoring path
    # consumes the original GPU field; it never reloads this checkpoint.
    save_field_signature(
        path, row["id"], field.cpu(), evaluator.bank_hash, evaluator.adapter.signature
    )
    write_json(
        path.with_suffix(".sha256.json"), {"sha256": calculate_file_sha256(path)}
    )
    return field


class RebuiltSignatureStore:
    def __init__(self, source, bank, bank_hash, signature):
        self.root, self.bank, self.bank_hash, self.signature = (
            source,
            bank,
            bank_hash,
            signature,
        )

    def load_signature(self, split, row):
        return read_resident_field(
            self.root / "trajectory/features" / split / f"{row['id']}.pt",
            row,
            self.bank_hash,
            self.signature,
            self.bank["states"].shape,
        )
