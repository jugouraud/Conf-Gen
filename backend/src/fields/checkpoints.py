"""Paired legal/illegal prompt validation and detection-only decisions."""

from pathlib import Path

from backend.src.common.storage import (
    calculate_file_sha256,
    link_or_copy_frozen_file,
    read_json,
    read_jsonl,
    write_json,
)
from backend.src.fields.signatures import (
    calculate_model_signature,
    load_signature_space_config,
    read_probe_bank,
    validate_cached_signature,
)


def reuse_paired_detection_checkpoints(source, output):
    """Link only exact same-split prompts evaluated on the identical model and bank."""
    source, output = Path(source), Path(output)
    original = read_json(source / "experiment.json")
    expanded = read_json(output / "experiment.json")
    if expanded["continuation"]["source_experiment_sha256"] != calculate_file_sha256(
        source / "experiment.json"
    ):
        raise ValueError("Continuation source manifest changed")
    if (
        original["config"]["model"] != expanded["config"]["model"]
        or original["config"]["experiment"]["guidance_scale"]
        != expanded["config"]["experiment"]["guidance_scale"]
    ):
        raise ValueError("Cannot reuse checkpoints after model or guidance changes")
    fit = output / "trajectory/splits/safe_probe_fit.jsonl"
    if calculate_file_sha256(fit) != expanded["probe_fit_sha256"]:
        raise ValueError("Frozen probe fit split changed")
    source_bank = source / "trajectory/probes/probe_bank.pt"
    bank_path = output / "trajectory/probes/probe_bank.pt"
    link_or_copy_frozen_file(source_bank, bank_path)
    bank = read_probe_bank(load_signature_space_config(output, "trajectory"))
    bank_hash = calculate_file_sha256(bank_path)
    adapter_signature = calculate_model_signature(bank["metadata"])
    reused = {}
    for split in expanded["counts"]:
        old_path = source / "trajectory/splits" / f"{split}.jsonl"
        if calculate_file_sha256(old_path) != original["split_sha256"][split]:
            raise ValueError("Source split bytes changed")
        original_rows = {row["id"]: row for row in read_jsonl(old_path)}
        new_path = output / "trajectory/splits" / f"{split}.jsonl"
        if calculate_file_sha256(new_path) != expanded["split_sha256"][split]:
            raise ValueError("Expanded split bytes changed")
        for row in read_jsonl(new_path):
            relative = f"trajectory/features/{split}/{row['id']}.pt"
            checkpoint = source / relative
            if original_rows.get(row["id"]) != row or not checkpoint.exists():
                continue
            validate_cached_signature(checkpoint, bank_hash, adapter_signature)
            link_or_copy_frozen_file(checkpoint, output / relative)
            reused[relative] = calculate_file_sha256(checkpoint)
    receipt = {
        "source_experiment_sha256": calculate_file_sha256(source / "experiment.json"),
        "bank_sha256": bank_hash,
        "features": reused,
        "reused_features": len(reused),
        "policy": "Identical prompt records and split; unchanged model, guidance and frozen bank",
    }
    write_json(output / "checkpoint_reuse.provenance.json", receipt)
    return receipt
