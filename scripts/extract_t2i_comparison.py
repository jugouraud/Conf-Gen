"""Extract only new unsafe signatures on the existing paired probe banks."""

import argparse
from pathlib import Path

import torch

from backend.flow.common import ROOT, sha256
from backend.flow.feature_store import feature_file, save_feature, validate_cached_feature
from backend.flow.field_sampling import evaluate_probe_field
from backend.flow.model import FlowAdapter
from backend.flow.probe_bank import bank_file, read_bank
from backend.flow.t2i_prompts import UNSAFE_SPLITS, read_json, read_jsonl
from backend.flow.comparison import VARIANTS, space_config
from backend.flow.extraction_run import Run


def run_extraction(output, adapter=None, *, splits=UNSAFE_SPLITS,
                   provenance_name="unsafe_features.provenance.json", precomputed=None):
    checkpoint = Run(output)
    manifest = read_json(output / "experiment.json")
    checkpoint.state["total_features"] = len(VARIANTS) * sum(manifest["counts"][s] for s in splits)
    checkpoint.update()
    configs = {s: space_config(output, s) for s in VARIANTS}
    if adapter is None:
        adapter = FlowAdapter(configs["trajectory"])
    banks = {s: read_bank(configs[s], adapter) for s in VARIANTS}
    hashes = {s: sha256(bank_file(configs[s])) for s in VARIANTS}
    cached = {} if precomputed is None else precomputed["features"]
    if precomputed is not None and precomputed["experiment_sha256"] != sha256(output / "experiment.json"):
        raise ValueError("Restored features belong to a different experiment")
    try:
        for split in splits:
            checkpoint.update(phase=f"extract/{split}")
            rows = read_jsonl(output / "trajectory/splits" / f"{split}.jsonl")
            for row in rows:
                relative = lambda s: f"{s}/features/{split}/{row['id']}.pt"
                missing = [s for s in VARIANTS if not feature_file(configs[s], split, row["id"]).exists()
                           and relative(s) not in cached]
                embeddings = adapter.encode(row["prompt"]) if missing else None
                for space in VARIANTS:
                    path = feature_file(configs[space], split, row["id"])
                    if not path.exists() and relative(space) in cached:
                        # Verified local checkpoints are transferred separately;
                        # measurement checks their bytes before using any vector.
                        checkpoint.hashes[space][f"{split}/{row['id']}.pt"] = cached[relative(space)]
                        checkpoint.state["completed_features"] += 1
                        key = f"{space}/{split}"
                        checkpoint.state["counts"][key] = checkpoint.state["counts"].get(key, 0) + 1
                        continue
                    if path.exists():
                        validate_cached_feature(path, hashes[space], adapter.signature)
                    else:
                        field = evaluate_probe_field(configs[space], adapter, banks[space], embeddings)
                        if not torch.isfinite(field).all():
                            raise ValueError("Non-finite unsafe signature")
                        save_feature(path, row["id"], field, hashes[space], adapter.signature)
                    checkpoint.register(path, space, split, row["id"])
                checkpoint.update()
            checkpoint.bundle(force=True)
        for space in VARIANTS:
            from backend.flow.common import write_json
            path = output / space / provenance_name
            write_json(path, {"bank_sha256": hashes[space], "features": checkpoint.hashes[space],
                              "experiment_sha256": sha256(output / "experiment.json"),
                              "adapter_signature": adapter.signature})
            checkpoint.register(path)
        checkpoint.bundle(force=True)
        checkpoint.update(status="complete", phase="extraction_complete")
    except BaseException:
        import traceback
        checkpoint.bundle(force=True)
        checkpoint.update(status="failed", error=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/flow_t2i_two_mode_1000")
    parser.add_argument("--split", choices=(*UNSAFE_SPLITS, "additional_safe_test"))
    parser.add_argument("--precomputed", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    options = ({"splits": (args.split,), "provenance_name": "additional_features.provenance.json"}
               if args.split == "additional_safe_test" else
               {"splits": (args.split,)} if args.split else {})
    run_extraction(args.output.resolve(), precomputed=read_json(args.precomputed) if args.precomputed else None, **options)


if __name__ == "__main__":
    main()
