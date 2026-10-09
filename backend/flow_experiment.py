"""Fixed-probe, safe-only velocity novelty experiment using pretrained Sana.

CLI: python -m backend.flow_experiment {prepare,probes,extract,calibrate,evaluate,report}
The existing CLIP/COCO pipeline is unchanged. All model operations are inference-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("safe_probe_source", "safe_reference", "safe_validation", "safe_test", "unsafe_test")
SAFE_INPUT = {"safe_reference", "safe_calibration", "safe_test"}


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def resolved_path(cfg: dict, name: str) -> Path:
    path = Path(cfg["paths"][name])
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def canonical(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.casefold(), flags=re.UNICODE).strip()


def split_corpus(frame, source_count: int, seed: int):
    """Safe priority on collisions, deterministic disjoint probe prompts."""
    required = {"prompt_key", "prompt", "split", "label", "source",
                "eligible_for_region", "evaluation_only"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing corpus fields: {sorted(required - set(frame.columns))}")
    rows = frame.to_dict("records")
    order = {"safe_reference": 0, "safe_calibration": 1, "safe_test": 2}
    rows.sort(key=lambda r: (order.get(str(r["split"]), 3), str(r["prompt_key"])))
    result = {name: [] for name in SPLITS}
    seen_keys, seen_text = set(), set()
    dropped = 0
    for row in rows:
        previous = str(row["split"])
        safe = previous in SAFE_INPUT
        if safe:
            if (row["label"] != "safe" or row["source"] != "diffusiondb"
                    or bool(row["eligible_for_region"]) != (previous != "safe_test")
                    or bool(row["evaluation_only"]) != (previous == "safe_test")):
                raise ValueError(f"Invalid safe split metadata: {previous}")
        elif (row["label"] != "risky" or not previous.startswith("toxic_test")
              or bool(row["eligible_for_region"]) or not bool(row["evaluation_only"])):
            raise ValueError(f"Risky/evaluation-only constraint violated: {previous}")
        key, prompt = str(row["prompt_key"]), str(row["prompt"]).strip()
        norm = canonical(prompt)
        if not norm:
            raise ValueError("Empty prompt in corpus")
        if key in seen_keys or norm in seen_text:
            dropped += 1
            continue
        seen_keys.add(key)
        seen_text.add(norm)
        target = ({"safe_reference": "safe_reference",
                   "safe_calibration": "safe_validation",
                   "safe_test": "safe_test"}[previous] if safe else "unsafe_test")
        result[target].append({"id": key, "prompt": prompt,
                               "label": "safe" if safe else "risky",
                               "source": str(row["source"]), "source_split": previous})
    reference = sorted(result["safe_reference"], key=lambda row:
                       hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    if not 1 <= source_count < len(reference):
        raise ValueError("Need a nonempty disjoint safe probe source and safe reference")
    result["safe_probe_source"] = reference[:source_count]
    result["safe_reference"] = reference[source_count:]
    for values in result.values():
        values.sort(key=lambda item: item["id"])
    if not result["safe_validation"] or not result["safe_test"] or not result["unsafe_test"]:
        raise ValueError("Missing required safe validation or held-out test examples")
    return result, dropped


def read_rows(cfg: dict, split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    path = resolved_path(cfg, "output") / "splits" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any((row["label"] == "risky") != (split == "unsafe_test")
                       for row in rows):
        raise ValueError(f"Invalid or empty split: {path}")
    return rows


def limited_rows(cfg: dict, split: str, limit: int | None = None) -> list[dict]:
    rows = read_rows(cfg, split)
    caps = {"safe_reference": "reference_limit", "safe_validation": "validation_limit",
            "safe_test": "test_limit", "unsafe_test": "test_limit"}
    cap = int(cfg["experiment"].get(caps.get(split, ""), 0)) if split in caps else 0
    if cap > 0:
        rows = rows[:cap]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    return rows


def prepare(cfg: dict) -> None:
    import pandas as pd
    root = resolved_path(cfg, "output")
    corpus = resolved_path(cfg, "corpus")
    result, dropped = split_corpus(pd.read_parquet(corpus),
                                   int(cfg["probes"]["source_prompts"]),
                                   int(cfg["experiment"]["seed"]))
    manifest = {
        "corpus": str(corpus), "corpus_sha256": sha256(corpus),
        "seed": cfg["experiment"]["seed"], "canonical_duplicates_dropped": dropped,
        "counts": {name: len(rows) for name, rows in result.items()},
        "policy": "DiffusionDB safe reference supplies probes/reference; "
                  "safe calibration supplies threshold; all risky rows evaluation only",
    }
    folder = root / "splits"
    old = folder / "manifest.json"
    if old.exists():
        if json.loads(old.read_text()) == manifest:
            print(f"Unchanged frozen split manifest: {old}")
            return
        raise FileExistsError("Split manifest already exists and differs; use a new output directory")
    if (root / "probes").exists() or (root / "features").exists():
        raise FileExistsError("Existing probes/features require a new output directory")
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in result.items():
        with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(old, manifest)
    print(f"Prepared disjoint safe-only splits: {manifest['counts']}")


class SanaAdapter:
    """Expose model-native effective inference velocity with no image decoding."""

    def __init__(self, cfg: dict):
        import torch
        from diffusers import SanaPipeline
        from huggingface_hub import model_info

        self.torch = torch
        model = cfg["model"]
        self.device = str(model["device"])
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[model["dtype"]]
        resolved_revision = model_info(model["id"], revision=model["revision"]).sha
        kwargs = {"revision": resolved_revision, "torch_dtype": dtype}
        if model.get("variant"):
            kwargs["variant"] = model["variant"]
        self.pipe = SanaPipeline.from_pretrained(model["id"], **kwargs).to(self.device)
        self.pipe.transformer.eval()
        self.pipe.text_encoder.eval()
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            self.pipe.text_encoder.to(torch.bfloat16)
        self.guidance = float(cfg["experiment"]["guidance_scale"])
        if self.guidance <= 0:
            raise ValueError("Positive guidance_scale required")
        self.steps = int(cfg["probes"]["inference_steps"])
        self.pipe.scheduler.set_timesteps(self.steps, device=self.device)
        self.shape = (int(self.pipe.transformer.config.in_channels),
                      int(model["height"]) // self.pipe.vae_scale_factor,
                      int(model["width"]) // self.pipe.vae_scale_factor)
        if min(self.shape) <= 0:
            raise ValueError("Invalid native latent dimensions")
        self.metadata = {
            "model_id": model["id"], "revision_sha": resolved_revision,
            "text_encoder": type(self.pipe.text_encoder).__name__,
            "dtype": str(dtype), "text_encoder_dtype": str(self.pipe.text_encoder.dtype),
            "latent_shape": list(self.shape), "guidance_scale": self.guidance,
            "steps": self.steps, "scheduler": dict(self.pipe.scheduler.config),
            "timestep_scale": float(self.pipe.transformer.config.timestep_scale),
            "time_orientation": "descending scheduler noise-to-image",
            "clean_caption": False, "complex_human_instruction": None,
        }
        self.signature = hashlib.sha256(json.dumps(self.metadata, sort_keys=True,
                                                   default=str).encode()).hexdigest()
        self.unconditional = self.encode("")

    def encode(self, prompt: str):
        with self.torch.inference_mode():
            embeds, mask, _, _ = self.pipe.encode_prompt(
                prompt=prompt, do_classifier_free_guidance=False,
                device=self.device, clean_caption=False,
                complex_human_instruction=None,
            )
        return embeds.to(self.pipe.transformer.dtype), mask

    def _forward(self, x, time, pair):
        embeddings, mask = pair
        batch = x.shape[0]
        t = time.reshape(1).expand(batch)
        t = t * self.pipe.transformer.config.timestep_scale
        pred = self.pipe.transformer(
            hidden_states=x.to(dtype=self.pipe.transformer.dtype),
            encoder_hidden_states=embeddings.expand(batch, -1, -1),
            encoder_attention_mask=mask.expand(batch, -1),
            timestep=t, return_dict=False,
        )[0].float()
        if pred.shape[1] == 2 * x.shape[1]:
            pred = pred.chunk(2, dim=1)[0]
        if pred.shape != x.shape:
            raise RuntimeError(f"Unexpected Sana field shape: {pred.shape}")
        return pred

    def velocity(self, x, time, pair):
        with self.torch.inference_mode():
            conditioned = self._forward(x, time, pair)
            if self.guidance == 1:
                return conditioned
            unconditioned = self._forward(x, time, self.unconditional)
            return unconditioned + self.guidance * (conditioned - unconditioned)


def bank_file(cfg):
    return resolved_path(cfg, "output") / "probes" / "probe_bank.pt"


def freeze_probes(cfg: dict) -> None:
    import torch
    path = bank_file(cfg)
    if path.exists():
        raise FileExistsError(f"Frozen probe bank exists: {path}")
    adapter = SanaAdapter(cfg)
    options = cfg["probes"]
    n_times, per_time = int(options["time_points"]), int(options["states_per_time"])
    if not 2 <= n_times <= adapter.steps or per_time < 1:
        raise ValueError("Invalid time/probe counts")
    indices = [round(k * (adapter.steps - 1) / (n_times - 1)) for k in range(n_times)]
    if len(set(indices)) != n_times:
        raise ValueError("Duplicate selected timestep indices")
    pool = {step: [] for step in indices}
    for position, row in enumerate(read_rows(cfg, "safe_probe_source")):
        pair = adapter.encode(row["prompt"])
        for repeat in range(int(options["trajectories_per_prompt"])):
            seed = int(cfg["experiment"]["seed"]) + 100003 * position + repeat
            generator = torch.Generator(device="cpu").manual_seed(seed)
            x = torch.randn((1, *adapter.shape), generator=generator).to(adapter.device)
            adapter.pipe.scheduler.set_timesteps(adapter.steps, device=adapter.device)
            for step, time in enumerate(adapter.pipe.scheduler.timesteps):
                if step in pool:
                    pool[step].append((x[0].detach().cpu().clone(), float(time),
                                       {"prompt_id": row["id"], "seed": seed,
                                        "scheduler_step": step}))
                predicted = adapter.velocity(x, time, pair)
                x = adapter.pipe.scheduler.step(predicted, time, x, return_dict=False)[0]
    chosen = []
    for step in indices:
        candidates = pool[step]
        if len(candidates) < per_time:
            raise ValueError("Insufficient safe source trajectories for requested probe count")
        ordering = sorted(range(len(candidates)), key=lambda k: hashlib.sha256(
            f"{cfg['experiment']['seed']}:{step}:{k}".encode()).hexdigest())
        chosen.extend(candidates[k] for k in ordering[:per_time])
    states, times, provenance = zip(*chosen)
    bank = {
        "states": torch.stack(states).float(),
        "times": torch.tensor(times, dtype=torch.float32),
        "weights": torch.full((len(states),), 1 / len(states)),
        "provenance": list(provenance), "metadata": adapter.metadata,
        "adapter_signature": adapter.signature,
        "split_sha256": sha256(resolved_path(cfg, "output") / "splits" / "safe_probe_source.jsonl"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)
    write_json(path.parent / "probe_bank_manifest.json", {
        "bank_sha256": sha256(path), "metadata": adapter.metadata,
        "probe_count": len(states), "provenance": list(provenance),
        "split_sha256": bank["split_sha256"],
    })
    print(f"Frozen {len(states)} safe-only probes: {path}")


def read_bank(cfg, adapter=None):
    import torch
    bank = torch.load(bank_file(cfg), map_location="cpu", weights_only=True)
    if adapter is not None and bank["adapter_signature"] != adapter.signature:
        raise ValueError("Model revision, scheduler, guidance or precision changed")
    source = resolved_path(cfg, "output") / "splits" / "safe_probe_source.jsonl"
    if bank["split_sha256"] != sha256(source):
        raise ValueError("Safe probe source split changed after probe construction")
    if len(bank["states"]) != len(bank["times"]) or len(bank["times"]) != len(bank["weights"]):
        raise ValueError("Invalid frozen bank dimensions")
    if not torch.isclose(bank["weights"].sum(), torch.tensor(1.0), atol=1e-6):
        raise ValueError("Invalid probe weights")
    return bank


def feature_file(cfg, split, prompt_id):
    return resolved_path(cfg, "output") / "features" / split / f"{prompt_id}.pt"


def extract(cfg: dict, split: str, cli_limit: int | None = None) -> None:
    import torch
    if split not in ("safe_reference", "safe_validation", "safe_test", "unsafe_test"):
        raise ValueError("Only reference, validation and test splits may be extracted")
    threshold = resolved_path(cfg, "output") / "scores" / "threshold.json"
    if split == "unsafe_test" and not threshold.exists():
        raise ValueError("Unsafe queries are evaluation-only: calibrate on safe validation first")
    adapter = SanaAdapter(cfg)
    bank = read_bank(cfg, adapter=adapter)
    bank_hash = sha256(bank_file(cfg))
    if split == "unsafe_test" and json.loads(threshold.read_text())["bank_sha256"] != bank_hash:
        raise ValueError("Unsafe extraction is gated by the CURRENT frozen threshold")
    subtract = bool(cfg["experiment"]["subtract_empty"])
    chunk_size = int(cfg["experiment"].get("probe_batch_size", 4))
    if chunk_size < 1:
        raise ValueError("probe_batch_size must be positive")

    def evaluate(pair):
        outputs = []
        # Run shared probes in original time-major order; do not batch mismatched timesteps.
        for start in range(0, len(bank["times"]), chunk_size):
            end = min(start + chunk_size, len(bank["times"]))
            for i in range(start, end):
                state = bank["states"][i:i+1].to(adapter.device)
                t = bank["times"][i:i+1].to(adapter.device)
                outputs.append(adapter.velocity(state, t, pair)[0].cpu())
        return torch.stack(outputs)

    empty_field = evaluate(adapter.unconditional) if subtract else None
    rows = limited_rows(cfg, split, cli_limit)
    for counter, row in enumerate(rows, 1):
        path = feature_file(cfg, split, row["id"])
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=True)
            if (old["bank_sha256"] != bank_hash
                    or old["adapter_signature"] != adapter.signature
                    or old["subtract_empty"] != subtract):
                raise ValueError(f"Stale cached feature: {path}")
            continue
        values = evaluate(adapter.encode(row["prompt"]))
        if empty_field is not None:
            values -= empty_field
        if not torch.isfinite(values).all():
            raise ValueError(f"Non-finite field for prompt {row['id']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"values": values.half(), "id": row["id"],
                    "bank_sha256": bank_hash, "adapter_signature": adapter.signature,
                    "subtract_empty": subtract}, temporary)
        os.replace(temporary, path)
        if counter % 10 == 0 or counter == len(rows):
            print(f"{split}: extracted {counter}/{len(rows)}")


def flat_feature(cfg, split, row, bank, probe_hash):
    import torch
    path = feature_file(cfg, split, row["id"])
    feature = torch.load(path, map_location="cpu", weights_only=True)
    if (feature["bank_sha256"] != probe_hash or
            feature["subtract_empty"] != bool(cfg["experiment"]["subtract_empty"]) or
            feature["values"].shape != bank["states"].shape):
        raise ValueError(f"Incompatible flow signature: {path}")
    return (feature["values"].float() * bank["weights"].sqrt()[:, None, None, None]).flatten()


def nearest(query, references):
    import torch
    return min(float(torch.linalg.vector_norm(chunk - query, dim=1).min())
               for chunk in references.split(16))


def score(cfg, split: str, calibrate=False) -> None:
    import torch
    if calibrate != (split == "safe_validation") or split not in (
            "safe_validation", "safe_test", "unsafe_test"):
        raise ValueError("Calibration must use ONLY safe_validation")
    folder = resolved_path(cfg, "output") / "scores"
    threshold_path = folder / "threshold.json"
    if not calibrate and not threshold_path.exists():
        raise FileNotFoundError("Calibrate before evaluating")
    bank = read_bank(cfg)
    probe_hash = sha256(bank_file(cfg))
    refs = limited_rows(cfg, "safe_reference")
    ids = [row["id"] for row in refs]
    reference = torch.stack([flat_feature(cfg, "safe_reference", row, bank, probe_hash)
                             for row in refs])
    if not calibrate:
        threshold = json.loads(threshold_path.read_text())
        if threshold["bank_sha256"] != probe_hash or threshold["reference_ids"] != ids:
            raise ValueError("Calibration was frozen with different probes/reference history")
    rows = limited_rows(cfg, split)
    results = [{"id": row["id"], "label": row["label"], "source": row["source"],
                "split": split, "distance": nearest(
                    flat_feature(cfg, split, row, bank, probe_hash), reference)}
               for row in rows]
    if calibrate:
        if threshold_path.exists():
            raise FileExistsError("Refusing to overwrite safe-only threshold")
        alpha = float(cfg["experiment"]["alpha"])
        if not 0 < alpha < 1:
            raise ValueError("0 < alpha < 1 required")
        observed = sorted(item["distance"] for item in results)
        rank = math.ceil((len(observed) + 1) * (1 - alpha)) - 1
        if rank >= len(observed):
            raise ValueError("Not enough safe calibration examples for a finite threshold")
        threshold = {"alpha": alpha, "threshold": observed[rank],
                     "n_calibration": len(observed), "bank_sha256": probe_hash,
                     "reference_ids": ids,
                     "meaning": "novelty relative to safe field history, not harmfulness"}
        write_json(threshold_path, threshold)
    for row in results:
        row["outside_safe_history"] = row["distance"] > threshold["threshold"]
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{split}.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"{split}: {len(results)} scores; threshold {threshold['threshold']:.5f}")


def report(cfg: dict) -> None:
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score
    folder = resolved_path(cfg, "output") / "scores"
    safe = pd.read_csv(folder / "safe_test.csv")
    unsafe = pd.read_csv(folder / "unsafe_test.csv")
    merged = pd.concat((safe, unsafe), ignore_index=True)
    truth = merged["label"].eq("risky").astype(int)
    threshold = json.loads((folder / "threshold.json").read_text())
    metrics = {
        "safe_test_n": len(safe), "unsafe_test_n": len(unsafe),
        "safe_false_positive_rate": float(safe["outside_safe_history"].mean()),
        "unsafe_flagged_rate": float(unsafe["outside_safe_history"].mean()),
        "auroc_prompt_labels": float(roc_auc_score(truth, merged["distance"])),
        "auprc_prompt_labels": float(average_precision_score(truth, merged["distance"])),
        "threshold": threshold["threshold"],
        "warning": "Prompt-category detection does not establish image-safety guarantees.",
    }
    write_json(folder / "report.json", metrics)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "probes", "extract", "calibrate", "evaluate", "report"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "safe_flow.toml")
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--limit", type=int, help="Temporary extraction cap, must match configured scoring caps")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        prepare(cfg)
    elif args.command == "probes":
        freeze_probes(cfg)
    elif args.command == "extract":
        if args.split is None:
            parser.error("extract requires --split")
        extract(cfg, args.split, args.limit)
    elif args.command == "calibrate":
        score(cfg, "safe_validation", calibrate=True)
    elif args.command == "evaluate":
        if args.split not in ("safe_test", "unsafe_test"):
            parser.error("evaluate requires --split safe_test or unsafe_test")
        score(cfg, args.split)
    else:
        report(cfg)


if __name__ == "__main__":
    main()
