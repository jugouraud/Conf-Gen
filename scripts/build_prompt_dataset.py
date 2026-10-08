"""Build the one-class prompt corpus from DiffusionDB and held-out benchmarks.

DiffusionDB is the only source eligible for region fitting or calibration.
The default experiment uses 10,000 safe prompts and 3,000 risky evaluation
rows total, balanced across I2P and T2I-RiskyPrompt. Run from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download

from backend.paths import DATA_ROOT, VALIDATION_ROOT

DIFFUSIONDB_REPO = "poloclub/diffusiondb"
RISKY_URL = (
    "https://raw.githubusercontent.com/datar001/"
    "T2I-RiskyPrompt/main/T2I-RiskyPrompt.json"
)
SPLITS = ("safe_reference", "safe_calibration", "safe_test")
BASE_COLUMNS = (
    "prompt_key", "prompt", "split", "label", "source", "category",
    "risk_score", "prompt_nsfw", "image_nsfw", "word_count",
    "eligible_for_region", "evaluation_only",
)


def normalize_prompt(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip() if value is not None else ""


def prompt_key(prompt: str) -> str:
    return hashlib.sha256(prompt.casefold().encode("utf-8")).hexdigest()


def _metadata_path(path: Path | None) -> Path:
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    return Path(hf_hub_download(
        repo_id=DIFFUSIONDB_REPO, filename="metadata.parquet", repo_type="dataset",
        local_dir=DATA_ROOT / "cache" / "diffusiondb",
    ))


def build_safe(
    metadata_path: Path, *, size: int, prompt_nsfw_max: float = 0.01,
    image_nsfw_max: float = 0.05, min_words: int = 3, max_words: int = 60,
) -> pd.DataFrame:
    """Keep the lowest-hash eligible prompts without loading 2M rows into RAM."""
    if size < 1 or min_words < 1 or min_words > max_words:
        raise ValueError("Invalid safe sample size or word-count limits")
    selected: dict[str, dict] = {}
    # A max heap represented by negative integer hashes lets us retain the
    # lowest hash values with stable results across source file row orders.
    heap: list[tuple[int, str]] = []
    parquet = pq.ParquetFile(metadata_path)
    for batch in parquet.iter_batches(
        batch_size=50_000, columns=["prompt", "prompt_nsfw", "image_nsfw"]
    ):
        prompts, prompt_scores, image_scores = (col.to_pylist() for col in batch.columns)
        for raw, prompt_score, image_score in zip(
            prompts, prompt_scores, image_scores, strict=True
        ):
            if (prompt_score is None or image_score is None or
                prompt_score > prompt_nsfw_max or image_score > image_nsfw_max):
                continue
            prompt = normalize_prompt(raw)
            words = len(prompt.split())
            if not min_words <= words <= max_words:
                continue
            key = prompt_key(prompt)
            if key in selected:
                old = selected[key]
                if (prompt_score, image_score) < (old["prompt_nsfw"], old["image_nsfw"]):
                    old.update(prompt=prompt, prompt_nsfw=prompt_score, image_nsfw=image_score)
                continue
            rank = int(key, 16)
            if len(heap) == size and rank >= -heap[0][0]:
                continue
            if len(heap) == size:
                _, removed_key = heapq.heappop(heap)
                del selected[removed_key]
            heapq.heappush(heap, (-rank, key))
            selected[key] = {
                "prompt_key": key, "prompt": prompt,
                "prompt_nsfw": prompt_score, "image_nsfw": image_score,
                "word_count": words,
            }
    if len(selected) < size:
        raise ValueError(f"Only {len(selected):,} safe prompts passed filtering; requested {size:,}")
    rows = [selected[key] for key in sorted(selected)]
    reference_end = round(size * 0.70)
    calibration_end = round(size * 0.85)
    for index, row in enumerate(rows):
        split = SPLITS[0 if index < reference_end else 1 if index < calibration_end else 2]
        row.update(
            split=split, label="safe", source="diffusiondb", category=None,
            risk_score=None, eligible_for_region=split != "safe_test",
            evaluation_only=split == "safe_test",
        )
    return pd.DataFrame(rows)


def build_i2p(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    rows = []
    for record in source.to_dict("records"):
        prompt = normalize_prompt(record.get("prompt"))
        if not prompt:
            continue
        rows.append({
            "prompt_key": prompt_key(prompt), "prompt": prompt,
            "split": "toxic_test_i2p", "label": "risky", "source": "i2p",
            "category": record.get("categories"),
            "risk_score": record.get("inappropriate_percentage"),
            "word_count": len(prompt.split()), "eligible_for_region": False,
            "evaluation_only": True,
            "i2p_prompt_toxicity": record.get("prompt_toxicity"),
        })
    return (pd.DataFrame(rows).sort_values("prompt_key", kind="stable")
            .drop_duplicates("prompt_key"))


def build_t2i_risky(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.values() if isinstance(payload, dict) else payload
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        prompt = normalize_prompt(record.get("prompt") or record.get("text") or record.get("caption"))
        if not prompt:
            continue
        rows.append({
            "prompt_key": prompt_key(prompt), "prompt": prompt,
            "split": "toxic_test_t2i_risky", "label": "risky",
            "source": "t2i_risky_prompt",
            "category": json.dumps(record.get("label"), ensure_ascii=False),
            "word_count": len(prompt.split()), "eligible_for_region": False,
            "evaluation_only": True,
        })
    return (pd.DataFrame(rows).sort_values("prompt_key", kind="stable")
            .drop_duplicates("prompt_key"))


def _risky_path(path: Path | None) -> Path:
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    destination = DATA_ROOT / "cache" / "T2I-RiskyPrompt.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        response = requests.get(RISKY_URL, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
    return destination


def build_corpus(args: argparse.Namespace) -> dict:
    safe_path = _metadata_path(args.diffusiondb_metadata)
    safe = build_safe(
        safe_path, size=args.safe_size, prompt_nsfw_max=args.prompt_nsfw_max,
        image_nsfw_max=args.image_nsfw_max, min_words=args.min_words,
        max_words=args.max_words,
    )
    evaluation = [build_i2p(args.i2p_csv)]
    if not args.skip_t2i_risky:
        evaluation.append(build_t2i_risky(_risky_path(args.t2i_risky_json)))
    # Keep the safe sample independent of evaluation data. Exclude matching
    # evaluation rows before taking the deterministic risky subsets.
    safe_keys = set(safe["prompt_key"])
    overlap_counts = {
        str(frame["source"].iloc[0]): int(frame["prompt_key"].isin(safe_keys).sum())
        for frame in evaluation
    }
    source_sizes = (
        [args.eval_size] if len(evaluation) == 1 else
        [args.eval_size // 2, args.eval_size - args.eval_size // 2]
    )
    evaluation = [
        frame.loc[~frame["prompt_key"].isin(safe_keys)].head(size)
        for frame, size in zip(evaluation, source_sizes, strict=True)
    ]
    if sum(len(frame) for frame in evaluation) != args.eval_size:
        raise ValueError("Not enough risky prompts to fill the requested evaluation sample")
    frames = [safe, *evaluation]
    corpus = pd.concat(frames, ignore_index=True)
    eligible = corpus["eligible_for_region"]
    if not eligible.eq(corpus["split"].isin(SPLITS[:2])).all():
        raise AssertionError("Region eligibility does not match the safe splits")
    if not corpus.loc[eligible, "label"].eq("safe").all():
        raise AssertionError("Risky prompt marked eligible for region construction")
    if set(safe["prompt_key"]) & set(pd.concat(evaluation)["prompt_key"]):
        raise AssertionError("Evaluation prompt overlaps a safe prompt")
    if corpus["prompt_key"].duplicated().any():
        raise ValueError("Duplicate prompt across corpus sources or splits")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    columns = [*BASE_COLUMNS, *(c for c in corpus if c not in BASE_COLUMNS)]
    corpus = corpus.reindex(columns=columns)
    corpus.to_parquet(output / "confgen_prompts.parquet", index=False)
    corpus.to_csv(output / "confgen_prompts.csv", index=False)
    manifest = {
        "safe_source": DIFFUSIONDB_REPO,
        "safe_metadata": str(safe_path.resolve()),
        "safe_size_requested": args.safe_size,
        "evaluation_size_total_requested": args.eval_size,
        "safe_filters": {
            "prompt_nsfw_max": args.prompt_nsfw_max,
            "image_nsfw_max": args.image_nsfw_max,
            "min_words": args.min_words, "max_words": args.max_words,
        },
        "split_counts": corpus["split"].value_counts().to_dict(),
        "source_counts": corpus["source"].value_counts().to_dict(),
        "evaluation_rows_excluded_for_safe_overlap": overlap_counts,
        "region_rule": "Only safe_reference and safe_calibration are eligible for region construction; safe_test and all risky sources are evaluation-only.",
    }
    (output / "confgen_prompt_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "prompts")
    parser.add_argument("--diffusiondb-metadata", type=Path)
    parser.add_argument("--i2p-csv", type=Path, default=VALIDATION_ROOT / "i2p_benchmark.csv")
    parser.add_argument("--t2i-risky-json", type=Path)
    parser.add_argument("--safe-size", type=int, default=10_000)
    parser.add_argument("--eval-size", type=int, default=3_000,
                        help="Total risky evaluation rows, balanced across sources")
    parser.add_argument("--prompt-nsfw-max", type=float, default=0.01)
    parser.add_argument("--image-nsfw-max", type=float, default=0.05)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=60)
    parser.add_argument("--skip-t2i-risky", action="store_true")
    args = parser.parse_args()
    if args.eval_size < 1:
        parser.error("--eval-size must be positive")
    print(json.dumps(build_corpus(args)["split_counts"], indent=2))


if __name__ == "__main__":
    main()
