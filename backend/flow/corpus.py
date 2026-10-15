"""Validate and freeze the safe-only experiment partitions."""

import json
import re
import unicodedata

import pandas as pd

from .common import SAFE_INPUT, SPLITS, resolved_path, sha256, write_json

SOURCE_SPLIT_ORDER = {"safe_reference": 0, "safe_calibration": 1, "safe_test": 2}
SAFE_TARGET_SPLITS = {
    "safe_reference": "safe_reference",
    "safe_calibration": "safe_validation",
    "safe_test": "safe_test",
}


# Hierarchy 3: atomic actions.

def canonical(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # Unicode symbols include emoji, which can be the entire prompt. Keep
    # them when detecting duplicates instead of turning valid prompts empty.
    characters = (
        char if unicodedata.category(char)[0] in "LMNS" else " "
        for char in normalized
    )
    return re.sub(r"\s+", " ", "".join(characters)).strip()


def _validate_source_row(row: dict) -> bool:
    """Enforce the safe-only fitting policy before removing duplicates."""
    source_split = str(row["split"])
    is_safe = source_split in SAFE_INPUT
    if is_safe:
        valid = (
            row["label"] == "safe"
            and row["source"] == "diffusiondb"
            and bool(row["eligible_for_region"]) == (source_split != "safe_test")
            and bool(row["evaluation_only"]) == (source_split == "safe_test")
        )
        if not valid:
            raise ValueError(f"Invalid safe split metadata: {source_split}")
    else:
        valid = (
            row["label"] == "risky"
            and source_split.startswith("toxic_test")
            and not bool(row["eligible_for_region"])
            and bool(row["evaluation_only"])
        )
        if not valid:
            raise ValueError(f"Risky/evaluation-only constraint violated: {source_split}")
    return is_safe


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


def _load_corpus(cfg):
    return pd.read_parquet(resolved_path(cfg, "corpus"))


def _split_manifest(cfg, result, dropped):
    corpus = resolved_path(cfg, "corpus")
    return {
        "corpus": str(corpus), "corpus_sha256": sha256(corpus),
        "seed": cfg["experiment"]["seed"], "canonical_duplicates_dropped": dropped,
        "counts": {name: len(rows) for name, rows in result.items()},
        "policy": "The same DiffusionDB safe fit set supplies probes/reference; "
                  "safe calibration supplies threshold; all risky rows evaluation only",
    }


def _has_identical_manifest(cfg, manifest):
    path = resolved_path(cfg, "output") / "splits" / "manifest.json"
    if not path.exists():
        return False
    if json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise FileExistsError("Split manifest already exists and differs; use a new output directory")
    print(f"Unchanged frozen split manifest: {path}")
    return True


def _require_clean_output(cfg):
    root = resolved_path(cfg, "output")
    if (root / "probes").exists() or (root / "features").exists():
        raise FileExistsError("Existing probes/features require a new output directory")


def _write_split_rows(cfg, result):
    folder = resolved_path(cfg, "output") / "splits"
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in result.items():
        with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_split_manifest(cfg, manifest):
    path = resolved_path(cfg, "output") / "splits" / "manifest.json"
    write_json(path, manifest)
    print(f"Prepared disjoint safe-only splits: {manifest['counts']}")


# Hierarchy 2: subworkflows.

def split_corpus(frame):
    """Freeze disjoint fit, calibration and evaluation prompts."""
    required = {"prompt_key", "prompt", "split", "label", "source",
                "eligible_for_region", "evaluation_only"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing corpus fields: {sorted(required - set(frame.columns))}")
    rows = frame.to_dict("records")
    rows.sort(key=lambda r: (SOURCE_SPLIT_ORDER.get(str(r["split"]), 3),
                             str(r["prompt_key"])))
    result = {name: [] for name in SPLITS}
    seen_keys, seen_text = set(), set()
    dropped = 0
    for row in rows:
        previous = str(row["split"])
        safe = _validate_source_row(row)
        key, prompt = str(row["prompt_key"]), str(row["prompt"]).strip()
        norm = canonical(prompt)
        if not norm:
            raise ValueError("Empty prompt in corpus")
        if key in seen_keys or norm in seen_text:
            dropped += 1
            continue
        seen_keys.add(key)
        seen_text.add(norm)
        target = SAFE_TARGET_SPLITS[previous] if safe else "unsafe_test"
        result[target].append({"id": key, "prompt": prompt,
                               "label": "safe" if safe else "risky",
                               "source": str(row["source"]), "source_split": previous})
    if not result["safe_reference"]:
        raise ValueError("Missing safe fit examples")
    for values in result.values():
        values.sort(key=lambda item: item["id"])
    if not result["safe_validation"] or not result["safe_test"] or not result["unsafe_test"]:
        raise ValueError("Missing required safe validation or held-out test examples")
    return result, dropped


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


def _partition_corpus(cfg, frame):
    return split_corpus(frame)


# Hierarchy 1: workflow orchestration.

def prepare(cfg: dict) -> None:
    """Freeze disjoint safe-only split files from the prompt corpus."""
    frame = _load_corpus(cfg)
    result, dropped = _partition_corpus(cfg, frame)
    manifest = _split_manifest(cfg, result, dropped)
    if _has_identical_manifest(cfg, manifest):
        return
    _require_clean_output(cfg)
    _write_split_rows(cfg, result)
    _write_split_manifest(cfg, manifest)
