"""Shared prompt normalization, partition transformation, and frozen-row access."""

import json
import re
import unicodedata

from .common import SAFE_INPUT, SPLITS, resolved_path

SOURCE_SPLIT_ORDER = {"safe_reference": 0, "safe_calibration": 1, "safe_test": 2}
SAFE_TARGET_SPLITS = {
    "safe_reference": "safe_reference",
    "safe_calibration": "safe_validation",
    "safe_test": "safe_test",
}


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
