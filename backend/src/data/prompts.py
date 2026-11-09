"""Prompt normalization, split loading, and unsafe partition selection."""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from backend.src.common.storage import read_json, resolve_config_path
from backend.src.data.definitions import SPLITS


def select_detected_unsafe_repair_cases(
    groups: dict[str, list[dict]],
    distances: dict[str, np.ndarray],
    calibration: dict,
    options: dict,
    stage: str,
) -> list[tuple[str, int, dict]]:
    """Select a fixed category-balanced cohort flagged in every configured mode."""
    split = "unsafe_validation" if stage == "tune" else "unsafe_test"
    limit = options["tuning_per_category" if stage == "tune" else "pilot_per_category"]
    safe = distances[f"safe__{split}"].min(axis=1)
    unsafe = distances[f"unsafe__{split}"].min(axis=1)
    flagged = safe > calibration["safe_cutoff"]
    # Requiring the safe-only flag also guarantees the combined-mode flag.
    if "safe_attraction" not in options["modes"]:
        flagged = flagged | (unsafe <= calibration["unsafe_cutoff"])
    rng = np.random.default_rng(options["seed"])
    cases = []
    categories = options.get(
        "categories", tuple(dict.fromkeys(row["category"] for row in groups[split]))
    )
    for category in categories:
        indexes = [
            i
            for i, row in enumerate(groups[split])
            if row["category"] == category and flagged[i]
        ]
        if len(indexes) < limit:
            raise ValueError(
                f"Need {limit} detected-unsafe prompts in {split}/{category}; found {len(indexes)}"
            )
        rng.shuffle(indexes)
        cases.extend((split, i, groups[split][i]) for i in sorted(indexes[:limit]))
    return cases


def normalize_prompt_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # Unicode symbols include emoji, which can be the entire prompt. Keep
    # them when detecting duplicates instead of turning valid prompts empty.
    characters = (
        char if unicodedata.category(char)[0] in "LMNS" else " " for char in normalized
    )
    return re.sub(r"\s+", " ", "".join(characters)).strip()


def read_prompt_split(cfg: dict, split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    path = resolve_config_path(cfg, "output") / "splits" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or any(
        (row["label"] == "risky") != (split == "unsafe_test") for row in rows
    ):
        raise ValueError(f"Invalid or empty split: {path}")
    return rows


def read_limited_prompt_split(
    cfg: dict, split: str, limit: int | None = None
) -> list[dict]:
    rows = read_prompt_split(cfg, split)
    caps = {
        "safe_reference": "reference_limit",
        "safe_validation": "validation_limit",
        "safe_test": "test_limit",
        "unsafe_test": "test_limit",
    }
    cap = int(cfg["experiment"].get(caps.get(split, ""), 0)) if split in caps else 0
    if cap > 0:
        rows = rows[:cap]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    return rows


def select_unsafe_prompts(
    dataset: Path, safe_rows: dict[str, list[dict]], settings: dict
) -> dict[str, list[dict]]:
    payload = read_json(dataset)
    records = list(payload.values()) if isinstance(payload, dict) else payload
    categories = settings.get(
        "categories",
        tuple(
            dict.fromkeys(
                label.split("--", 1)[0]
                for row in records
                for label in row.get("label", {})
            )
        ),
    )
    excluded = {
        normalize_prompt_text(r["prompt"]) for rows in safe_rows.values() for r in rows
    }
    pools = {category: [] for category in categories}
    seen = set(excluded)
    # Assign multi-label rows once, preferring the least abundant risk category.
    for row in records:
        prompt = " ".join(str(row.get("prompt", "")).split())
        key = normalize_prompt_text(prompt)
        labels = row.get("label", {})
        matches = [
            c
            for c in reversed(categories)
            if any(k.startswith(c + "--") for k in labels)
        ]
        if not key or key in seen or not matches:
            continue
        seen.add(key)
        identifier = hashlib.sha256(key.encode("utf-8")).hexdigest()
        pools[matches[0]].append(
            {
                "id": identifier,
                "prompt": prompt,
                "label": "risky",
                "source": settings.get("source_name", dataset.stem),
                "category": matches[0],
                "categories": sorted({k.split("--")[0] for k in labels}),
                "dataset_labels": labels,
            }
        )
    for pool in pools.values():
        pool.sort(
            key=lambda r: hashlib.sha256(
                f"{settings['seed']}:{r['id']}".encode()
            ).hexdigest()
        )
    groups = {}
    # Fill the equally sized anchors and tests first; allocate remaining category
    # capacity to calibration without reusing a prompt or reducing either set.
    for split, size in (
        ("unsafe_reference", settings["unsafe_anchors"]),
        ("unsafe_test", settings["unsafe_test"]),
        ("unsafe_validation", settings["unsafe_calibration"]),
    ):
        chosen, counts = [], Counter()
        for _ in range(size):
            available = [c for c in categories if pools[c]]
            if not available:
                raise ValueError(
                    "Insufficient distinct prompts for the requested partitions"
                )
            category = min(available, key=lambda c: (counts[c], categories.index(c)))
            row = pools[category].pop()
            row.update(
                split=split,
                eligible_for_region=split != "unsafe_test",
                evaluation_only=split == "unsafe_test",
            )
            chosen.append(row)
            counts[category] += 1
        groups[split] = sorted(chosen, key=lambda r: r["id"])
    return groups
