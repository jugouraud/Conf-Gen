"""Deterministic, category-stratified T2I partitions independent of safe caches."""

import hashlib
import json
from collections import Counter

from .prompt_records import canonical

CATEGORIES = ("Violence", "Pornography", "Illegal_Activities")
UNSAFE_SPLITS = ("unsafe_reference", "unsafe_validation", "unsafe_test")
SAFE_SPLITS = ("safe_reference", "safe_validation", "safe_test")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_unsafe_prompts(dataset, safe_rows, settings):
    payload = read_json(dataset)
    records = payload.values() if isinstance(payload, dict) else payload
    excluded = {canonical(r["prompt"]) for rows in safe_rows.values() for r in rows}
    pools = {category: [] for category in CATEGORIES}
    seen = set(excluded)
    # Assign multi-label rows once, preferring the least abundant risk category.
    for row in records:
        prompt = " ".join(str(row.get("prompt", "")).split())
        key = canonical(prompt)
        labels = row.get("label", {})
        matches = [c for c in reversed(CATEGORIES) if any(k.startswith(c + "--") for k in labels)]
        if not key or key in seen or not matches:
            continue
        seen.add(key)
        identifier = hashlib.sha256(key.encode("utf-8")).hexdigest()
        pools[matches[0]].append({"id": identifier, "prompt": prompt, "label": "risky",
                                  "source": "t2i_risky_prompt", "category": matches[0],
                                  "categories": sorted({k.split("--")[0] for k in labels}),
                                  "dataset_labels": labels})
    for pool in pools.values():
        pool.sort(key=lambda r: hashlib.sha256(f"{settings['seed']}:{r['id']}".encode()).hexdigest())
    groups = {}
    # Fill the equally sized anchors and tests first; allocate remaining category
    # capacity to calibration without reusing a prompt or reducing either set.
    for split, size in (("unsafe_reference", settings["unsafe_anchors"]),
                        ("unsafe_test", settings["unsafe_test"]),
                        ("unsafe_validation", settings["unsafe_calibration"])):
        chosen, counts = [], Counter()
        for _ in range(size):
            available = [c for c in CATEGORIES if pools[c]]
            if not available:
                raise ValueError("Insufficient distinct T2I prompts for the requested partitions")
            category = min(available, key=lambda c: (counts[c], CATEGORIES.index(c)))
            row = pools[category].pop()
            row.update(split=split, eligible_for_region=split != "unsafe_test",
                       evaluation_only=split == "unsafe_test")
            chosen.append(row)
            counts[category] += 1
        groups[split] = sorted(chosen, key=lambda r: r["id"])
    return groups
