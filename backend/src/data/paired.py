"""Paired legal/illegal prompt validation and detection-only decisions."""

import csv
from collections import Counter

from backend.src.data.definitions import SAFE_SPLITS, UNSAFE_SPLITS
from backend.src.data.prompts import normalize_prompt_text


def read_paired_detection_prompts(path, *, prompt_columns=None):
    if prompt_columns is None:
        prompt_columns = {"safe": "legal_prompt", "unsafe": "illegal_prompt"}
    if (
        set(prompt_columns) != {"safe", "unsafe"}
        or len(set(prompt_columns.values())) != 2
    ):
        raise ValueError("Paired prompts need distinct safe and unsafe columns")
    groups = {split: [] for split in SAFE_SPLITS + UNSAFE_SPLITS}
    split_names = {"train": "reference", "validation": "validation", "test": "test"}
    pairs, scenarios, prompts = set(), {}, set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            pair, scenario, split = row["pair_id"], row["scenario_id"], row["split"]
            if not pair or not scenario or pair in pairs or split not in split_names:
                raise ValueError("Invalid or duplicate pair ID, scenario, or split")
            pairs.add(pair)
            if scenario in scenarios and scenarios[scenario] != split:
                raise ValueError("Scenario crosses dataset splits")
            scenarios[scenario] = split
            for label, column in prompt_columns.items():
                prompt = row[column].strip()
                canonical = normalize_prompt_text(prompt)
                if not canonical or canonical in prompts:
                    raise ValueError("Empty or duplicate prompt in paired dataset")
                prompts.add(canonical)
                groups[f"{label}_{split_names[split]}"].append(
                    {
                        "id": f"{pair}_{label}",
                        "pair_id": pair,
                        "scenario_id": scenario,
                        "category": row["category"],
                        "variant_id": row["variant_id"],
                        "prompt": prompt,
                        "label": label,
                        "dataset_split": split,
                    }
                )
    if any(not rows for rows in groups.values()):
        raise ValueError("Every class must have nonempty train/validation/test splits")
    return groups, {
        "pairs": len(pairs),
        "prompts": len(prompts),
        "scenarios": dict(Counter(scenarios.values())),
        "scenario_disjoint": True,
        "pair_disjoint": True,
    }
