"""Score every validation-folder caption against the saved COCO region."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.embeddings.define_region import DEFAULT_REGION_PATH
from backend.embeddings.project_region import DEFAULT_PROJECTION_PATH
from backend.paths import VALIDATION_ROOT
from frontend.coco_region_validation import (
    DEFAULT_PROMPT_MAPPING_PATH,
    DEFAULT_VALIDATION_DIRECTORY,
    LiveCocoRegionValidator,
    load_validation_items,
)

DEFAULT_REPORT_PATH = VALIDATION_ROOT / "runs" / "coco_region_blocking.json"


def _dataset_records(path: Path) -> dict[str, dict]:
    """Index dataset metadata by the image basename used in toxic_source."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    records = {}
    for item in payload:
        images = item.get("image", [])
        if isinstance(images, str):
            images = [images]
        for image in images:
            name = Path(str(image).replace("\\", "/")).name
            if name in records and records[name] != item:
                raise ValueError(f"Conflicting dataset records for {name}")
            records[name] = item
    return records


def _statistics(rows: list[dict]) -> dict:
    scores = np.asarray([row["distance"] for row in rows], dtype=np.float64)
    blocked = sum(row["blocked"] for row in rows)
    return {
        "count": len(rows), "blocked": blocked, "allowed": len(rows) - blocked,
        "block_rate": blocked / len(rows),
        "distance_min": float(scores.min()),
        "distance_p10": float(np.quantile(scores, 0.10)),
        "distance_median": float(np.median(scores)),
        "distance_p90": float(np.quantile(scores, 0.90)),
        "distance_max": float(scores.max()),
    }


def _group_statistics(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [{"category": category, **_statistics(group)} for category, group in sorted(groups.items())]


def analyze_validation_region(
    *, directory: str | Path = DEFAULT_VALIDATION_DIRECTORY,
    prompts_path: str | Path = DEFAULT_PROMPT_MAPPING_PATH,
    region_path: str | Path = DEFAULT_REGION_PATH,
    projection_path: str | Path = DEFAULT_PROJECTION_PATH,
    output_path: str | Path = DEFAULT_REPORT_PATH,
    device: str | None = None,
    batch_size: int = 16,
) -> dict:
    """Write image-level and unique-caption blocking statistics to JSON and CSV."""
    prompts_path = Path(prompts_path)
    items = load_validation_items(directory, prompts_path)
    if any(item.prompt_source != "Dataset caption" for item in items):
        raise ValueError("Every validation image needs a mapped dataset caption for this report.")
    dataset = _dataset_records(prompts_path)
    unique_prompts = list(dict.fromkeys(item.prompt for item in items))
    print(f"Encoding {len(unique_prompts)} distinct captions for {len(items)} images...", flush=True)
    validator = LiveCocoRegionValidator(region_path, projection_path, device=device)
    vectors = validator.encode_prompts(unique_prompts, batch_size=batch_size)
    scores = validator.region.score(vectors)
    score_by_prompt = dict(zip(unique_prompts, map(float, scores), strict=True))
    radius = float(validator.region.radius)
    rows = []
    for item in items:
        record = dataset.get(item.image_path.name, {})
        score = score_by_prompt[item.prompt]
        rows.append({
            "image": item.image_path.name,
            "caption": item.prompt,
            "image_category": str(record.get("image_category", "unknown")),
            "text_category": str(record.get("text_category", "unknown")),
            "base_category": str(record.get("base_category", "unknown")),
            "distance": score,
            "radius": radius,
            "margin": score - radius,
            "blocked": score > radius,
        })
    unique_rows = [next(row for row in rows if row["caption"] == prompt) for prompt in unique_prompts]
    report = {
        "metric": "full_768d_nearest_anchor_distance_of_clip_text_eos",
        "decision_input": "mapped_caption_text_not_image_pixels",
        "region": str(Path(region_path).resolve()),
        "image_directory": str(Path(directory).resolve()),
        "caption_source": str(prompts_path.resolve()),
        "radius": radius,
        "image_level": _statistics(rows),
        "unique_caption_level": _statistics(unique_rows),
        "by_image_category": _group_statistics(rows, "image_category"),
        "by_text_category": _group_statistics(rows, "text_category"),
        "by_base_category": _group_statistics(rows, "base_category"),
        "closest_to_boundary": sorted(rows, key=lambda row: abs(row["margin"]))[:12],
        "rows": rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Image-level: {report['image_level']['blocked']}/{len(rows)} blocked "
          f"({report['image_level']['block_rate']:.1%})")
    print(f"Unique captions: {report['unique_caption_level']['blocked']}/{len(unique_rows)} blocked "
          f"({report['unique_caption_level']['block_rate']:.1%})")
    print(f"Saved {output} and {output.with_suffix('.csv')}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze COCO-region blocking of validation-folder captions.")
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIRECTORY)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPT_MAPPING_PATH)
    parser.add_argument("--region", type=Path, default=DEFAULT_REGION_PATH)
    parser.add_argument("--projections", type=Path, default=DEFAULT_PROJECTION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    analyze_validation_region(
        directory=args.validation_dir, prompts_path=args.prompts,
        region_path=args.region, projection_path=args.projections,
        output_path=args.output, device=args.device, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
